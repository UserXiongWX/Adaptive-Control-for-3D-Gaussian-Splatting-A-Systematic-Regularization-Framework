import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from simple_knn import _C as simple_knn_C
    KNN_CUDA_AVAILABLE = True
    print("Successfully imported simple_knn._C for CUDA KNN.")
except ImportError:
    print("WARNING: simple_knn._C could not be imported. CUDA KNN for normal smoothness will not be available.")
    KNN_CUDA_AVAILABLE = False

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.small_gaussian = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._knn_indices_cache_normal = None
        self._knn_cache_iter_normal = -1
        self._anchor_normal = None
        self._normal_is_trusted = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._normal,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args, finetune):
        (self.active_sh_degree, 
        self._xyz,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args

        self._opacity.data = torch.cat([self._opacity.data] * 2, dim=1)

        self._normal = torch.ones_like(self._xyz) * 0.6

        self.training_setup(training_args, finetune=finetune)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        return self._normal
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda"))
        if pcd.normals is not None and pcd.normals.shape == pcd.points.shape:
            print("Initializing Gaussians' cutting plane normals from PCD normals.")
            normal_param_init_raw = torch.tensor(np.asarray(pcd.normals)).float().cuda()
            normal_param_init = F.normalize(normal_param_init_raw, p=2, dim=1)
            self._anchor_normal = normal_param_init.detach().clone() 
        else:
            print("PCD normals not available or incompatible. Initializing cutting plane normals to default.")
            default_normal_vec = torch.tensor([0.6, 0.6, 0.6], device="cuda")
            normal_param_init = F.normalize(default_normal_vec, p=2, dim=0).unsqueeze(0).repeat(fused_point_cloud.shape[0], 1)
            self._anchor_normal = normal_param_init.detach().clone()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(normal_param_init.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, normal_lr=0.003, finetune=False):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")        

        if finetune:
            l = [
                {'params': [self._normal], 'lr': 0.003, "name": "normal"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._normal], 'lr': 0.003, "name": "normal"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        self.knn_recompute_interval_normal = getattr(training_args, 'knn_recompute_interval', 1000)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity1')
        l.append('opacity2')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normal, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.02))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def generate_small_cov(self, cov_input, n):
        cov_input = cov_input.to(torch.float64)
        n = n.to(torch.float64)

        num = cov_input.shape[0]

        n_norm = torch.norm(n, dim=1, keepdim=True)
        n = n / n_norm

        device = n.device

        cov = torch.zeros((num, 3, 3), dtype=torch.float64).to(device)
        cov[:, 0, 0] = cov_input[:, 0]
        cov[:, 0, 1] = cov[:, 1, 0] = cov_input[:, 1]
        cov[:, 0, 2] = cov[:, 2, 0] = cov_input[:, 2]
        cov[:, 1, 1] = cov_input[:, 3]
        cov[:, 1, 2] = cov[:, 2, 1] = cov_input[:, 4]
        cov[:, 2, 2] = cov_input[:, 5]

        v1 = torch.zeros((num, 3), dtype=torch.float64).to(device)
        v2 = torch.zeros((num, 3), dtype=torch.float64).to(device)

        zero_indices = (n[:, 0] == 0) & (n[:, 1] == 0)

        v1[zero_indices] = torch.tensor([1, 0, 0], dtype=torch.float64).to(device)
        v2[zero_indices] = torch.tensor([0, 1, 0], dtype=torch.float64).to(device)

        v1[~zero_indices] = torch.stack(
            [n[~zero_indices][:, 1], -n[~zero_indices][:, 0], torch.zeros_like(n[~zero_indices][:, 0]).to(device)],
            dim=1)
        v1[~zero_indices] = v1[~zero_indices] / torch.norm(v1[~zero_indices], dim=1, keepdim=True)
        v2[~zero_indices] = torch.cross(n[~zero_indices], v1[~zero_indices])
        v2[~zero_indices] = v2[~zero_indices] / torch.norm(v2[~zero_indices], dim=1, keepdim=True)

        R_transform = torch.stack([v1, v2, n], dim=2)
        basis = R_transform.transpose(1, 2)

        cov_transformed = torch.matmul(basis, torch.matmul(cov, R_transform))
        cov_inv2 = torch.linalg.inv(cov_transformed)

        twoD_mat = cov_inv2[:, :2, :2]

        eig_value, eig_vector = torch.linalg.eig(twoD_mat)
        eig_value = eig_value.real
        eig_vector = eig_vector.real
        eig_value_sorted, indices = torch.sort(eig_value, descending=False)
        eig_vector_sorted = torch.gather(eig_vector, 2, indices.unsqueeze(1).expand(-1, eig_vector.size(1), -1))

        lambda1 = eig_value_sorted[:, 0]
        lambda2 = eig_value_sorted[:, 1]

        v1_2d = eig_vector_sorted[:, :, 0]
        v2_2d = eig_vector_sorted[:, :, 1]

        v1_2d = torch.stack([v1_2d[:, 0], v1_2d[:, 1], torch.zeros_like(v1_2d[:, 0])], dim=1)
        v2_2d = torch.stack([v2_2d[:, 0], v2_2d[:, 1], torch.zeros_like(v2_2d[:, 0])], dim=1)

        v3_2d = torch.tensor([0, 0, 1], dtype=torch.float64).expand(num, -1).to(device)

        v_3d = torch.stack([v1_2d, v2_2d, v3_2d], dim=2)

        lam_mat = torch.zeros((num, 3, 3), dtype=torch.float64).to(device)

        lam_mat[:, 0, 0] = 1 / lambda1
        lam_mat[:, 1, 1] = 1 / lambda2
        lam_mat[:, 2, 2] = 0.001 / torch.max(lambda1, lambda2)

        cov_new_3d = torch.matmul(v_3d, torch.matmul(lam_mat, v_3d.transpose(1, 2)))
        cov_new_3d = torch.matmul(R_transform, torch.matmul(cov_new_3d, basis))

        cov_six_elements = torch.zeros((cov_new_3d.shape[0], 6), dtype=torch.float64, device=cov_new_3d.device)
        cov_six_elements[:, 0] = cov_new_3d[:, 0, 0]
        cov_six_elements[:, 1] = cov_new_3d[:, 0, 1]
        cov_six_elements[:, 2] = cov_new_3d[:, 0, 2]
        cov_six_elements[:, 3] = cov_new_3d[:, 1, 1]
        cov_six_elements[:, 4] = cov_new_3d[:, 1, 2]
        cov_six_elements[:, 5] = cov_new_3d[:, 2, 2]

        return cov_six_elements.float()

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        if "opacity1" in plydata.elements[0]:
            normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
            opacities1 = np.asarray(plydata.elements[0]["opacity1"])
            opacities2 = np.asarray(plydata.elements[0]["opacity2"])
        else:
            normal = np.ones(np.shape(xyz))*0.6

            opacities1 = np.asarray(plydata.elements[0]["opacity"])
            opacities2 = np.asarray(plydata.elements[0]["opacity"])

        opacities = np.stack((opacities1,opacities2), axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.small_gaussian = self.generate_small_cov(self.get_covariance(), self._normal)


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self._anchor_normal is not None:
            self._anchor_normal = self._anchor_normal[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._knn_indices_cache_normal = None

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_anchor_normal):
        d = {"xyz": new_xyz,
        "normal": new_normal,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self._anchor_normal is not None:
            self._anchor_normal = torch.cat((self._anchor_normal, new_anchor_normal), dim=0)
        else:
            self._anchor_normal = new_anchor_normal

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._knn_indices_cache_normal = None 

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points_before_split = self.get_xyz.shape[0]
        if n_init_points_before_split == 0:
            return

        padded_grad = torch.zeros((n_init_points_before_split), device="cuda")
        current_grad_len = grads.shape[0]
        if current_grad_len > 0:
            if current_grad_len == n_init_points_before_split:
                 padded_grad = grads.squeeze()
            elif current_grad_len < n_init_points_before_split:
                 padded_grad[:current_grad_len] = grads.squeeze()
                 print(f"Warning: Grads length ({current_grad_len}) is less than n_init_points_before_split ({n_init_points_before_split}) in densify_and_split.")
            else:
                 print(f"Error: Grads length ({current_grad_len}) is greater than n_init_points_before_split ({n_init_points_before_split}). This should not happen.")
                 return


        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        if self._scaling.numel() > 0 and self._scaling.shape[0] == n_init_points_before_split:
            activated_scales_for_check = torch.exp(self._scaling)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(activated_scales_for_check, dim=1).values > self.percent_dense * scene_extent)
        elif self._scaling.numel() > 0:
             print(f"Warning: Scaling shape {self._scaling.shape} mismatch with current points {n_init_points_before_split} in densify_and_split. Skipping scale check.")

        if not torch.any(selected_pts_mask):
            return

        parent_xyz_selected = self._xyz[selected_pts_mask]
        parent_normal_param_selected = self._normal[selected_pts_mask]
        parent_rotation_q_selected = self._rotation[selected_pts_mask]
        parent_scaling_log_selected = self._scaling[selected_pts_mask]
        parent_opacity_logits_selected = self._opacity[selected_pts_mask]
        parent_features_dc_selected = self._features_dc[selected_pts_mask]
        parent_features_rest_selected = self._features_rest[selected_pts_mask]
        
        parent_anchor_normal_selected = None
        if hasattr(self, '_anchor_normal') and self._anchor_normal is not None and \
           self._anchor_normal.shape[0] == n_init_points_before_split:
            parent_anchor_normal_selected = self._anchor_normal[selected_pts_mask]

        num_parents_to_split = selected_pts_mask.sum().item()
        if num_parents_to_split == 0:
            return

        all_new_xyz_list = []
        
        for i in range(num_parents_to_split):
            current_parent_xyz = parent_xyz_selected[i]
            current_parent_n_param = parent_normal_param_selected[i]
            current_parent_rot_q = parent_rotation_q_selected[i].unsqueeze(0)
            current_parent_scales_log = parent_scaling_log_selected[i]
            
            n_c_world = F.normalize(current_parent_n_param, p=2, dim=0)
            R_e2w = build_rotation(current_parent_rot_q).squeeze(0)
            activated_parent_scales = torch.exp(current_parent_scales_log)

            scaled_ellipse_axes_world = torch.zeros((3, 3), device=self._xyz.device, dtype=torch.float)
            scaled_ellipse_axes_world[0, :] = R_e2w[:, 0] * activated_parent_scales[0]
            scaled_ellipse_axes_world[1, :] = R_e2w[:, 1] * activated_parent_scales[1]
            scaled_ellipse_axes_world[2, :] = R_e2w[:, 2] * activated_parent_scales[2]
            
            dot_prods_with_n = torch.sum(scaled_ellipse_axes_world * n_c_world.unsqueeze(0), dim=1)
            projected_scaled_axes_on_plane = scaled_ellipse_axes_world - dot_prods_with_n.unsqueeze(1) * n_c_world.unsqueeze(0)
            projected_lengths_sq = torch.sum(projected_scaled_axes_on_plane**2, dim=1)
            
            split_direction_in_plane_world = torch.zeros_like(n_c_world)
            split_magnitude = 0.0

            if torch.all(projected_lengths_sq < 1e-8): 
                abs_n_coords = torch.abs(n_c_world)
                min_idx_n = torch.argmin(abs_n_coords)
                temp_vec_n = torch.zeros_like(n_c_world)
                temp_vec_n[min_idx_n] = 1.0
                split_direction_candidate = torch.cross(n_c_world, temp_vec_n)
                if torch.norm(split_direction_candidate) < 1e-6 :
                    temp_vec_n_alt = torch.zeros_like(n_c_world)
                    temp_vec_n_alt[(min_idx_n + 1)%3] = 1.0
                    split_direction_candidate = torch.cross(n_c_world, temp_vec_n_alt)
                split_direction_in_plane_world = F.normalize(split_direction_candidate, p=2, dim=0)
                split_magnitude = torch.max(activated_parent_scales) * 0.6
            else:
                longest_proj_idx = torch.argmax(projected_lengths_sq)
                if projected_lengths_sq[longest_proj_idx] > 1e-8 :
                    split_direction_in_plane_world = F.normalize(projected_scaled_axes_on_plane[longest_proj_idx], p=2, dim=0)
                else: 
                    abs_n_coords = torch.abs(n_c_world)
                    min_idx_n = torch.argmin(abs_n_coords)
                    temp_vec_n = torch.zeros_like(n_c_world)
                    temp_vec_n[min_idx_n] = 1.0
                    split_direction_candidate = torch.cross(n_c_world, temp_vec_n)
                    if torch.norm(split_direction_candidate) < 1e-6:
                        temp_vec_n_alt = torch.zeros_like(n_c_world)
                        temp_vec_n_alt[(min_idx_n + 1)%3] = 1.0
                        split_direction_candidate = torch.cross(n_c_world, temp_vec_n_alt)
                    split_direction_in_plane_world = F.normalize(split_direction_candidate, p=2, dim=0)
                split_magnitude = activated_parent_scales[longest_proj_idx] * 0.6

            offset_vector = split_direction_in_plane_world * split_magnitude * 0.5 
            new_xyz_child1 = current_parent_xyz + offset_vector
            new_xyz_child2 = current_parent_xyz - offset_vector
            all_new_xyz_list.append(new_xyz_child1.unsqueeze(0))
            all_new_xyz_list.append(new_xyz_child2.unsqueeze(0))

        new_xyz = torch.cat(all_new_xyz_list, dim=0)
        num_new_children = new_xyz.shape[0]

        new_normal_params = parent_normal_param_selected.repeat_interleave(N, dim=0)
        new_rotation_q = parent_rotation_q_selected.repeat_interleave(N, dim=0)
        new_features_dc = parent_features_dc_selected.repeat_interleave(N, dim=0)
        new_features_rest = parent_features_rest_selected.repeat_interleave(N, dim=0)
        new_opacity_logits = parent_opacity_logits_selected.repeat_interleave(N, dim=0)

        parent_activated_scales_repeated = torch.exp(parent_scaling_log_selected).repeat_interleave(N, dim=0)
        scale_down_factor = 1.6 
        new_activated_scales = parent_activated_scales_repeated / scale_down_factor
        new_scaling_log = self.scaling_inverse_activation(torch.clamp_min(new_activated_scales, 1e-8))

        new_anchor_normal_for_children = None
        if hasattr(self, '_anchor_normal') and self._anchor_normal is not None:
            if parent_anchor_normal_selected is not None and parent_anchor_normal_selected.shape[0] == num_parents_to_split:
                new_anchor_normal_for_children = parent_anchor_normal_selected.repeat_interleave(N, dim=0)
            else:
                new_anchor_normal_for_children = new_normal_params.detach().clone()

        self.densification_postfix(
            new_xyz, new_normal_params, 
            new_features_dc, new_features_rest, 
            new_opacity_logits, new_scaling_log, new_rotation_q,
            new_anchor_normal_for_children
        )
        
        mask_for_pruning_current_array = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        mask_for_pruning_current_array[:n_init_points_before_split][selected_pts_mask] = True
        self.prune_points(mask_for_pruning_current_array)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_opacities = inverse_sigmoid(
            1 - torch.sqrt(1 - torch.sigmoid(new_opacities)))
        self._opacity[selected_pts_mask].copy_(new_opacities.detach())
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self._anchor_normal is not None:
            new_anchor_normal_clone = self._anchor_normal[selected_pts_mask]
        else:
            new_anchor_normal_clone = self._normal[selected_pts_mask].detach().clone()

        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_anchor_normal_clone)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        max_elements, _ = torch.max(self.get_opacity,dim=1)

        prune_mask = (max_elements < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def get_normal_smoothness_loss(self, k, current_iteration):
        num_points = self._xyz.shape[0]
        if not KNN_CUDA_AVAILABLE or num_points < k + 1: 
            return torch.tensor(0.0, device=self._xyz.device)

        if self._knn_indices_cache_normal is None or \
            current_iteration // self.knn_recompute_interval_normal != self._knn_cache_iter_normal // self.knn_recompute_interval_normal or \
            self._knn_indices_cache_normal.shape[0] != num_points or \
            self._knn_indices_cache_normal.shape[1] != k:

            print(f"[Iteration {current_iteration}] Recomputing KNN for normal smoothness (k={k})...")
            points_xyz_for_knn = self._xyz.detach()

            try:

                neighbor_indices, _ = simple_knn_C.knn_indices_cuda(points_xyz_for_knn, k)
                self._knn_indices_cache_normal = neighbor_indices.long()
                self._knn_cache_iter_normal = current_iteration
                print("KNN for normal smoothness recomputation done.")
            except Exception as e:
                print(f"ERROR during CUDA KNN for normal smoothness: {e}. Skipping loss term.")
                self._knn_indices_cache_normal = None
                return torch.tensor(0.0, device=self._xyz.device)
        
        if self._knn_indices_cache_normal is None:
            return torch.tensor(0.0, device=self._xyz.device)

        knn_indices = self._knn_indices_cache_normal
        activated_normals = F.normalize(self._normal, p=2, dim=1)
        num_points = self._xyz.shape[0]

        valid_neighbor_mask = (knn_indices >= 0) & (knn_indices < num_points)

        row_indices = torch.arange(num_points, device=self._xyz.device).unsqueeze(1).expand(-1, k)

        flat_row_indices_all_pairs = row_indices.reshape(-1)
        flat_col_indices_all_pairs = knn_indices.reshape(-1)
        flat_valid_mask = valid_neighbor_mask.reshape(-1)

        valid_row_indices = flat_row_indices_all_pairs[flat_valid_mask]
        valid_col_indices = flat_col_indices_all_pairs[flat_valid_mask]

        if valid_row_indices.numel() == 0:
            return torch.tensor(0.0, device=self._xyz.device)

        normals_i = activated_normals[valid_row_indices]
        normals_j = activated_normals[valid_col_indices]

        dot_products = torch.sum(normals_i * normals_j, dim=1)
        loss = (1.0 - dot_products).pow(2).mean()

        return loss
    

    def get_adaptive_opacity_consistency_loss(self, visibility_filter, radii):
        current_opacities = self.get_opacity
        num_points = self._xyz.shape[0]

        if current_opacities.shape[0] != num_points or \
           visibility_filter.shape[0] != num_points or \
           radii.shape[0] != num_points:
            print(f"Warning: Shape mismatch in get_adaptive_opacity_consistency_loss. Skipping. "
                  f"Opacities: {current_opacities.shape}, Visibility: {visibility_filter.shape}, Radii: {radii.shape}, Points: {num_points}")
            return torch.tensor(0.0, device=self._xyz.device)

        if current_opacities.shape[1] != 2 or not torch.any(visibility_filter):
            return torch.tensor(0.0, device=self._xyz.device)
        
        quantile_val = getattr(self, '_smooth_region_quantile', 0.65)

        visible_indices = torch.where(visibility_filter)[0]
        if visible_indices.numel() == 0:
             return torch.tensor(0.0, device=self._xyz.device)
        
        radii_of_visible = radii[visible_indices]

        if radii_of_visible.numel() < 2 :
            if radii_of_visible.numel() == 1 and radii_of_visible[0] > 0:
                 threshold = radii_of_visible[0].float()
            else:
                return torch.tensor(0.0, device=self._xyz.device)
        else:
            try:
                threshold = torch.quantile(radii_of_visible.float(), quantile_val)
            except RuntimeError as e:
                print(f"Warning: torch.quantile failed in adaptive_opacity: {e}. Using mean radii as fallback.")
                threshold = radii_of_visible.float().mean()
        
        smooth_gaussians_indices = visible_indices[radii_of_visible > threshold]
        
        if smooth_gaussians_indices.numel() == 0:
            return torch.tensor(0.0, device=self._xyz.device)
            
        smooth_opacities = current_opacities[smooth_gaussians_indices]
        
        if smooth_opacities.numel() > 0:
            loss = torch.abs(smooth_opacities[:, 0] - smooth_opacities[:, 1]).mean()
            return loss
        else:
            return torch.tensor(0.0, device=self._xyz.device)
    
    def set_adaptive_opacity_params(self, smooth_region_quantile=0.65):
        self._smooth_region_quantile = smooth_region_quantile


    def get_selective_normal_smoothness_loss(self, k, current_iteration, visibility_filter, radii):
        current_point_count = self._xyz.shape[0]

        if current_point_count < 500000: freq_divisor = 2
        elif current_point_count < 1500000: freq_divisor = 3
        else: freq_divisor = 4
        
        if (current_iteration % freq_divisor != 0 or 
            not KNN_CUDA_AVAILABLE or 
            current_point_count < k + 1):
            return torch.tensor(0.0, device=self._xyz.device)
        
        if current_point_count < 500000: knn_recompute_freq_multiplier = 1
        elif current_point_count < 1500000: knn_recompute_freq_multiplier = 2
        else: knn_recompute_freq_multiplier = 4
        
        effective_knn_recompute_interval = self.knn_recompute_interval_normal * knn_recompute_freq_multiplier

        if (self._knn_indices_cache_normal is None or
            current_iteration // effective_knn_recompute_interval != self._knn_cache_iter_normal // effective_knn_recompute_interval or
            self._knn_indices_cache_normal.shape[0] != current_point_count or
            self._knn_indices_cache_normal.shape[1] != k):
            
            print(f"[Iter {current_iteration}, Pts {current_point_count/1000:.0f}K] "
                  f"Recomputing GLOBAL KNN (k={k}) for selective normal smoothness. "
                  f"Effective interval: {effective_knn_recompute_interval}")
            points_xyz_for_knn = self._xyz.detach()
            try:
                neighbor_indices, _ = simple_knn_C.knn_indices_cuda(points_xyz_for_knn, k)
                self._knn_indices_cache_normal = neighbor_indices.long()
                self._knn_cache_iter_normal = current_iteration
            except Exception as e:
                print(f"ERROR during CUDA KNN for selective_normal_smoothness: {e}. Skipping loss.")
                self._knn_indices_cache_normal = None
                return torch.tensor(0.0, device=self._xyz.device)

        if self._knn_indices_cache_normal is None:
            return torch.tensor(0.0, device=self._xyz.device)
        
        visible_radii = radii[visibility_filter]
        if visible_radii.numel() < 2:
            return torch.tensor(0.0, device=self._xyz.device)
            
        interior_quantile = getattr(self, '_interior_region_quantile', 0.7)
        try:
            radii_threshold = torch.quantile(visible_radii.float(), interior_quantile)
        except RuntimeError:
            radii_threshold = visible_radii.float().mean()

        interior_smooth_mask_global = visibility_filter & (radii > radii_threshold)
        interior_smooth_indices_global = torch.where(interior_smooth_mask_global)[0]

        if interior_smooth_indices_global.numel() == 0:
            return torch.tensor(0.0, device=self._xyz.device)

        if current_point_count < 500000: max_points_for_loss = 800
        elif current_point_count < 1500000: max_points_for_loss = 500
        else: max_points_for_loss = 300
        num_to_sample = min(max_points_for_loss, interior_smooth_indices_global.numel())

        if num_to_sample <= k :
            return torch.tensor(0.0, device=self._xyz.device)

        perm = torch.randperm(interior_smooth_indices_global.numel(), device=self._xyz.device)
        selected_ref_indices_in_interior_smooth = perm[:num_to_sample]
        ref_gaussians_indices = interior_smooth_indices_global[selected_ref_indices_in_interior_smooth]

        if ref_gaussians_indices.numel() == 0:
            return torch.tensor(0.0, device=self._xyz.device)
            
        activated_normals = F.normalize(self._normal, p=2, dim=1)
        
        ref_normals = activated_normals[ref_gaussians_indices]

        knn_indices_for_refs = self._knn_indices_cache_normal[ref_gaussians_indices] 

        total_loss_sum = torch.tensor(0.0, device=self._xyz.device)
        total_valid_pairs = 0

        if current_point_count < 800000: num_neighbors_to_use = min(k, 3)
        else: num_neighbors_to_use = min(k, 2)

        for j_neighbor_idx in range(num_neighbors_to_use):
            current_neighbor_original_indices = knn_indices_for_refs[:, j_neighbor_idx]           
            valid_indices_mask = (current_neighbor_original_indices >= 0) & \
                                 (current_neighbor_original_indices < current_point_count)
            
            valid_and_interior_smooth_neighbor_mask = torch.zeros_like(current_neighbor_original_indices, dtype=torch.bool)
            if valid_indices_mask.sum() > 0:
                actual_valid_neighbor_indices = current_neighbor_original_indices[valid_indices_mask]
                are_these_neighbors_interior_smooth = interior_smooth_mask_global[actual_valid_neighbor_indices]
                valid_and_interior_smooth_neighbor_mask[valid_indices_mask] = are_these_neighbors_interior_smooth
            
            not_self_mask = (current_neighbor_original_indices != ref_gaussians_indices)
            final_neighbor_mask = valid_and_interior_smooth_neighbor_mask & not_self_mask

            if not torch.any(final_neighbor_mask):
                continue
            
            current_ref_normals_for_loss = ref_normals[final_neighbor_mask]
            final_neighbor_indices_for_loss = current_neighbor_original_indices[final_neighbor_mask]
            current_neighbor_normals_for_loss = activated_normals[final_neighbor_indices_for_loss]

            if current_ref_normals_for_loss.numel() > 0:
                dot_products = torch.sum(current_ref_normals_for_loss * current_neighbor_normals_for_loss, dim=1)
                loss_for_this_k_neighbor = (1.0 - dot_products).pow(2).sum()
                total_loss_sum += loss_for_this_k_neighbor
                total_valid_pairs += current_ref_normals_for_loss.shape[0]

        if total_valid_pairs > 0:
            return total_loss_sum / total_valid_pairs
        else:
            return torch.tensor(0.0, device=self._xyz.device)

    def set_selective_normal_params(self, interior_region_quantile=0.7):
        self._interior_region_quantile = interior_region_quantile