# Adaptive Control for 3D Gaussian Splatting: A Systematic Regularization Framework

This repository is the official implementation for our paper, **"Adaptive Control for 3D Gaussian Splatting: A Systematic Regularization Framework"**. Our work introduces a systematic, context-aware regularization framework for 3D Half-Gaussian Splatting (3D-HGS) that significantly improves rendering quality by resolving interdependent trade-offs between detail, smoothness, and stability.

---

## 1. Installation

Our project is built upon the official [3D-HGS](https://github.com/hli-plus/3D-HGS) repository.

**a. Clone the repository:**
```bash
git clone https://github.com/UserXiongWX/Adaptive-GS.git
```

**b. Create and activate a Conda environment:**
```bash
conda create -n adaptive-gs python=3.9
conda activate adaptive-gs
pip install -r requirements.txt
```

**c. Build custom CUDA extensions:**
```bash
This step requires a C++ compiler and a compatible CUDA toolkit (e.g., CUDA 11.8) to be installed.
pip install ./simple-knn

pip install ./submodules/diff-gaussian-rasterization```
*(Note: Please ensure the paths to `simple-knn` and `diff-gaussian-rasterization` match your project structure.)*
```
---

## 2. Training

**a. Download Datasets:**
Please download the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [Tanks and Temples](https://www.tanksandtemples.org/), and [Deep Blending](https://github.com/hedman/deep_blending_dataset) datasets and structure them as expected by the original 3D-GS framework.

**b. Run Training:**
Our regularization components are enabled by default. To train a model on a scene, simply run the `train.py` script:

```bash
python train.py -s /path/to/your/datasets -m ./output
```

## 3. Rendering and Evaluation
To render a video from the test camera path or evaluate the performance metrics (PSNR, SSIM, LPIPS) of a trained model, use the render.py script.
```bash
# Render a video of the test trajectory
python render.py -m ./output --skip_train

# Evaluate metrics on the test set
python render.py -m ./output --skip_train --eval
```