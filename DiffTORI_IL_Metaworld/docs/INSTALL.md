# Installing Conda Environment from Zero to Hero

First, git clone this repo and `cd` into it.
```
git clone https://github.com/wkwan7/DiffTORI.git
```

**Please strictly follow the guidance to avoid any potential errors. Especially, make sure Gym version is the same.**

**Don't worry about the gym version now. Just install my version in `third_party/gym-0.21.0` and you will be fine.**

# 0 create python/pytorch env
```
conda remove -n difftori --all
conda create -n difftori python=3.8
conda activate difftori
```

# 1 install some basic packages
```
pip3 install torch==2.0.1 torchvision torchaudio
# or for cuda 12.1
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install --no-cache-dir wandb ipdb gpustat visdom notebook mediapy torch_geometric natsort scikit-video easydict pandas moviepy imageio imageio-ffmpeg termcolor av open3d dm_control dill==0.3.5.1 hydra-core==1.2.0 einops==0.4.1 diffusers==0.11.1 zarr==2.12.0 numba==0.56.4 pygame==2.1.2 shapely==1.8.4 tensorboard==2.10.1 tensorboardx==2.5.1 absl-py==0.13.0 pyparsing==2.4.7 jupyterlab==3.0.14 scikit-image yapf==0.31.0 opencv-python==4.5.3.56 psutil av matplotlib setuptools==59.5.0

cd third_party
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ../..

```

# 2 install DiffTORI
```bash
cd third_party/robomimic-0.2.0
pip install -e .
cd ../..

cd DiffTORI
pip install -e .
cd ..

```

# 3 install environments
```
pip install --no-cache-dir patchelf==0.17.2.0
cd third_party


cd gym-0.21.0
pip install -e .
cd ..
```

install mujoco in `~/.mujoco`
```
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

tar -xvzf mujoco210.tar.gz
```
and put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl

```
and then install mujoco-py (in the folder of `third_party`):
```
cd YOUR_PATH_TO_THIRD_PARTY
cd mujoco-py-2.1.2.14
pip install -e .
cd ../..
```

Install MetaWorld environments:
```bash
cd third_party/Metaworld
pip install -e .
cd ../..
```

# 4 install Theseus
We are using [Theseus](https://github.com/facebookresearch/theseus) to perform differentiable trajectory optimization which enables computation of the gradient of the actions with respect to the model parameters $\frac{\partial a(\theta)}{\partial\theta}$.

```bash
pip install theseus-ai
```

# 5 error catching
You can refer to [ERROR_CATCH.md](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/ERROR_CATCH.md) in DP3 for some error catching.
