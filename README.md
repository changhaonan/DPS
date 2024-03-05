# DPS: Diffusion Point Cloud Segmentation

## Install
```

```

## Dependency

We only tested on `CUDA11.8`.

1. Install other dependency
```
pip install -r requirements.txt
```

2. Install torch 2.1.0 first. This is to get adapt with torch3d.

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Install pytorch3d's dependency:
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

4. Install pytorch3d
```
conda install pytorch3d -c pytorch3d
```

5. Install torch_scatter.
```
pip install torch_geometric
pip install torch-scatter
pip install torch_cluster
```

6. Install pytorchlightning==2.2.1

7. Build code for superoint; Comment out everything before `echo "⭐ Installing FRNN"`

You need to replace the `line 19: extra_compile_args = {"cxx": ["-std=c++14"]}` to `extra_compile_args = {"cxx": ["-std=c++17"]}` in `external/superpoint_transformer/src/dependencies/FRNN/setup.py`.

```
cd external/superpoint_transformer
./install.sh
```

You need to install `pytorch3d`, `superpoint`.

- [Superpoint] (https://github.com/drprojects/superpoint_transformer)

8. Install detectron2.