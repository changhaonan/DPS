# DPS: Diffusion Point Cloud Segmentation

The goal of this project is to use diffusion model to perform affordance segmentation.

## Generate data

1. Run generate superpoint

```
python dps/scripts/gen_superpoint.py
```

2. Run preprocess

```
python preprocess_data.py
```

## Evaluation on RPdiff dataset

1. Set environment variable `RPDIFF_SOURCE_DIR`:
```
export RPDIFF_SOURCE_DIR=$PWD/dps/external/rpdiff
```

2. Run meshcat viewer:
```
meshcat-server
```

## Install

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

4. Install pytorch3d from source
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

5. Install torch_scatter.
```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

7. Install detectron2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

7. Build code for superoint; Comment out everything before `echo "⭐ Installing FRNN"`

You need to replace the `line 19: extra_compile_args = {"cxx": ["-std=c++14"]}` to `extra_compile_args = {"cxx": ["-std=c++17"]}` in `external/superpoint_transformer/src/dependencies/FRNN/setup.py`.

```
cd external/superpoint_transformer
./install.sh
```

You need to install `pytorch3d`, `superpoint`.

- [Superpoint] (https://github.com/drprojects/superpoint_transformer)

8. Install detectron2.

## TODO

- [x] Fix the normal for shape-completion.
- [x] Link the system with rpdiff pipeline.
- [ ] Add training for real object.
- [x] Improving the bbox points.
- [x] We can augment the pose registration with superpoint-augmentation; randomly dropping or superpoint.
- [ ] We can also apply a range filter.