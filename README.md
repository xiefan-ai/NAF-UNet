# NAF-UNet
NAF-UNet: Neighbor-Aware Multi-Scale Fusion with Channel Attention for Medical Image Segmentation. A novel encoder-decoder architecture achieving SOTA on Synapse, ACDC, and ISIC benchmarks.

## Architecture

<p align="center">
<img src="figs/main.jpg" width=80% height=80% 
class="center">
</p>

## Quantitative Results

<p align="center">
<img src="figs/synapse.jpg" width=80% height=80%>
</p>

<p align="center">
<img src="figs/acdc.jpg" width=40% height=40% 
class="center">
</p>


## Qualitative Results

<p align="center">
<img src="figs/limits.jpg" width=80% height=80% 
class="center">
</p>

<p align="center">
<img src="figs/visulization_synapse.jpg" width=80% height=80% 
class="center">
</p>

<p align="center">
<img src="figs/visulization_isic18.jpg" width=80% height=80% 
class="center">
</p>


## Usage
### 1. Prepare data

Synapse (BTCV preprocessed data) and ACDC data are available at [TransUNet](https://github.com/Beckschen/TransUNet/tree/main)'s repo. 

The directory structure of the whole project is as follows:

```
.
├── NAF-UNet
│   ├──datasets
│   │       └── dataset_*.py
│   ├──train.py
│   ├──test.py
│   ├──...
│   └──data
│        └── Synapse
│        │        ├── test_vol_h5
│        │        │   ├── case0001.npy.h5
│        │        │   └── *.npy.h5
│        │        └── train_npz
│        │          ├── case0005_slice000.npz
│        │          └── *.npz
│        │
│        └──ACDC
│             ├── ACDC_training_volumes
│             │      ├── patient100_frame01.h5
│             │      └── *.h5
│             └── ACDC_training_slices
│                    ├── patient100_frame13_slice_0.h5
│                    └── *.h5   

```
### 2. Environment

Please prepare an environment with python=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Download weights

Pretrained weights can be downloaded at (https://pan.baidu.com/s/18QIV1EHL9TEyvLqLXL5BdA?pwd=hnsi).


### 4. Train/Test

- Run the training script on the Synapse dataset.

```bash
python train.py --dataset Synapse --max_epochs 300 --img_size 224 
```

- Run the test script on the Synapse dataset.

```bash
python test.py --dataset Synapse --max_epochs 300 --img_size 224
```


## Acknowledgements
This code base uses certain code blocks and helper functions from [TransUNet](https://github.com/Beckschen/TransUNet/tree/main).

## Citations


