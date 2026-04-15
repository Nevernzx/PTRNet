# PTRNet
## A Novel Multimodal Learning Method for Predicting Treatment Resistance in MPO-AAV with Lung Involvement

[[`Model`](#wait)] [[`Paper`](#wait)] [[`BibTeX`](#wait)]

## Method Overview

<p align="center">
    <img src="assets/overview.png" width="100%"> <br>
</p>

## Install

1. Download the repository and open the PTRNet
```
git clone https://github.com/yinangit/PTRNet.git
cd PTRNet
```

2. Required requirements

```bash
cd env
conda env create -f environment.yml
conda activate PTRNet
cd tab-transformer-pytorch
pip install -e .
```

3. Optional requirements (for feature extraction)

```bash
conda activate PTRNet
cd env/CLIP
pip install -e .
cd ../lungmask
pip install -e .
cd ../timm-0.9.12
pip install -e .
```

## Run

- **Train**
```bash
conda activate PTRNet
python scripts/train_mri.py --saveName mri_biomedclip --model_mode image --d_model 256
```

- **Test**
```bash
conda activate PTRNet
python scripts/test.py --saveName mri_biomedclip --model_mode image --d_model 256 --weight_path log/mri_biomedclip/model_final.pt
```

- **Ablation**

Taking `ablation of hyperparameter γ` as an example:
```bash
conda activate PTRNet
cd launch
bash ablation_gamma.sh
```

## Acknowledgements


## Citation
