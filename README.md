<!-- Title -->
<div align="center">
<h1>Quantifying Shared Information Between Morphology and Spatial Transcriptomics via Cross-Modal Trajectory Correlation</h1>

<!-- Short description -->
Pre-print available on arXiv.

</div>

# How to install
Follow the steps below to create and activate the environment, and install the project dependencies:
```python
conda create --name pseudotime python=3.9
conda activate pseudotime
pip install -r requirements.txt
```

# Patch-based Feature Extraction
Please refer to preprocessing/data for the dataset format. Note that you need to download all required datasets and pretrained models by yourself using the links provided at the end, and update the corresponding paths before running the scripts.

You can use the following commands to extract image patches from the HEST dataset, either at the spot size or at a fixed resolution:
```python
python preprocessing/patch_extractor.py
```

For standard architectures such as ResNet18 and ViT-B/16, we leverage the [timm](https://huggingface.co/timm) library. For other specialized models, we use the [Trident](https://github.com/mahmoodlab/TRIDENT) framework. You can use the following commands to extract features:

```python
python preprocessing/feature_extractor.py
```

# HEST Data Download Link

| Dataset                  | Download Link |
|--------------------------|--------------|
| HEST-1k                     | [Link](https://github.com/mahmoodlab/HEST) |


# Pretrained Model Download Link
Note: Access to each model must be requested before downloading.
| Dataset                  | Download Link |
|--------------------------|--------------|
| Resnet18                 | [Link](https://huggingface.co/timm) |
| ViT-B/16                 | [Link](https://huggingface.co/timm) |
| CTransPath               | [Link](https://github.com/Xiyue-Wang/TransPath) |
| CONCH                    | [Link](https://huggingface.co/MahmoodLab/CONCH/tree/main) |
| CONCHv1.5                | [Link](https://huggingface.co/MahmoodLab/conchv1_5) |
| Prov-GiagPath            | [Link](https://huggingface.co/prov-gigapath/prov-gigapath) |
| UNI                      | [Link](https://huggingface.co/MahmoodLab/UNI) |
| UNI2-h                   | [Link](https://huggingface.co/MahmoodLab/UNI2-h) |
| Virchow                  | [Link](https://huggingface.co/paige-ai/Virchow) |
| Virchow2                 | [Link](https://huggingface.co/paige-ai/Virchow2) |



# Pathology to Spatial Transcriptomics
For the pathology-to-spatial transcriptomics prediction task, we adopt the method proposed in the [MISO](https://www.nature.com/articles/s41467-025-66691-y) paper.
The official implementation is available at [here](https://github.com/owkin/miso_code).

# How to cite
If you find our work useful in your research or if you use parts of this code, please consider citing our papers:
```python
@article{,
  title={},
  author={},
  journal={},
  year={2026}
}
```
