<!-- Title -->
<div align="center">
<h1>Quantifying Shared Information Between Morphology and Spatial Transcriptomics via Cross-Modal Trajectory Correlation</h1>
</div>

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


# Patch-based Feature Extraction
For standard architectures such as ResNet18 and ViT-B/16, we leverage the [timm](https://huggingface.co/timm) library:
```python
import timm

model = timm.create_model('resnet18', pretrained=True)
```

For other specialized models, we use the [Trident](https://github.com/mahmoodlab/TRIDENT) framework:
```python
from trident.patch_encoder_models import encoder_factory

model = encoder_factory('conch_v15', weights_path=custom_ckpt_path)
```

# Pathology to Spatial Transcriptomics
For the pathology-to-spatial transcriptomics prediction task, we adopt the method proposed in the [MISO](https://www.nature.com/articles/s41467-025-66691-y) paper.
The official implementation is available at [here](https://github.com/owkin/miso_code).
