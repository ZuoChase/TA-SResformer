# TA-SResformer

Core model implementation of TA-SResformer for event-based rotating-machinery
fault diagnosis.

## Release scope

This repository is a model-core release. It contains:

- the complete TA-SResformer architecture used in the paper;
- a checkpoint-schema regression test.

It intentionally excludes datasets, data loaders, training and evaluation
entry points, pretrained checkpoints, ablation variants, comparison models,
experiment logs, paper drafts, and generated figures. Accordingly, this
repository does not reproduce the paper's experiments by itself.

Only `ta_sresformer` is registered in the public model module. Public
architecture classes use the paper-facing names `TASResformer`, `FRNSTSA`, and
`SGCFFN`. The naming refactor does not change historical checkpoint parameter
keys.

## Repository layout

```text
.
|-- models/
|   |-- ta_sresformer.py
|   `-- submodules/layers.py
|-- tests/test_behavior.py
|-- requirements.txt
`-- LICENSE
```

## Data availability

No dataset is included in this release. The self-built event-camera bearing
fault dataset is planned for a future public release together with its data
preparation description. XJTU-DV-Rotor is a third-party dataset and is not
redistributed here; obtain it from its official provider and comply with its
license and terms of use.

## Environment

The reported environment used Windows 11, Python 3.10.19, PyTorch
2.10.0+cu128, and an NVIDIA RTX 5080. A CUDA-capable GPU is required for the
forward pass because the released spiking-neuron implementation uses the CuPy
backend.

```bash
conda create -n ta-sresformer python=3.10 -y
conda activate ta-sresformer

# Install the PyTorch build matching your CUDA version first.
pip install torch
pip install -r requirements.txt
```

`requirements.txt` selects `cupy-cuda12x`. Replace it with `cupy-cuda11x` when
using CUDA 11.

## Verification

Run the CPU-compatible model API and checkpoint-schema tests:

```bash
python -m unittest discover -s tests -v
```

## Model use

```python
from timm.models import create_model
import models  # registers TA-SResformer

model = create_model(
    "ta_sresformer",
    T=4,
    in_channels=2,
    num_classes=4,
    img_size=64,
)
```

The model accepts event tensors shaped `[B, T, 2, H, W]` and returns logits
shaped `[T, B, num_classes]`. Average the logits over the temporal dimension
for final classification.

## Checkpoint compatibility

Load the model state from a compatible checkpoint as follows:

```python
import torch

checkpoint = torch.load("checkpoint_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
```

The checkpoint must use the released architecture configuration and the same
number of output classes as the instantiated model. No pretrained checkpoint
is included in this repository.

## Reproducibility boundary

Dataset preparation, event voxelization, data partitioning, optimization, and
evaluation protocols are outside this model-core release. Any downstream
experiment must document those choices independently. Do not claim that this
repository alone reproduces the paper's accuracy, robustness, complexity, or
energy results.

## Citation

The TA-SResformer citation will be added after the final bibliographic record
is available.

## License

Released under the MIT License. See `LICENSE`.
