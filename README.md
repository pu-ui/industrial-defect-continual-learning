# Anonymous Core Implementation

This anonymous repository provides the core implementation for peer review. The complete training pipeline, detailed configurations, data splits, and evaluation scripts will be released after acceptance.

## Overview

This repository contains a compact implementation of a risk-aware, parameter-efficient continual learning framework for industrial defect recognition. The released code focuses on the method components needed to inspect the main technical design, not on reproducing all experiments.

## Included Modules

- `anonymous_idcl/risk.py`: risk-aware sample selection, high-risk thresholding, low-risk safe thresholding, and selection label utilities.
- `anonymous_idcl/model.py`: ResNet-50 classifier with Coordinate Attention modules and the PECL parameter-freezing policy.
- `anonymous_idcl/training.py`: class-balanced focal loss, knowledge distillation loss, batch-normalization handling, and a compact classifier training loop.
- `anonymous_idcl/calibration.py`: temperature scaling for confidence calibration.
- `anonymous_idcl/data.py`: minimal image dataset and manifest reader.
- `configs/example_config.yaml`: anonymized example configuration.
- `examples/minimal_usage.py`: a small script showing the basic method calls.

## Not Included

This review release intentionally omits the complete training pipeline, full experiment scripts, detailed data splits, production data, model weights, logs, generated figures, manuscript files, and reviewer-response materials.

## Basic Dependencies

The core modules require:

```text
Python 3.9+
PyTorch
torchvision
NumPy
Pillow
PyYAML
scikit-learn
```

The PatchCore-based risk scorer additionally requires the PatchCore backend used by the full project. The supervised model, PECL policy, knowledge distillation, and calibration modules can be imported without that backend.

## Example Usage

Install dependencies in your own environment, then run:

```bash
python examples/minimal_usage.py
```

The example constructs the Coordinate Attention classifier, applies the PECL trainable-parameter policy, computes risk thresholds from toy scores, selects samples by risk region, and runs a dummy forward pass.

## Directory Structure

```text
.
├── anonymous_idcl/
│   ├── calibration.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── risk.py
│   └── training.py
├── configs/
│   └── example_config.yaml
├── examples/
│   └── minimal_usage.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Review-Stage Scope

This repository is intended only to expose the core implementation for anonymous peer review. It should not be interpreted as a complete reproducibility package for all tables and figures in the manuscript.
