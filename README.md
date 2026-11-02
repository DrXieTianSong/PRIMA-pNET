# PRIMA

PRIMA (Prior-knowledge and Report-text Integrated Multi-task Radiomics model) is an
interpretable multi-task model for the preoperative prediction of pathologic grade and
lymph node metastasis in pancreatic neuroendocrine tumors.

The model fuses three modalities through cross-attention with dual-task heads:

- quantitative radiomics features
- prior-knowledge-driven (PKD) features encoding tumor aggressiveness
- structured radiology report text encoded by RadBERT

## Repository structure

```
prima/
├── preprocessing/
│   └── image.py       # CT/mask resampling to isotropic spacing
├── features/
│   ├── radiomics.py   # PyRadiomics feature extractor
│   └── pkd.py         # prior-knowledge-driven feature extraction
├── text/
│   └── radbert.py     # RadBERT [CLS] text encoder
└── model/
    └── prima.py       # dual-residual cross-attention multi-task model
```

## Model

`DualResidualMTMLP` takes a table feature vector and a 768-dimensional text embedding,
projects them into a shared space, applies multi-head cross-attention, and predicts
two task heads (grade and lymph node metastasis).

## Dependencies

Install with:

```
pip install -r requirements.txt
```

## Usage

```python
from prima.model.prima import DualResidualMTMLP

model = DualResidualMTMLP(table_dim=32)
```

Each module accepts imaging or text inputs as numpy arrays or SimpleITK images and
returns the corresponding features or model outputs. Data loading, training, and
evaluation routines are not included in this repository.
