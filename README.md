# Multimodal Skin Lesion Classification

Research code sample from collaborative work on skin-lesion classification,
organized into configuration, data preparation, dataset, model, and training blocks.

## Research

**Accurate Skin Lesion Classification Using Multimodal Learning on the HAM10000
and ISIC 2017 Datasets**

Abdulmateen Adebiyi, Nader Abdalnabi, Emily Hoffman Smith, Jesse Hirner,
Eduardo J. Simoes, Mirna Becevic, and Praveen Rao.
[Read the preprint](https://doi.org/10.1101/2024.05.30.24308213).

The study reported accuracy/ROC-AUC of 0.9411/0.9426 on HAM10000 and
0.7971/0.8253 on ISIC 2017. These are paper results, not measurements from
this simplified implementation.

## Implementation

The sample combines a **small CNN image encoder** with a pretrained **BERT text
encoder**, concatenates their projected representations, and trains a multiclass
classification head. It is **not a Vision Transformer or an ALBEF reproduction**.
Optional age, sex, and anatomical-location metadata form the text context.

- Lesion-disjoint train/validation/test partitions by default; use patient IDs
  instead when available with `--group-col patient_id`.
- Approximate 70/15/15 split by group count, not guaranteed class stratification.
- Training-only image augmentation, attention masks, AdamW, gradient clipping,
  and restoration of the lowest-validation-loss checkpoint in memory.
- Held-out accuracy, confusion matrix, and per-class precision/recall/F1.
- Explicit checks for missing images, duplicate image IDs, and ambiguous paths.

## Run

Python 3.10+ in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python multimodal_skin_lesion_training.py \
  --data-dir /path/to/authorized/HAM10000 \
  --metadata HAM10000_metadata.csv --group-col lesion_id \
  --epochs 20 --batch-size 16
```

The directory must contain the metadata CSV and image files in any subfolders.
Required columns: `image_id`, `dx`, and the selected grouping column. Optional:
`age`, `sex`, `localization`. Image stems must match `image_id` and be unique.
Initial BERT downloads contact Hugging Face. Datasets are obtained separately
under their own access and licensing conditions; no images or metadata are bundled.

## Validation and limitations

A synthetic grouping test checks partition separation. Full image-model training
and paper reproduction have not been validated in this preparation environment.
The script reports results to the console; it does not persist trained weights.
Lesion separation is not patient separation when several lesions belong to the
same person. Class imbalance, demographic bias, external validation, calibration,
and confidence intervals require further study. Not a clinical diagnostic tool.

## Privacy

Only source code and documentation belong in this repository. Do not commit
patient records, image datasets, metadata exports, notebooks with outputs,
credentials, or trained weights. `.gitignore` is a convenience, not an access
control or a guarantee that a file is safe to publish.
