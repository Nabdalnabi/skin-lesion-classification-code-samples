# Skin Lesion Classification Code Samples

This private repository contains an employer-facing, cleaned version of a skin
lesion classification project based on multimodal learning.


## Research Publication

This code sample is associated with the research article:

**Accurate Skin Lesion Classification Using Multimodal Learning on the
HAM10000 and ISIC 2017 Datasets**

Abdulmateen Adebiyi, Nader Abdalnabi, Emily Hoffman Smith, Jesse Hirner,
Eduardo J. Simoes, Mirna Becevic, and Praveen Rao.

[Read the medRxiv preprint](https://doi.org/10.1101/2024.05.30.24308213)

The study reported accuracy/AUC-ROC of 0.9411/0.9426 on HAM10000 and
0.7971/0.8253 on ISIC 2017. This repository is a compact, cleaned code sample
for review and is not a clinical tool or an exact reproduction package.

## Project Summary

The project addresses multiclass skin lesion classification using image data and
optional patient/context metadata. The cleaned code sample uses an ALBEF-style
multimodal architecture that combines:

- image embeddings from a Vision Transformer
- text/context embeddings from a BERT encoder
- a fused representation for skin lesion class prediction

## Contents

- `multimodal_skin_lesion_training.py`: organized training and evaluation
  pipeline.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: safeguards against committing data, model weights, outputs, and
  local environment files.

## Data Privacy

No private data, image files, patient records, metadata CSV files, trained model
weights, or notebook outputs are included in this repository. The script expects
the user to provide a local/public dataset path at runtime.

## Expected Data Layout

```text
data/
  HAM10000_metadata.csv
  class_or_folder_name/
    image_1.jpg
    image_2.jpg
```

The metadata file should include at least:

- `image_id`
- `dx`

Optional context columns used for text features:

- `age`
- `sex`
- `localization`

## Example Usage

```bash
python multimodal_skin_lesion_training.py \
  --data-dir /path/to/ham10000 \
  --metadata HAM10000_metadata.csv \
  --epochs 20 \
  --batch-size 16
```

## Notes

This repository is intentionally compact and privacy-safe for employer review.
It demonstrates project structure, model design, preprocessing, training, and
evaluation while avoiding raw research data or private artifacts.
