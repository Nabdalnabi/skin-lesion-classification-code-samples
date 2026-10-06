"""
Multimodal skin lesion classification training pipeline.

This is an employer-facing, cleaned code sample inspired by a public
Skin-Lesion-Classification notebook. It removes hardcoded local paths and
expects the user to provide a local/public dataset directory at runtime.

Example:
    python multimodal_skin_lesion_training.py \
        --data-dir /path/to/HAM10000 \
        --metadata HAM10000_metadata.csv
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from split_utils import split_data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# -----------------------------
# Configuration
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal skin lesion classifier")
    parser.add_argument("--data-dir", required=True, help="Dataset root directory")
    parser.add_argument("--metadata", default="HAM10000_metadata.csv", help="Metadata CSV filename")
    parser.add_argument("--image-id-col", default="image_id", help="Image ID column")
    parser.add_argument("--label-col", default="dx", help="Class label column")
    parser.add_argument("--group-col", default="lesion_id", help="Use patient ID when available; otherwise lesion ID")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum text token length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Optional device override: cpu, cuda, or mps")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Data Preparation
# -----------------------------

def build_image_lookup(data_dir):
    data_dir = Path(data_dir)
    lookup = {}
    for path in data_dir.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.stem in lookup:
                raise ValueError("Duplicate image stems; resolve ambiguous image paths.")
            lookup[path.stem] = str(path)
    return lookup


def build_text_context(df):
    context_cols = [col for col in ["age", "sex", "localization"] if col in df.columns]
    if not context_cols:
        return pd.Series(["skin lesion image"] * len(df), index=df.index)

    text = df[context_cols].astype("string").fillna("unknown").agg(" ".join, axis=1)
    return text.replace("", "skin lesion image")


def load_metadata(data_dir, metadata_filename, image_id_col, label_col):
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_filename

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required = {image_id_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    image_lookup = build_image_lookup(data_dir)
    df["image_path"] = df[image_id_col].astype(str).map(image_lookup)
    if df[["image_path", label_col]].isna().any().any():
        raise ValueError("Missing images or labels; no records were silently removed.")
    if df[image_id_col].duplicated().any():
        raise ValueError("Duplicate image IDs in metadata.")
    df["text"] = build_text_context(df)

    label_names = sorted(df[label_col].astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    df["label_id"] = df[label_col].astype(str).map(label_to_id)

    return df, label_to_id, id_to_label


# -----------------------------
# Dataset
# -----------------------------

class SkinLesionDataset(Dataset):
    def __init__(self, df, tokenizer, transform, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        encoded_text = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label_id"], dtype=torch.long),
        }


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


# -----------------------------
# Model
# -----------------------------

class MultimodalSkinLesionClassifier(nn.Module):
    def __init__(self, num_classes, text_model_name="bert-base-uncased"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # Lightweight CNN image encoder keeps this code sample runnable without
        # external custom ALBEF source files.
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
        )

        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, 256)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_cls)

        fused = torch.cat([image_features, text_features], dim=1)
        return self.classifier(fused)


# -----------------------------
# Training And Evaluation
# -----------------------------

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(mode=train)
    losses = []
    labels = []
    predictions = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses.append(float(loss.item()) * len(y))
            predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

    return {
        "loss": float(sum(losses) / len(labels)),
        "accuracy": accuracy_score(labels, predictions),
    }


def evaluate(model, loader, id_to_label, device):
    model.eval()
    labels = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            logits = model(image, input_ids, attention_mask)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    target_names = [id_to_label[i] for i in sorted(id_to_label)]
    print("Test accuracy:", round(accuracy_score(labels, predictions), 4))
    print("\nConfusion matrix")
    print(confusion_matrix(labels, predictions, labels=sorted(id_to_label)))
    print("\nClassification report")
    print(classification_report(labels, predictions, labels=sorted(id_to_label), target_names=target_names, zero_division=0))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    df, label_to_id, id_to_label = load_metadata(
        args.data_dir,
        args.metadata,
        args.image_id_col,
        args.label_col,
    )
    train_df, valid_df, test_df = split_data(df, args.seed, args.group_col)

    print(f"Classes: {label_to_id}")
    print(f"Train/valid/test sizes: {len(train_df)}, {len(valid_df)}, {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_transform, eval_transform = get_transforms()

    train_loader = DataLoader(
        SkinLesionDataset(train_df, tokenizer, train_transform, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        SkinLesionDataset(valid_df, tokenizer, eval_transform, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SkinLesionDataset(test_df, tokenizer, eval_transform, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MultimodalSkinLesionClassifier(num_classes=len(label_to_id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_acc={valid_metrics['accuracy']:.4f}"
        )

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    evaluate(model, test_loader, id_to_label, device)


if __name__ == "__main__":
    main()
