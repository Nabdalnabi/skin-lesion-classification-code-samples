"""Keep all images from a lesion (or patient) within a single partition."""
from sklearn.model_selection import GroupShuffleSplit


def split_data(df, seed=42, group_col="lesion_id"):
    if group_col not in df or df[group_col].isna().any():
        raise ValueError("A complete grouping column is required to prevent image-level leakage.")
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, other_idx = next(first.split(df, groups=df[group_col]))
    train, other = df.iloc[train_idx], df.iloc[other_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(second.split(other, groups=other[group_col]))
    val, test = other.iloc[val_idx], other.iloc[test_idx]
    known = set(train.label_id)
    if len(known) < 2 or not set(df.label_id).issubset(known):
        raise ValueError("Training split lacks classes. Supply more groups or a prespecified split protocol.")
    return tuple(part.reset_index(drop=True) for part in (train, val, test))
