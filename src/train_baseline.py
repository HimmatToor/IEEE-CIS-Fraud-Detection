import argparse
from pathlib import Path

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from build_graph import load_raw_tables
from train import time_based_split

DROP_COLUMNS = ["TransactionID", "isFraud", "TransactionDT"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("ieee-fraud-detection-data"))
    args = parser.parse_args()

    df = load_raw_tables(args.data_dir, "train")
    train_idx, val_idx = time_based_split(df)

    X = df.drop(columns=DROP_COLUMNS)
    for column in X.select_dtypes(include="object").columns:
        X[column] = X[column].astype("category").cat.codes
    y = df["isFraud"]

    X_train, y_train = X.iloc[train_idx.numpy()], y.iloc[train_idx.numpy()]
    X_val, y_val = X.iloc[val_idx.numpy()], y.iloc[val_idx.numpy()]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
    )
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"train_auc {train_auc:.4f}")
    print(f"val_auc {val_auc:.4f}")


if __name__ == "__main__":
    main()
