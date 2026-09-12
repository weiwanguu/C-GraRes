import copy
import glob
import io
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


DATASET_FOLDER = "dataset/"
NUM_OBJECTS = 25
SAMPLES_PER_OBJECT = 30
NUM_CHANNELS = 36
TIME_INTERVAL = 0.03
MAX_TIME = 16
NUM_ROWS_SELECTED = int(MAX_TIME / TIME_INTERVAL)
WINDOW_SIZE = 110
SHIFT_STEP = 77
WINDOWS_PER_SAMPLE = 6
NOISE_COPIES = 5
NOISE_LEVEL = 0.01
RANDOM_STATE = 42
LOSS_CURVE_Y_LIMIT = (0.0, 2.5)
NPZ_ARRAY_KEY = "data"

CALCULATED_WINDOWS = int(np.floor((NUM_ROWS_SELECTED - WINDOW_SIZE) / SHIFT_STEP) + 1)
assert CALCULATED_WINDOWS == WINDOWS_PER_SAMPLE


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_sensor_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.endswith(","):
            line = line[:-1]
        fields = line.split(",")
        if len(fields) < NUM_CHANNELS:
            continue
        if len(fields) > NUM_CHANNELS:
            fields = fields[:NUM_CHANNELS]
        cleaned_lines.append(",".join(fields))

    cleaned_data = "\n".join(cleaned_lines)
    df = pd.read_csv(io.StringIO(cleaned_data), sep=",", header=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return pad_or_trim_rows(df, NUM_ROWS_SELECTED).values.astype(np.float32)


def read_sensor_npz(file_path):
    with np.load(file_path) as data:
        if NPZ_ARRAY_KEY not in data:
            raise KeyError(f"Missing '{NPZ_ARRAY_KEY}' in {file_path}")
        array = data[NPZ_ARRAY_KEY].astype(np.float32)

    expected_shape = (NUM_ROWS_SELECTED, NUM_CHANNELS)
    if array.shape != expected_shape:
        raise ValueError(f"{file_path} has shape {array.shape}, expected {expected_shape}.")
    return array


def pad_or_trim_rows(df, target_rows):
    if len(df) == 0:
        raise ValueError("Empty sensor dataframe after cleaning.")
    if len(df) >= target_rows:
        return df.iloc[:target_rows].reset_index(drop=True)

    pad_rows = target_rows - len(df)
    pad_values = np.repeat(df.iloc[[-1]].values, pad_rows, axis=0)
    pad_df = pd.DataFrame(pad_values, columns=df.columns)
    return pd.concat([df, pad_df], ignore_index=True)


def numeric_stem_sort_key(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    if stem.isdigit():
        return int(stem)
    return stem


def load_raw_sequences():
    if not os.path.exists(DATASET_FOLDER):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_FOLDER}")

    raw_data = []
    labels = []
    for object_id in range(1, NUM_OBJECTS + 1):
        object_folder = os.path.join(DATASET_FOLDER, f"object_{object_id}")
        file_paths = sorted(
            glob.glob(os.path.join(object_folder, "*.npz")),
            key=numeric_stem_sort_key,
        )
        if len(file_paths) < SAMPLES_PER_OBJECT:
            raise ValueError(
                f"{object_folder} has {len(file_paths)} npz files, "
                f"expected at least {SAMPLES_PER_OBJECT}."
            )
        if len(file_paths) > SAMPLES_PER_OBJECT:
            print(
                f"Warning: {object_folder} has {len(file_paths)} npz files; "
                f"using the first {SAMPLES_PER_OBJECT}."
            )
        file_paths = file_paths[:SAMPLES_PER_OBJECT]

        for file_path in file_paths:
            try:
                raw_data.append(read_sensor_npz(file_path))
                labels.append(object_id - 1)
            except Exception as exc:
                print(f"Failed to read {file_path}: {exc}")

    if not raw_data:
        raise ValueError("No raw data loaded.")
    return raw_data, labels


def apply_sliding_windows(data_list, label_list):
    windows = []
    labels = []
    for data, label in zip(data_list, label_list):
        for window_idx in range(WINDOWS_PER_SAMPLE):
            start_row = window_idx * SHIFT_STEP
            end_row = start_row + WINDOW_SIZE
            window = data[start_row:end_row, :].T  # [channels, time]
            windows.append(window)
            labels.append(label)
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def standardize_raw_windows(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def add_noise_augmentation(features, labels, noise_level=NOISE_LEVEL, num_copies=NOISE_COPIES):
    augmented_features = [features]
    augmented_labels = [labels]
    for _ in range(num_copies - 1):
        noise = np.random.normal(0, noise_level, features.shape).astype(np.float32)
        augmented_features.append(features + noise)
        augmented_labels.append(labels)
    return np.vstack(augmented_features), np.hstack(augmented_labels)


def prepare_raw_datasets():
    raw_data, raw_labels = load_raw_sequences()
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        raw_data,
        raw_labels,
        test_size=0.4,
        stratify=raw_labels,
        random_state=RANDOM_STATE,
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=RANDOM_STATE,
    )

    X_train, y_train = apply_sliding_windows(train_data, train_labels)
    X_val, y_val = apply_sliding_windows(val_data, val_labels)
    X_test, y_test = apply_sliding_windows(test_data, test_labels)

    X_train, X_val, X_test = standardize_raw_windows(X_train, X_val, X_test)
    X_train, y_train = add_noise_augmentation(X_train, y_train)

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


class RawWindowDataset(Dataset):
    def __init__(self, features, labels, layout="conv"):
        self.features = features
        self.labels = labels
        self.layout = layout

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.layout == "lstm":
            feature = feature.T  # [time, channels]
        elif self.layout == "flat":
            feature = feature.reshape(-1)
        feature_tensor = torch.FloatTensor(feature)
        label_tensor = torch.LongTensor([self.labels[idx]])
        return feature_tensor, label_tensor


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, model_name="Model"):
    train_losses = []
    val_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.squeeze().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"{model_name} best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return train_losses, val_losses


def evaluate_model(model, test_loader, device, model_name, train_losses, val_losses, num_epochs):
    model.eval()
    y_pred = []
    y_true = []
    y_proba = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    acc_test = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Test Accuracy: {acc_test:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Cohen's Kappa:  {kappa:.4f}")
    print(f"Matthews Coef:  {mcc:.4f}")

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    old_mask = y_true_np < 15
    new_mask = y_true_np >= 15
    old_acc = accuracy_score(y_true_np[old_mask], y_pred_np[old_mask])
    new_acc = accuracy_score(y_true_np[new_mask], y_pred_np[new_mask])
    print(f"Obj1-Obj15 Accuracy:  {old_acc:.4f}")
    print(f"Obj16-Obj25 Accuracy: {new_acc:.4f}")

    print("\nPer-class Accuracy:")
    for class_idx in range(NUM_OBJECTS):
        class_mask = y_true_np == class_idx
        class_acc = accuracy_score(y_true_np[class_mask], y_pred_np[class_mask])
        correct = (y_true_np[class_mask] == y_pred_np[class_mask]).sum()
        total = class_mask.sum()
        print(f"Obj{class_idx + 1:02d}: {class_acc:.4f} ({correct}/{total})")

    target_class = 19
    target_mask = y_true_np == target_class
    wrong_preds = y_pred_np[target_mask][y_pred_np[target_mask] != target_class]
    wrong_counter = Counter(int(pred + 1) for pred in wrong_preds)
    print(f"\nObj20 misclassified as: {wrong_counter}")

    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_loss_curve(train_losses, val_losses, num_epochs, model_name)


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    numeric_labels = [str(i + 1) for i in range(NUM_OBJECTS)]
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(12, 8))
    hm = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=numeric_labels,
        yticklabels=numeric_labels,
        annot_kws={"size": 12},
        cbar_kws={"shrink": 1},
        linewidths=0,
        ax=ax,
    )
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.tight_layout(pad=1.5)
    plt.savefig(f"Confusion_Matrix_{model_name}_raw.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_loss_curve(train_losses, val_losses, num_epochs, model_name):
    plt.figure(figsize=(2.362, 1.574))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o", markersize=1.0, linewidth=0.5)
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="s", markersize=1.0, linewidth=0.5)
    plt.xlabel("Epoch", fontsize=8)
    plt.ylabel("Loss", fontsize=8)
    plt.xlim(0, num_epochs)
    plt.xticks(range(0, num_epochs + 1, 10), fontsize=7)
    plt.ylim(*LOSS_CURVE_Y_LIMIT)
    plt.yticks(fontsize=7)
    plt.legend(fontsize=6)
    plt.tight_layout(pad=0.2)
    plt.savefig(f"{model_name}_raw_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{model_name}_raw_loss_curve.eps", format="eps", bbox_inches="tight")
    plt.show()
