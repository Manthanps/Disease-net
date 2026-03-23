# -*- coding: utf-8 -*-
"""mini project (local)

Converted from Colab to run locally.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import json
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
BLOOD_DIR = BASE_DIR / "blood cancer"
LUNG_DIR = BASE_DIR / "lung cancer"
HEART_DIR = BASE_DIR / "heart attack"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CONFIG_PATH = BASE_DIR / "datasets.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    DATASETS_CONFIG = json.load(f)


def get_tabular_cfg(name: str) -> Dict[str, Any]:
    cfg = DATASETS_CONFIG.get("tabular", {})
    if name not in cfg:
        raise KeyError(f"Tabular config not found for: {name}")
    return cfg[name]


def get_image_cfg(name: str) -> Dict[str, Any]:
    cfg = DATASETS_CONFIG.get("images", {})
    if name not in cfg:
        raise KeyError(f"Image config not found for: {name}")
    return cfg[name]


# -----------------------------
# CLI input/output
# -----------------------------

AVAILABLE_SECTIONS = [
    "blood_tabular",
    "blood_images",
    "blood_technical",
    "lung_images",
    "lung_gradcam",
    "lung_tabular",
    "heart_tabular",
    "heart_mimic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini project runner")
    parser.add_argument(
        "--sections",
        default="all",
        help="Comma-separated list of sections to run. Use 'all' to run everything.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help="Directory to write output metrics (JSONL).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for interactive text/image input after training.",
    )
    parser.add_argument(
        "--image-path",
        default="",
        help="Path to a single image for image prediction (optional).",
    )
    return parser.parse_args()


ARGS = parse_args()
ARTIFACTS_DIR = Path(ARGS.artifacts_dir)
ARTIFACTS_DIR.mkdir(exist_ok=True)

selected_sections = {s.strip() for s in ARGS.sections.split(",") if s.strip()}
run_all = "all" in selected_sections or not selected_sections
RUN_SECTIONS = {name: (run_all or name in selected_sections) for name in AVAILABLE_SECTIONS}

print("Running sections:", [k for k, v in RUN_SECTIONS.items() if v])


def record_metrics(label: str, metrics: dict) -> None:
    out_path = ARTIFACTS_DIR / "metrics.jsonl"
    payload = {"label": label, **metrics}
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\\n")


def prompt_for_value(col: str, default: Any, is_numeric: bool) -> Any:
    default_str = "" if default is None else str(default)
    while True:
        raw = input(f"Enter value for '{col}' [default: {default_str}]: ").strip()
        if raw == "":
            return default
        if is_numeric:
            try:
                return float(raw)
            except ValueError:
                print("Please enter a numeric value.")
        else:
            return raw


def prompt_tabular_input(df_raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    features = df_raw.drop(columns=[target_col])
    defaults_num = features.median(numeric_only=True).to_dict()
    defaults_cat = (
        features.select_dtypes(exclude=np.number).mode().iloc[0].to_dict()
        if not features.select_dtypes(exclude=np.number).empty
        else {}
    )
    row = {}
    for col in features.columns:
        is_numeric = pd.api.types.is_numeric_dtype(features[col])
        default = defaults_num.get(col) if is_numeric else defaults_cat.get(col)
        row[col] = prompt_for_value(col, default, is_numeric)
    return pd.DataFrame([row])


def align_one_hot(df_input: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df_input, drop_first=True)
    return df_encoded.reindex(columns=train_columns, fill_value=0)


def prompt_image_path(label: str) -> Path:
    if ARGS.image_path:
        path = Path(ARGS.image_path).expanduser()
    else:
        raw = input(f"Enter image path for {label}: ").strip()
        path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


# -----------------------------
# Explainability helpers
# -----------------------------

def explain_tabular_model(model, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, label: str) -> None:
    """Generate SHAP summary plot for a tabular model."""
    try:
        sample = X_test_df.sample(min(500, len(X_test_df)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        print(f"SHAP summary for {label}")
        shap.summary_plot(shap_values, sample, show=True)
    except Exception as e:
        print(f"SHAP explanation skipped for {label}: {e}")


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str, pred_index: Optional[int] = None) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()


# -----------------------------
# BLOOD CANCER (tabular)
if RUN_SECTIONS.get('blood_tabular', False):
    blood_cfg = get_tabular_cfg("blood")
    file_path = BASE_DIR / blood_cfg["csv_path"]
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    
    df = pd.read_csv(file_path)
    print("Dataset Shape:", df.shape)

    if blood_cfg.get("strip_columns"):
        df.columns = [c.strip() for c in df.columns]
    drop_cols = blood_cfg.get("drop_columns") or []
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
    df_raw = df.copy()
    
    # Fill numeric NaNs
    # Note: non-numeric columns are left as-is and handled by get_dummies
    numeric_medians = df.median(numeric_only=True)
    df = df.fillna(numeric_medians)
    
    target_col = blood_cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {file_path}")

    label_map = blood_cfg.get("label_map")
    le = None
    if label_map:
        df[target_col] = df[target_col].map(label_map)
    elif not pd.api.types.is_numeric_dtype(df[target_col]):
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    df.dropna(subset=[target_col], inplace=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if le is not None:
        print("Classes:", le.classes_)
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    X_df = X.copy()
    
    # Split once, then apply SMOTE only to training data
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if blood_cfg.get("use_smote"):
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train_df, y_train)
    else:
        X_train = X_train_df
    
    scaler = None
    if blood_cfg.get("use_scaler"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test_df)
    else:
        X_test = X_test_df
    
    model_cfg = blood_cfg.get("model", {})
    model_params = model_cfg.get("params", {})
    model = RandomForestClassifier(**model_params)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    record_metrics(
        "blood_tabular",
        {"accuracy": float(accuracy), "f1_weighted": float(f1)}
    )
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Explainable AI (SHAP) for heart attack model
    explain_tabular_model(rf, X_train, X_test, "Heart Attack (tabular)")
    
    explain_tabular_model(model, X_train_df, X_test_df, "Blood (tabular)")

    if ARGS.interactive:
        print("\nInteractive input for Blood (tabular)")
        input_df_raw = prompt_tabular_input(df_raw, target_col)
        input_encoded = align_one_hot(input_df_raw, X_df.columns)
        if scaler is not None:
            input_encoded = scaler.transform(input_encoded)
        pred = model.predict(input_encoded)[0]
        if label_map:
            inv = {v: k for k, v in label_map.items()}
            pred_label = inv.get(int(pred), pred)
        elif le is not None:
            pred_label = le.inverse_transform([pred])[0]
        else:
            pred_label = pred
        print("Predicted Class:", pred_label)
        record_metrics("blood_tabular_input", {"predicted_class": str(pred_label)})
    
    
    # -----------------------------
# Image dataset helpers
# -----------------------------

def has_images(root: Path) -> bool:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        if list(root.rglob(ext)):
            return True
    return False


def find_first_image_root(root: Path) -> Optional[Path]:
    if root.exists() and root.is_dir() and has_images(root):
        return root
    if root.exists() and root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and has_images(child):
                return child
    return None


def extract_zip_if_needed(zip_path: Path, extract_path: Path) -> Path:
    if extract_path.exists() and any(extract_path.iterdir()):
        return extract_path
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction completed to {extract_path}")
    return extract_path


def pick_sample_image(root: Path) -> Optional[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files = list(root.rglob(ext))
        if files:
            return files[0]
    return None


# -----------------------------
# BLOOD CANCER (images)
if RUN_SECTIONS.get('blood_images', False):
    blood_img_cfg = get_image_cfg("blood")
    blood_image_roots: List[Path] = []
    
    # 1) Folder dataset: Blood cell Cancer [ALL]
    blood_folder = BASE_DIR / blood_img_cfg.get("dataset_path", "blood cancer/Blood cell Cancer [ALL]")
    if blood_folder.exists() and has_images(blood_folder):
        blood_image_roots.append(blood_folder)
    
    # 2) Zip dataset: C-NMC_Leukemia.zip
    zip_path = BLOOD_DIR / "C-NMC_Leukemia.zip"
    extract_path = BLOOD_DIR / "extracted_dataset"
    if zip_path.exists():
        extracted_root = extract_zip_if_needed(zip_path, extract_path)
        blood_zip_root = find_first_image_root(extracted_root)
        if blood_zip_root:
            blood_image_roots.append(blood_zip_root)
    
    if not blood_image_roots:
        raise FileNotFoundError(
            "No blood cancer image dataset found. Expected a folder with images "
            "or a valid C-NMC_Leukemia.zip."
        )
    
    # Use the first available dataset for training; others remain available.
    dataset_path = blood_image_roots[0]
    print(f"Blood image datasets found: {[str(p) for p in blood_image_roots]}")
    print(f"Blood image dataset path set to: {dataset_path}")
    
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 40
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    cnn_model = Sequential()
    
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(2, 2))
    
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(2, 2))
    
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(2, 2))
    
    cnn_model.add(Flatten())
    
    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dropout(0.5))
    
    cnn_model.add(Dense(train_generator.num_classes, activation='softmax'))
    
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_path = BASE_DIR / blood_img_cfg.get("model_path", "artifacts/blood_cnn_best_model.h5")
    checkpoint = ModelCheckpoint(
        str(model_path),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    history = cnn_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint]
    )
    if "val_accuracy" in history.history:
        best_val_acc = max(history.history["val_accuracy"])
        record_metrics("blood_images", {"best_val_accuracy": float(best_val_acc)})

    if ARGS.interactive:
        try:
            print("\nInteractive input for Blood (image)")
            img_path = prompt_image_path("blood")
            img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            preds = cnn_model.predict(img_array)
            idx = int(np.argmax(preds[0]))
            class_map = {v: k for k, v in train_generator.class_indices.items()}
            pred_label = class_map.get(idx, str(idx))
            print("Predicted Class:", pred_label)
            record_metrics("blood_image_input", {"predicted_class": str(pred_label)})
        except Exception as e:
            print(f"Blood image prediction skipped: {e}")
    
    # Grad-CAM for blood CNN (explainable AI)
    try:
        blood_sample_img = pick_sample_image(dataset_path)
        if blood_sample_img is not None:
            img = tf.keras.utils.load_img(blood_sample_img, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
    
            last_conv = get_last_conv_layer_name(cnn_model)
            heatmap = make_gradcam_heatmap(img_array, cnn_model, last_conv)
    
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.imshow(heatmap, cmap="jet", alpha=0.4)
            plt.title("Blood CNN Grad-CAM")
            plt.axis("off")
            plt.show()
        else:
            print("No sample image found for Blood CNN Grad-CAM.")
    except Exception as e:
        print(f"Blood CNN Grad-CAM skipped: {e}")
    
    
    # -----------------------------
# LUNG CANCER (images)
if RUN_SECTIONS.get('lung_images', False):
    
    def ensure_final_dataset(base_path: Path) -> Path:
        lung_path = base_path / "lung_image_sets"
        colon_path = base_path / "colon_image_sets"
        final_path = base_path / "final_dataset"
    
        if final_path.exists() and any(final_path.iterdir()):
            return final_path
    
        if not lung_path.exists() or not colon_path.exists():
            raise FileNotFoundError(
                "Expected lung_image_sets and colon_image_sets under "
                f"{base_path}"
            )
    
        final_path.mkdir(parents=True, exist_ok=True)
    
        for src_parent in [lung_path, colon_path]:
            for cls_dir in src_parent.iterdir():
                if not cls_dir.is_dir():
                    continue
                dest = final_path / cls_dir.name
                if dest.exists():
                    continue
                try:
                    os.symlink(cls_dir, dest, target_is_directory=True)
                except OSError:
                    shutil.copytree(cls_dir, dest)
    
        print("All classes prepared in final_dataset folder")
        return final_path
    
    
    lung_img_cfg = get_image_cfg("lung")
    dataset_cfg_path = BASE_DIR / lung_img_cfg.get("dataset_path", "lung cancer/lung_colon_image_set/final_dataset")
    base_path = dataset_cfg_path.parent if dataset_cfg_path.name == "final_dataset" else dataset_cfg_path
    final_dataset_path = ensure_final_dataset(base_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=str(final_dataset_path), transform=transform)
    
    print("Classes detected:", dataset.classes)
    print("Total images:", len(dataset))
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # DENSE NET 121
    model = torchvision.models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        train_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}  Loss: {running_loss / len(train_loader):.4f}  Train Acc: {train_acc:.2f}%")
    
    # Save model for Grad-CAM usage
    lung_model_path = BASE_DIR / lung_img_cfg.get("model_path", "artifacts/densenet_lung.pth")
    torch.save(model.state_dict(), lung_model_path)
    
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
    
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
    
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * correct / total
    print("\n Test Accuracy:", test_acc)
    record_metrics("lung_images", {"test_accuracy": float(test_acc)})
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=dataset.classes,
        yticklabels=dataset.classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    if ARGS.interactive:
        try:
            print("\nInteractive input for Lung (image)")
            img_path = prompt_image_path("lung")
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                pred_idx = int(outputs.argmax(dim=1).item())
            pred_label = dataset.classes[pred_idx] if pred_idx < len(dataset.classes) else str(pred_idx)
            print("Predicted Class:", pred_label)
            record_metrics("lung_image_input", {"predicted_class": str(pred_label)})
        except Exception as e:
            print(f"Lung image prediction skipped: {e}")
    
    
    # -----------------------------
# GRAD-CAM (lung model)
if RUN_SECTIONS.get('lung_gradcam', False):
    if not RUN_SECTIONS.get("lung_images", False):
        print("Skipping lung Grad-CAM because lung_images section did not run.")
    else:
        sample_img_path = pick_sample_image(final_dataset_path)
        if sample_img_path is None:
            raise FileNotFoundError(f"No images found under {final_dataset_path}")
        
        class_names = dataset.classes
        
        model_cam = torchvision.models.densenet121(pretrained=False)
        num_features = model_cam.classifier.in_features
        model_cam.classifier = torch.nn.Linear(num_features, 5)
        
        model_cam.load_state_dict(torch.load(lung_model_path, map_location=device))
        model_cam = model_cam.to(device)
        model_cam.eval()
        
        target_layers = [model_cam.features[-1]]
        
        image = Image.open(sample_img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        output = model_cam(input_tensor)
        pred_class = output.argmax(dim=1).item()
        
        print("Predicted Class:", class_names[pred_class])
        
        cam = GradCAM(model=model_cam, target_layers=target_layers)
        
        targets = [ClassifierOutputTarget(pred_class)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(visualization)
        plt.title(f"Grad-CAM: {class_names[pred_class]}")
        plt.axis('off')
        plt.show()
    
    
    # -----------------------------
# LUNG CANCER (tabular: survey lung cancer.csv)
if RUN_SECTIONS.get('lung_tabular', False):
    lung_cfg = get_tabular_cfg("lung")
    lung_survey_csv = BASE_DIR / lung_cfg["csv_path"]
    if not lung_survey_csv.exists():
        raise FileNotFoundError(f"Missing file: {lung_survey_csv}")
    
    lung_df = pd.read_csv(lung_survey_csv)
    if lung_cfg.get("strip_columns"):
        lung_df.columns = [c.strip() for c in lung_df.columns]
    lung_df_raw = lung_df.copy()
    
    target_col = lung_cfg["target_col"]
    if target_col not in lung_df.columns:
        raise ValueError(f"Expected '{target_col}' column in {lung_survey_csv}")
    
    label_map = lung_cfg.get("label_map")
    if label_map:
        lung_df[target_col] = lung_df[target_col].map(label_map)
    lung_df.dropna(subset=[target_col], inplace=True)
    
    X_lung = lung_df.drop(target_col, axis=1)
    y_lung = lung_df[target_col].astype(int)
    
    X_lung = pd.get_dummies(X_lung, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_lung, y_lung, test_size=0.2, random_state=42, stratify=y_lung
    )
    
    model_cfg = lung_cfg.get("model", {})
    model_params = model_cfg.get("params", {})
    lung_rf = RandomForestClassifier(**model_params)
    
    lung_rf.fit(X_train, y_train)
    y_pred = lung_rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Lung Survey Accuracy:", accuracy)
    print("\nLung Survey Classification Report:\n")
    print(classification_report(y_test, y_pred))
    record_metrics("lung_tabular", {"accuracy": float(accuracy)})
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Lung Survey Confusion Matrix")
    plt.show()
    
    # Explainable AI (SHAP) for lung survey model
    explain_tabular_model(lung_rf, X_train, X_test, "Lung Survey (tabular)")

    if ARGS.interactive:
        print("\nInteractive input for Lung Survey (tabular)")
        input_df_raw = prompt_tabular_input(lung_df_raw, target_col)
        input_encoded = align_one_hot(input_df_raw, X_lung.columns)
        pred = lung_rf.predict(input_encoded)[0]
        if label_map:
            inv = {v: k for k, v in label_map.items()}
            pred_label = inv.get(int(pred), pred)
        else:
            pred_label = pred
        print("Predicted Class:", pred_label)
        record_metrics("lung_tabular_input", {"predicted_class": str(pred_label)})
    
    
    # -----------------------------
# HEART ATTACK (tabular)
if RUN_SECTIONS.get('heart_tabular', False):
    heart_cfg = get_tabular_cfg("heart")
    heart_csv = BASE_DIR / heart_cfg["csv_path"]
    if not heart_csv.exists():
        raise FileNotFoundError(f"Missing file: {heart_csv}")
    
    df = pd.read_csv(heart_csv)
    heart_df_raw = df.copy()
    
    print("Shape of df after loading:", df.shape)
    print("Head of df after loading:")
    print(df.head())
    
    target_col = heart_cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {heart_csv}")

    label_map = heart_cfg.get("label_map")
    if label_map:
        df[target_col] = df[target_col].map(label_map)
    df.dropna(subset=[target_col], inplace=True)
    
    df[target_col] = df[target_col].astype(int)
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"NaNs in y after final processing: {y.isnull().sum()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model_cfg = heart_cfg.get("model", {})
    model_params = model_cfg.get("params", {})
    rf = RandomForestClassifier(**model_params)
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    record_metrics("heart_tabular", {"accuracy": float(accuracy)})
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Explainable AI (SHAP) for heart attack model
    explain_tabular_model(rf, X_train, X_test, "Heart Attack (tabular)")

    if ARGS.interactive:
        print("\nInteractive input for Heart Attack (tabular)")
        input_df_raw = prompt_tabular_input(heart_df_raw, target_col)
        input_encoded = align_one_hot(input_df_raw, X.columns)
        pred = rf.predict(input_encoded)[0]
        if label_map:
            inv = {v: k for k, v in label_map.items()}
            pred_label = inv.get(int(pred), pred)
        else:
            pred_label = pred
        print("Predicted Class:", pred_label)
        record_metrics("heart_tabular_input", {"predicted_class": str(pred_label)})
    
    
    # -----------------------------
# HEART ATTACK (mimic-iv-ext-cardiac-disease-1.0.0)
if RUN_SECTIONS.get('heart_mimic', False):
    heart_mimic_dir = HEART_DIR / "mimic-iv-ext-cardiac-disease-1.0.0"
    if heart_mimic_dir.exists():
        image_root = find_first_image_root(heart_mimic_dir)
        if image_root is None:
            print("No heart image folders found under MIMIC-IV dataset (CSV-only).")
        mimic_csv = heart_mimic_dir / "heart_diagnoses.csv"
        if mimic_csv.exists():
            mimic_df = pd.read_csv(mimic_csv)
            print("MIMIC-IV heart_diagnoses.csv shape:", mimic_df.shape)
            print(mimic_df.head())
            record_metrics("heart_mimic", {"rows": int(mimic_df.shape[0]), "cols": int(mimic_df.shape[1])})
        else:
            print("MIMIC-IV dataset found, but heart_diagnoses.csv is missing.")
    else:
        print("MIMIC-IV dataset folder not found under heart attack.")
    
    
    # -----------------------------
# BLOOD CANCER (technical: Leukemia_GSE9476.csv)
if RUN_SECTIONS.get('blood_technical', False):
    technical_csv = BLOOD_DIR / "Leukemia_GSE9476.csv"
    if technical_csv.exists():
        tech_df = pd.read_csv(technical_csv)
        print("Leukemia_GSE9476.csv shape:", tech_df.shape)
        if "type" in tech_df.columns:
            y_tech = tech_df["type"]
            X_tech = tech_df.drop(columns=[c for c in ["samples", "type"] if c in tech_df.columns])
    
            le_tech = LabelEncoder()
            y_tech = le_tech.fit_transform(y_tech)
    
            X_train, X_test, y_train, y_test = train_test_split(
                X_tech, y_tech, test_size=0.2, random_state=42, stratify=y_tech
            )
    
            tech_rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            tech_rf.fit(X_train, y_train)
            y_pred = tech_rf.predict(X_test)

            tech_acc = accuracy_score(y_test, y_pred)
            print("Technical Dataset Accuracy:", tech_acc)
            print("\nTechnical Dataset Classification Report:\n")
            print(classification_report(y_test, y_pred))
            record_metrics("blood_technical", {"accuracy": float(tech_acc)})

            # Explainable AI (SHAP) for technical dataset
            explain_tabular_model(tech_rf, X_train, X_test, "Leukemia_GSE9476 (technical)")

            if ARGS.interactive:
                print("\nInteractive input for Leukemia_GSE9476 (technical)")
                input_df_raw = prompt_tabular_input(tech_df, "type")
                input_encoded = input_df_raw.reindex(columns=X_tech.columns)
                pred = tech_rf.predict(input_encoded)[0]
                pred_label = le_tech.inverse_transform([pred])[0]
                print("Predicted Class:", pred_label)
                record_metrics("blood_technical_input", {"predicted_class": str(pred_label)})
        else:
            print("Leukemia_GSE9476.csv does not have a 'type' column; skipping model.")
