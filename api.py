from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional
import json
import base64

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Header
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

tf = None
torch = None
torchvision = None
transforms = None
shap = None
GradCAM = None
ClassifierOutputTarget = None
show_cam_on_image = None

try:
    import multipart  # type: ignore  # noqa: F401
    HAS_MULTIPART = True
except Exception:
    HAS_MULTIPART = False

langchain_openai = None
langchain_core = None

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
BLOOD_DIR = BASE_DIR / "blood cancer"
LUNG_DIR = BASE_DIR / "lung cancer"
HEART_DIR = BASE_DIR / "heart attack"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CONFIG_PATH = BASE_DIR / "datasets.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    DATASETS_CONFIG = json.load(f)

API_TITLE = "Mini Project Disease Prediction API"
API_DESC = (
    "Endpoints for tabular and image predictions for blood, lung, and heart datasets. "
    "Supports training, single prediction, and batch prediction."
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESC,
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def serve_ui() -> HTMLResponse:
    ui_path = BASE_DIR / "ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))

# -----------------------------
# Auth
# -----------------------------

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("MINI_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# -----------------------------
# Helpers
# -----------------------------

def _lazy_import_tf() -> Optional[Any]:
    global tf
    if tf is None:
        try:
            import tensorflow as _tf  # type: ignore
            tf = _tf
        except Exception:
            return None
    return tf


def _lazy_import_torch() -> Optional[Any]:
    global torch
    if torch is None:
        try:
            import torch as _torch  # type: ignore
            torch = _torch
        except Exception:
            return None
    return torch


def _lazy_import_torchvision() -> Optional[Any]:
    global torchvision, transforms
    if torchvision is None or transforms is None:
        try:
            import torchvision as _torchvision  # type: ignore
            from torchvision import transforms as _transforms  # type: ignore
            torchvision = _torchvision
            transforms = _transforms
        except Exception:
            return None
    return torchvision


def _lazy_import_shap() -> Optional[Any]:
    global shap
    if shap is None:
        try:
            import shap as _shap  # type: ignore
            shap = _shap
        except Exception:
            return None
    return shap


def _lazy_import_gradcam() -> bool:
    global GradCAM, ClassifierOutputTarget, show_cam_on_image
    if GradCAM is None or ClassifierOutputTarget is None or show_cam_on_image is None:
        try:
            from pytorch_grad_cam import GradCAM as _GradCAM  # type: ignore
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget as _ClassifierOutputTarget  # type: ignore
            from pytorch_grad_cam.utils.image import show_cam_on_image as _show_cam_on_image  # type: ignore
            GradCAM = _GradCAM
            ClassifierOutputTarget = _ClassifierOutputTarget
            show_cam_on_image = _show_cam_on_image
        except Exception:
            return False
    return True


def _lazy_import_langchain() -> bool:
    global langchain_openai, langchain_core
    if langchain_openai is None or langchain_core is None:
        try:
            import langchain_openai as _langchain_openai  # type: ignore
            import langchain_core as _langchain_core  # type: ignore
            langchain_openai = _langchain_openai
            langchain_core = _langchain_core
        except Exception:
            return False
    return True


def require_tf(context: str) -> None:
    if _lazy_import_tf() is None:
        raise HTTPException(
            status_code=500,
            detail=f"TensorFlow is not available. Install tensorflow to use {context}.",
        )


def require_torch(context: str) -> None:
    if _lazy_import_torch() is None:
        raise HTTPException(
            status_code=500,
            detail=f"PyTorch is not available. Install torch to use {context}.",
        )


def require_torchvision(context: str) -> None:
    if _lazy_import_torchvision() is None:
        raise HTTPException(
            status_code=500,
            detail=f"Torchvision is not available. Install torchvision to use {context}.",
        )


def require_shap(context: str) -> None:
    if _lazy_import_shap() is None:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP is not available. Install shap to use {context}.",
        )


def require_gradcam(context: str) -> None:
    if not _lazy_import_gradcam():
        raise HTTPException(
            status_code=500,
            detail=f"pytorch-grad-cam is not available. Install pytorch-grad-cam to use {context}.",
        )


def require_langchain(context: str) -> None:
    if not _lazy_import_langchain():
        raise HTTPException(
            status_code=500,
            detail=f"LangChain is not available. Install langchain and langchain-openai to use {context}.",
        )

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


def align_one_hot(df_input: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df_input, drop_first=True)
    return df_encoded.reindex(columns=train_columns, fill_value=0)


def get_defaults_from_df(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    features = df.drop(columns=[target_col])
    defaults = {}
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            defaults[col] = float(features[col].median())
        else:
            defaults[col] = str(features[col].mode().iloc[0])
    return defaults


def shap_top_features(model, input_array: np.ndarray, feature_names: list[str], top_k: int = 5) -> list[Dict[str, Any]]:
    # Prefer SHAP if available; fallback to model feature_importances_.
    if _lazy_import_shap() is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_array)

        if isinstance(shap_values, list):
            # multiclass: use the predicted class contribution
            shap_vals = shap_values[0]
            if len(shap_values) > 1:
                shap_vals = np.mean(np.abs(np.array(shap_values)), axis=0)
        else:
            shap_vals = shap_values

        contrib = np.array(shap_vals[0]).reshape(-1)
        n = min(len(contrib), len(feature_names))
        if n == 0:
            return []
        contrib = contrib[:n]
        top_idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        result = []
        for i in top_idx:
            result.append({
                "feature": feature_names[int(i)],
                "impact": float(contrib[int(i)]),
            })
        return result

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        n = min(len(importances), len(feature_names))
        if n == 0:
            return []
        importances = importances[:n]
        top_idx = np.argsort(importances)[::-1][:top_k]
        return [
            {"feature": feature_names[int(i)], "impact": float(importances[int(i)])}
            for i in top_idx
        ]

    return []


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    require_tf("Grad-CAM for blood images")
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str, pred_index: Optional[int] = None) -> np.ndarray:
    require_tf("Grad-CAM for blood images")
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
# Caches
# -----------------------------

TABULAR_MODELS: Dict[str, Dict[str, Any]] = {}
IMAGE_MODELS: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Training functions (tabular)
# -----------------------------

def get_tabular_config(disease: str) -> Dict[str, Any]:
    cfg = DATASETS_CONFIG.get("tabular", {})
    if disease not in cfg:
        raise HTTPException(status_code=400, detail="Unknown disease. Use blood, lung, heart, or blood_clinical.")
    return cfg[disease]


def train_tabular_from_config(disease: str) -> Dict[str, Any]:
    cfg = get_tabular_config(disease)
    csv_path = BASE_DIR / cfg["csv_path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "strip_columns" in cfg and cfg["strip_columns"]:
        df.columns = [c.strip() for c in df.columns]

    drop_cols = cfg.get("drop_columns") or []
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

    sample_rows = cfg.get("sample_rows")
    if isinstance(sample_rows, int) and sample_rows > 0 and sample_rows < len(df):
        df = df.sample(sample_rows, random_state=42)

    df_raw = df.copy()
    df = df.fillna(df.median(numeric_only=True))

    target_col = cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    label_map = cfg.get("label_map")
    label_encoder = None
    if label_map:
        df[target_col] = df[target_col].map(label_map)
    elif not pd.api.types.is_numeric_dtype(df[target_col]):
        label_encoder = LabelEncoder()
        df[target_col] = label_encoder.fit_transform(df[target_col])

    df.dropna(subset=[target_col], inplace=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X, drop_first=True)
    X_df = X.copy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    if cfg.get("use_smote"):
        if SMOTE is None:
            raise HTTPException(status_code=500, detail="imblearn is not available. Install imbalanced-learn to use SMOTE.")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train_df, y_train)
    else:
        X_train = X_train_df

    scaler = None
    if cfg.get("use_scaler"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test_df)
    else:
        X_test = X_test_df

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "RandomForestClassifier")
    params = model_cfg.get("params", {})

    if model_type != "RandomForestClassifier":
        raise ValueError(f"Unsupported model type: {model_type}")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "label_map": label_map,
        "feature_columns": X_df.columns,
        "raw_df": df_raw,
        "target_col": target_col,
        "accuracy": float(acc),
    }


# -----------------------------
# Training / loading (images)
# -----------------------------

def get_image_config(disease: str) -> Dict[str, Any]:
    cfg = DATASETS_CONFIG.get("images", {})
    if disease not in cfg:
        raise HTTPException(status_code=400, detail="Unknown disease. Use blood or lung for images.")
    return cfg[disease]


def get_image_model_from_config(disease: str) -> Dict[str, Any]:
    cfg = get_image_config(disease)
    model_type = cfg.get("model_type")
    model_path = BASE_DIR / cfg.get("model_path", "")
    dataset_path = BASE_DIR / cfg.get("dataset_path", "")

    if model_type == "keras_cnn":
        require_tf("blood image predictions")
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path.name} not found. Train blood_images in project.py first.")
        model = tf.keras.models.load_model(model_path)

        if not dataset_path.exists():
            dataset_path = find_first_image_root(BASE_DIR)
        if dataset_path is None or not dataset_path.exists():
            raise FileNotFoundError("No image dataset found for blood.")

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=1,
            class_mode="categorical",
            shuffle=False,
        )
        class_map = {v: k for k, v in generator.class_indices.items()}
        return {"model": model, "class_map": class_map}

    if model_type == "torch_densenet":
        require_torch("lung image predictions")
        require_torchvision("lung image predictions")
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path.name} not found. Train lung_images in project.py first.")
        if not dataset_path.exists():
            raise FileNotFoundError("final_dataset not found. Run lung_images section first.")

        classes = sorted([p.name for p in dataset_path.iterdir() if p.is_dir()])
        model = torchvision.models.densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return {"model": model, "classes": classes, "device": device, "transform": transform}

    raise ValueError(f"Unsupported model_type: {model_type}")


def load_image_model_bundle(disease: str, context: str) -> Dict[str, Any]:
    try:
        return get_image_model_from_config(disease)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{context} failed: {exc}") from exc


# -----------------------------
# API endpoints
# -----------------------------

@app.get("/health", dependencies=[Depends(require_api_key)])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/schema/tabular/{disease}", dependencies=[Depends(require_api_key)])
def schema_tabular(disease: str) -> Dict[str, Any]:
    disease = disease.lower()
    if disease not in TABULAR_MODELS:
        TABULAR_MODELS[disease] = train_tabular_from_config(disease)
    bundle = TABULAR_MODELS[disease]
    target_col = bundle["target_col"]
    defaults = get_defaults_from_df(bundle["raw_df"], target_col)
    fields = []
    for name, default in defaults.items():
        dtype = "number" if isinstance(default, (int, float)) else "text"
        fields.append({"name": name, "type": dtype, "default": default})

    # If schema is huge (e.g., clinical dataset), return a hint for CSV upload.
    if len(fields) > 200:
        return {
            "disease": disease,
            "target_col": target_col,
            "fields": [],
            "defaults": {},
            "note": "Dataset has many columns. Use CSV upload for clinical predictions.",
        }

    return {
        "disease": disease,
        "target_col": target_col,
        "fields": fields,
        "defaults": defaults,
    }


@app.post("/train/tabular/{disease}", dependencies=[Depends(require_api_key)])
def train_tabular(disease: str) -> Dict[str, Any]:
    disease = disease.lower()
    try:
        model_bundle = train_tabular_from_config(disease)
        TABULAR_MODELS[disease] = model_bundle
        return {"status": "trained", "accuracy": model_bundle.get("accuracy")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/tabular/{disease}", dependencies=[Depends(require_api_key)])
def predict_tabular(disease: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    disease = disease.lower()
    if disease not in TABULAR_MODELS:
        # auto-train on first request
        TABULAR_MODELS[disease] = train_tabular_from_config(disease)

    bundle = TABULAR_MODELS[disease]

    # Build input row
    df_raw = bundle["raw_df"]
    target_col = bundle["target_col"]
    features = df_raw.drop(columns=[target_col])

    row = {}
    for col in features.columns:
        if col in payload:
            row[col] = payload[col]
        else:
            # Use defaults from training data
            if pd.api.types.is_numeric_dtype(features[col]):
                row[col] = float(features[col].median())
            else:
                row[col] = features[col].mode().iloc[0]

    input_df = pd.DataFrame([row])

    if disease == "blood":
        input_encoded = align_one_hot(input_df, bundle["feature_columns"])
        if bundle.get("scaler") is not None:
            input_encoded = bundle["scaler"].transform(input_encoded)
        pred = int(bundle["model"].predict(input_encoded)[0])
        if bundle.get("label_map"):
            inv = {v: k for k, v in bundle["label_map"].items()}
            return {"prediction": str(inv.get(pred, pred))}
        if bundle.get("label_encoder") is not None:
            label = bundle["label_encoder"].inverse_transform([pred])[0]
            return {"prediction": str(label)}
        return {"prediction": str(pred)}

    if disease == "lung":
        input_encoded = align_one_hot(input_df, bundle["feature_columns"])
        pred = int(bundle["model"].predict(input_encoded)[0])
        inv = {v: k for k, v in (bundle.get("label_map") or {}).items()}
        return {"prediction": str(inv.get(pred, "YES" if pred == 1 else "NO"))}

    if disease == "heart":
        input_encoded = align_one_hot(input_df, bundle["feature_columns"])
        pred = int(bundle["model"].predict(input_encoded)[0])
        inv = {v: k for k, v in (bundle.get("label_map") or {}).items()}
        return {"prediction": str(inv.get(pred, "Yes" if pred == 1 else "No"))}

    raise HTTPException(status_code=400, detail="Unknown disease.")


@app.post("/predict/tabular/{disease}/explain", dependencies=[Depends(require_api_key)])
def predict_tabular_explain(disease: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    disease = disease.lower()
    if disease not in TABULAR_MODELS:
        TABULAR_MODELS[disease] = train_tabular_from_config(disease)

    bundle = TABULAR_MODELS[disease]
    df_raw = bundle["raw_df"]
    target_col = bundle["target_col"]
    features = df_raw.drop(columns=[target_col])

    row = {}
    for col in features.columns:
        if col in payload:
            row[col] = payload[col]
        else:
            if pd.api.types.is_numeric_dtype(features[col]):
                row[col] = float(features[col].median())
            else:
                row[col] = features[col].mode().iloc[0]

    input_df = pd.DataFrame([row])
    input_encoded = align_one_hot(input_df, bundle["feature_columns"])
    model_input = input_encoded
    if bundle.get("scaler") is not None:
        model_input = bundle["scaler"].transform(input_encoded)

    pred = int(bundle["model"].predict(model_input)[0])
    if bundle.get("label_map"):
        inv = {v: k for k, v in bundle["label_map"].items()}
        pred_label = inv.get(pred, pred)
    elif bundle.get("label_encoder") is not None:
        pred_label = bundle["label_encoder"].inverse_transform([pred])[0]
    else:
        pred_label = pred

    top_features = shap_top_features(bundle["model"], model_input, list(bundle["feature_columns"]))
    return {"prediction": str(pred_label), "top_features": top_features}


if HAS_MULTIPART:
    @app.post("/predict/image/{disease}", dependencies=[Depends(require_api_key)])
    async def predict_image(disease: str, file: UploadFile = File(...)) -> Dict[str, Any]:
        disease = disease.lower()

        if disease == "blood":
            require_tf("blood image predictions")
            if disease not in IMAGE_MODELS:
                IMAGE_MODELS[disease] = load_image_model_bundle(disease, "Blood image prediction")
            bundle = IMAGE_MODELS[disease]

            image_bytes = await file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((224, 224))
            arr = tf.keras.utils.img_to_array(img)
            arr = np.expand_dims(arr, axis=0) / 255.0
            preds = bundle["model"].predict(arr)
            idx = int(np.argmax(preds[0]))
            label = bundle["class_map"].get(idx, str(idx))
            return {"prediction": str(label)}

        if disease == "lung":
            require_torch("lung image predictions")
            require_torchvision("lung image predictions")
            if disease not in IMAGE_MODELS:
                IMAGE_MODELS[disease] = load_image_model_bundle(disease, "Lung image prediction")
            bundle = IMAGE_MODELS[disease]

            image_bytes = await file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = bundle["transform"](img).unsqueeze(0).to(bundle["device"])
            with torch.no_grad():
                outputs = bundle["model"](tensor)
                pred_idx = int(outputs.argmax(dim=1).item())
            classes = bundle["classes"]
            label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            return {"prediction": str(label)}

        raise HTTPException(status_code=400, detail="Unknown disease. Use blood or lung for images.")


    @app.post("/predict/image/{disease}/explain", dependencies=[Depends(require_api_key)])
    async def predict_image_explain(disease: str, file: UploadFile = File(...)) -> Dict[str, Any]:
        disease = disease.lower()
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if disease == "blood":
            require_tf("blood image explanations")
            if disease not in IMAGE_MODELS:
                IMAGE_MODELS[disease] = load_image_model_bundle(disease, "Blood Grad-CAM")
            bundle = IMAGE_MODELS[disease]
            model = bundle["model"]

            img_resized = img.resize((224, 224))
            arr = tf.keras.utils.img_to_array(img_resized)
            arr = np.expand_dims(arr, axis=0) / 255.0
            preds = model.predict(arr)
            idx = int(np.argmax(preds[0]))
            label = bundle["class_map"].get(idx, str(idx))

            last_conv = get_last_conv_layer_name(model)
            heatmap = make_gradcam_heatmap(arr, model, last_conv)

            # overlay
            heatmap = np.uint8(255 * heatmap)
            heatmap = np.stack([heatmap] * 3, axis=-1)
            overlay = (0.6 * np.array(img_resized) + 0.4 * heatmap).astype(np.uint8)
            out_img = Image.fromarray(overlay)

            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"prediction": str(label), "gradcam_png_base64": b64}

        if disease == "lung":
            require_torch("lung image explanations")
            require_torchvision("lung image explanations")
            require_gradcam("lung image explanations")
            if disease not in IMAGE_MODELS:
                IMAGE_MODELS[disease] = load_image_model_bundle(disease, "Lung Grad-CAM")
            bundle = IMAGE_MODELS[disease]

            tensor = bundle["transform"](img).unsqueeze(0).to(bundle["device"])
            with torch.no_grad():
                outputs = bundle["model"](tensor)
                pred_idx = int(outputs.argmax(dim=1).item())
            classes = bundle["classes"]
            label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

            target_layers = [bundle["model"].features[-1]]
            cam = GradCAM(model=bundle["model"], target_layers=target_layers)
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
            rgb_img = np.array(img.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            out_img = Image.fromarray(visualization.astype(np.uint8))
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"prediction": str(label), "gradcam_png_base64": b64}

        raise HTTPException(status_code=400, detail="Unknown disease. Use blood or lung for images.")


# -----------------------------
# Batch endpoints
# -----------------------------

@app.post("/predict/tabular/{disease}/batch", dependencies=[Depends(require_api_key)])
def predict_tabular_batch(disease: str, payloads: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payloads, list) or len(payloads) == 0:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty list.")

    preds = []
    for payload in payloads:
        preds.append(predict_tabular(disease, payload)["prediction"])
    return {"predictions": preds}


if HAS_MULTIPART:
    @app.post("/predict/tabular/{disease}/csv", dependencies=[Depends(require_api_key)])
    async def predict_tabular_csv(disease: str, file: UploadFile = File(...)) -> Dict[str, Any]:
        """
        Predict from a CSV upload. Uses the first row for a single prediction.
        Intended for large clinical datasets where manual input is not feasible.
        """
        disease = disease.lower()
        if disease not in TABULAR_MODELS:
            TABULAR_MODELS[disease] = train_tabular_from_config(disease)
        bundle = TABULAR_MODELS[disease]

        data = await file.read()
        df = pd.read_csv(io.BytesIO(data))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")

        target_col = bundle["target_col"]
        if target_col in df.columns:
            df = df.drop(columns=[target_col])

        row = df.iloc[0].to_dict()
        return predict_tabular_explain(disease, row)


# -----------------------------
# LLM Chat (for Flutter)
# -----------------------------

@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple LLM chat endpoint for Flutter.
    Payload:
      {
        "message": "user text",
        "history": [{"role": "user"|"assistant"|"system", "content": "..."}]
      }
    """
    require_langchain("LLM chat")

    message = payload.get("message")
    history = payload.get("history") or []
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history must be a list")

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", ""))
        if not content:
            continue
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=str(message)))

    llm = ChatOpenAI(model=model_name, temperature=0.2)
    response = llm.invoke(messages)
    return {"response": response.content, "model": model_name}


if HAS_MULTIPART:
    @app.post("/predict/image/{disease}/batch", dependencies=[Depends(require_api_key)])
    async def predict_image_batch(disease: str, files: list[UploadFile] = File(...)) -> Dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        predictions = []
        for f in files:
            pred = await predict_image(disease, f)
            predictions.append(pred["prediction"])
        return {"predictions": predictions}
