from __future__ import annotations

import argparse
from typing import Any, Dict

import pandas as pd

import api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual terminal predictor (no API server)")
    parser.add_argument(
        "--disease",
        default="",
        help="blood | lung | heart (optional, will prompt if missing)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full training configuration (slower). Default uses a faster model.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2000,
        help="Rows to sample for fast training (ignored with --full).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many reason features to show",
    )
    return parser.parse_args()


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
        features.select_dtypes(exclude="number").mode().iloc[0].to_dict()
        if not features.select_dtypes(exclude="number").empty
        else {}
    )
    row: Dict[str, Any] = {}
    for col in features.columns:
        is_numeric = pd.api.types.is_numeric_dtype(features[col])
        default = defaults_num.get(col) if is_numeric else defaults_cat.get(col)
        row[col] = prompt_for_value(col, default, is_numeric)
    return pd.DataFrame([row])


def map_prediction(bundle: Dict[str, Any], pred: int) -> str:
    if bundle.get("label_map"):
        inv = {v: k for k, v in bundle["label_map"].items()}
        return str(inv.get(pred, pred))
    if bundle.get("label_encoder") is not None:
        return str(bundle["label_encoder"].inverse_transform([pred])[0])
    return str(pred)


def top_reasons(bundle: Dict[str, Any], model_input: Any, top_k: int) -> list[Dict[str, Any]]:
    try:
        return api.shap_top_features(bundle["model"], model_input, list(bundle["feature_columns"]), top_k=top_k)
    except Exception:
        model = bundle["model"]
        if not hasattr(model, "feature_importances_"):
            return []
        importances = model.feature_importances_
        cols = list(bundle["feature_columns"])
        ranked = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        try:
            if hasattr(model_input, "toarray"):
                values = model_input.toarray()[0]
            else:
                values = model_input[0]
        except Exception:
            values = None
        for idx, (col, imp) in enumerate(ranked):
            value = None
            if values is not None:
                try:
                    col_index = cols.index(col)
                    value = float(values[col_index])
                except Exception:
                    value = None
            results.append({"feature": col, "importance": float(imp), "value": value})
        return results


def main() -> None:
    args = parse_args()
    disease = args.disease.strip().lower()
    if not disease:
        disease = input("Select disease (blood | lung | heart): ").strip().lower()

    if disease not in {"blood", "lung", "heart"}:
        raise SystemExit("Invalid disease. Use: blood, lung, heart")

    # Speed up interactive runs by reducing heavy model sizes unless --full is used.
    if not args.full:
        cfg = api.get_tabular_config(disease).copy()
        model_cfg = dict(cfg.get("model", {}))
        params = dict(model_cfg.get("params", {}))
        if disease == "heart":
            params["n_estimators"] = min(80, int(params.get("n_estimators", 80)))
            params["max_depth"] = min(8, int(params.get("max_depth", 8)))
            params["min_samples_split"] = max(10, int(params.get("min_samples_split", 10)))
            params["min_samples_leaf"] = max(5, int(params.get("min_samples_leaf", 5)))
        elif disease == "blood":
            params["n_estimators"] = min(100, int(params.get("n_estimators", 100)))
            params["max_depth"] = min(8, int(params.get("max_depth", 8)))
        elif disease == "lung":
            params["n_estimators"] = min(80, int(params.get("n_estimators", 80)))
            params["max_depth"] = min(8, int(params.get("max_depth", 8)))
        model_cfg["params"] = params
        cfg["model"] = model_cfg
        cfg["use_smote"] = False
        cfg["use_scaler"] = False
        cfg["sample_rows"] = max(200, int(args.sample_rows))

        # Inline train using the adjusted config.
        original_cfg = api.DATASETS_CONFIG["tabular"][disease]
        api.DATASETS_CONFIG["tabular"][disease] = cfg
        try:
            bundle = api.train_tabular_from_config(disease)
        finally:
            api.DATASETS_CONFIG["tabular"][disease] = original_cfg
    else:
        bundle = api.train_tabular_from_config(disease)
    df_raw = bundle["raw_df"]
    target_col = bundle["target_col"]

    print("\nPlease answer the following questions:")
    input_df = prompt_tabular_input(df_raw, target_col)

    input_encoded = api.align_one_hot(input_df, bundle["feature_columns"])
    model_input = input_encoded
    if bundle.get("scaler") is not None:
        model_input = bundle["scaler"].transform(input_encoded)

    pred = int(bundle["model"].predict(model_input)[0])
    label = map_prediction(bundle, pred)

    reasons = top_reasons(bundle, model_input, args.top_k)

    print("\nPrediction:")
    print(label)

    if reasons:
        print("\nTop reasons (feature importance):")
        for item in reasons:
            if "impact" in item:
                print(f"- {item['feature']}: impact={item['impact']:.4f}")
            else:
                value = item.get("value")
                if value is None:
                    print(f"- {item['feature']}: importance={item['importance']:.4f}")
                else:
                    print(f"- {item['feature']}: importance={item['importance']:.4f}, value={value}")
    else:
        print("\nNo explanation available (missing SHAP or feature importances).")


if __name__ == "__main__":
    main()
