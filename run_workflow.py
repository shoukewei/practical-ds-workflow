# run_modeling_pipeline.py
from pathlib import Path

from modules.data_io import load_dataset
from modules.eda import missing_summary, skewness_summary
from modules.splitting import create_split, save_split
from modules.pipeline import PreprocessingPipeline, save_config
from modules.modeling import (
    compute_vif, select_significant_features, ModelRegistry, save_model
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def main():
    base = Path(__file__).parent

    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

    print("1/9 — Load dataset")
    df = load_dataset(url, index_col=0,
                      required_columns=["TV", "radio", "newspaper", "sales"])

    print("2/9 — Quick EDA")
    print("Missing columns with values:")
    print(missing_summary(df).to_string(index=False))
    print("\nSkewness summary:")
    print(skewness_summary(df)[["column", "skewness", "high_skew"]].to_string(index=False))

    print("3/9 — Create split and save")
    split = create_split(df, target="sales", test_size=0.2, random_state=42)
    save_split(split, str(base / "data" / "splits" / "advertising"))

    print("4/9 — Build preprocessing pipeline")
    config = {
        "missing":  {"strategies": {"TV": "median", "radio": "mean", "newspaper": 0.0}, "indicator_columns": []},
        "outliers": {"columns": ["TV", "newspaper"], "method": "iqr", "multiplier": 1.5},
        "scaling":  {"columns": ["TV", "radio", "newspaper"], "method": "standard"},
    }
    pipeline = PreprocessingPipeline(config)

    print("5/9 — Fit and transform training data")
    X_train_clean = pipeline.fit_transform(split.X_train)
    X_test_clean = pipeline.transform(split.X_test)

    print("6/9 — VIF analysis")
    print(compute_vif(X_train_clean[["TV", "radio", "newspaper"]]))

    print("7/9 — Feature selection (OLS p-values)")
    X_sig, sig_summary = select_significant_features(X_train_clean, split.y_train, alpha=0.05)
    print("Significant features:", list(X_sig.columns))

    print("8/9 — Model registry training and comparison")
    X_tr, X_val, y_tr, y_val = (
        X_train_clean, X_train_clean.sample(frac=0.8, random_state=42),
        split.y_train, split.y_train.sample(frac=0.8, random_state=42)
    )
    # Use a small deterministic registry as in the chapter
    registry = (
        ModelRegistry(task="regression")
        .register("linear_regression", LinearRegression())
        .register("ridge", Ridge(alpha=1.0))
        .register("random_forest", RandomForestRegressor(n_estimators=100, random_state=42))
        .register("gradient_boosting", GradientBoostingRegressor(n_estimators=100, random_state=42))
    )

    results = registry.fit_all(X_tr, y_tr, X_val, y_val)
    print("Model comparison (validation set):")
    print(results[["model", "r2", "rmse"]].to_string(index=False))

    print("9/9 — Final evaluation and save artifacts")
    best_name, best_model = registry.get_best()
    final = registry.models_[best_name]
    # Evaluate on the held-out test set
    from modules.modeling import evaluate_regression
    final_metrics = evaluate_regression(final, X_test_clean, split.y_test, label="test")
    print(f"\nBest model: {best_name}")
    print("Final metrics:")
    print(final_metrics)

    # Save config, pipeline and model
    save_config(config, str(base / "artifacts" / "models" / "pipeline_config.json"))
    pipeline.save(str(base / "artifacts" / "models" / "preprocessing_pipeline.pkl"))
    save_model(final, str(base / "artifacts" / "models" / "best_model.joblib"))


if __name__ == "__main__":
    main()
