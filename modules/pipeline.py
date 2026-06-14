# modules/pipeline.py

import json
import pickle
import pandas as pd
from pathlib import Path
from modules.preprocessing import (
    add_missing_indicator, compute_fill_values, fill_missing,
    compute_iqr_bounds, cap_with_bounds,
    compute_scaling_params, apply_scaling,
)
from modules.feature_engineering import (
    add_family_features, log_transform_feature, add_polynomial_features,
    fit_onehot_encoder, apply_onehot_encoder,
)


def save_config(config: dict, path: str) -> None:
    """Save a pipeline configuration dictionary to a JSON file."""
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to '{path}'")


def load_config(path: str) -> dict:
    """Load a pipeline configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)



class PreprocessingPipeline:
    """
    A configurable, fit-transform preprocessing pipeline.

    Applies missing data handling, outlier capping, and feature scaling
    in sequence. All parameters are computed from training data only.

    Parameters
    ----------
    config : dict
        Pipeline configuration. Expected keys:
        - 'missing': dict with 'strategies' and optional 'indicator_columns'
        - 'outliers': dict with 'columns', 'method', and 'multiplier'
        - 'scaling': dict with 'columns' and 'method'

    Attributes
    ----------
    fill_values_ : dict
        Imputation values computed during fit (from training data).
    iqr_bounds_ : dict
        IQR outlier bounds computed during fit (from training data).
    scaling_params_ : dict
        Scaling parameters computed during fit (from training data).
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> pipeline = PreprocessingPipeline(config)
    >>> X_train_clean = pipeline.fit_transform(X_train)
    >>> X_test_clean  = pipeline.transform(X_test)
    """

    def __init__(self, config: dict):
        self.config        = config
        self.fill_values_  = {}
        self.iqr_bounds_   = {}
        self.scaling_params_ = {}
        self.ohe_encoder_  = None
        self.is_fitted_    = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """
        Compute all preprocessing parameters from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Never call fit() on test data.

        Returns
        -------
        PreprocessingPipeline
            Returns self to allow method chaining.
        """
        # --- Feature engineering (fit-only steps) ---
        fe_cfg = self.config.get("feature_engineering", {})
        df_fe = df.copy()

        # Family features (deterministic)
        if fe_cfg.get("family_features"):
            df_fe = add_family_features(df_fe)

        # Log transforms (deterministic)
        for col in fe_cfg.get("log_columns", []):
            df_fe = log_transform_feature(df_fe, col, new_name=f"{col}_log")

        # Polynomial features (deterministic)
        poly_cfg = fe_cfg.get("polynomial", {})
        if poly_cfg.get("columns"):
            df_fe = add_polynomial_features(
                df_fe,
                columns=poly_cfg["columns"],
                degree=poly_cfg.get("degree", 2),
            )

        # One-hot encoder: fit on training data and store
        ohe_cols = fe_cfg.get("onehot_columns", [])
        if ohe_cols:
            self.ohe_encoder_ = fit_onehot_encoder(df_fe, ohe_cols)
            # Apply encoder to the engineered df so subsequent steps
            # (imputation/scaling) see the encoded columns
            df_fe = apply_onehot_encoder(df_fe, self.ohe_encoder_)

        # --- Missing data parameters ---
        missing_cfg = self.config.get("missing", {})
        strategies  = missing_cfg.get("strategies", {})
        if strategies:
            # compute fill values on the feature-engineered training frame
            self.fill_values_ = compute_fill_values(df_fe, strategies)

        # --- Outlier parameters ---
        outlier_cfg = self.config.get("outliers", {})
        outlier_cols = outlier_cfg.get("columns", [])
        if outlier_cols:
            self.iqr_bounds_ = compute_iqr_bounds(
                df_fe,
                columns=outlier_cols,
                multiplier=outlier_cfg.get("multiplier", 1.5),
            )

        # --- Scaling parameters ---
        scaling_cfg  = self.config.get("scaling", {})
        scaling_cols = scaling_cfg.get("columns", [])
        if scaling_cols:
            self.scaling_params_ = compute_scaling_params(
                df_fe,
                columns=scaling_cols,
                method=scaling_cfg.get("method", "standard"),
            )

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing parameters to a DataFrame.

        Can be called on any split — train, validation, or test.
        Parameters are always those computed during fit().

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.

        Raises
        ------
        RuntimeError
            If transform() is called before fit().
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit() on training data first."
            )

        df = df.copy()

        # --- Feature engineering (apply with stored encoders) ---
        fe_cfg = self.config.get("feature_engineering", {})

        if fe_cfg.get("family_features"):
            df = add_family_features(df)

        for col in fe_cfg.get("log_columns", []):
            df = log_transform_feature(df, col, new_name=f"{col}_log")

        poly_cfg = fe_cfg.get("polynomial", {})
        if poly_cfg.get("columns"):
            df = add_polynomial_features(
                df,
                columns=poly_cfg["columns"],
                degree=poly_cfg.get("degree", 2),
            )

        ohe_cols = fe_cfg.get("onehot_columns", [])
        if ohe_cols:
            if self.ohe_encoder_ is None:
                raise RuntimeError("One-hot encoder not fitted. Call fit() first.")
            df = apply_onehot_encoder(df, self.ohe_encoder_)

        # Step 1: Add missing indicators (before imputation)
        missing_cfg      = self.config.get("missing", {})
        indicator_cols   = missing_cfg.get("indicator_columns", [])
        if indicator_cols:
            df = add_missing_indicator(df, indicator_cols)

        # Step 2: Impute missing values
        if self.fill_values_:
            df = fill_missing(df, self.fill_values_)

        # Step 3: Cap outliers
        if self.iqr_bounds_:
            df = cap_with_bounds(df, self.iqr_bounds_)

        # Step 4: Scale features
        if self.scaling_params_:
            df = apply_scaling(
                df,
                self.scaling_params_,
                method=self.config.get("scaling", {}).get("method", "standard"),
            )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on df and return the transformed result.

        Equivalent to fit(df).transform(df). Use on training data only.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        pd.DataFrame
            Preprocessed training data.
        """
        return self.fit(df).transform(df)

    def get_params(self) -> dict:
        """
        Return all fitted parameters as a dictionary.

        Returns
        -------
        dict
            Keys: fill_values, iqr_bounds, scaling_params.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted.")
        return {
            "fill_values":    self.fill_values_,
            "iqr_bounds":     self.iqr_bounds_,
            "scaling_params": self.scaling_params_,
        }

    def save(self, path: str) -> None:
        """
        Serialise the fitted pipeline to disk.

        Parameters
        ----------
        path : str
            File path for the saved pipeline (.pkl).
        """
        if not self.is_fitted_:
            raise RuntimeError("Fit the pipeline before saving.")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to '{path}'")

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        """
        Load a fitted pipeline from disk.

        Parameters
        ----------
        path : str
            Path to a saved pipeline file (.pkl).

        Returns
        -------
        PreprocessingPipeline
            The loaded, fitted pipeline.
        """
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from '{path}'")
        return pipeline
 
def run_feature_engineering(
    df: pd.DataFrame,
    config: dict,
    ohe_encoder: dict = None,
    is_train: bool = True
) -> tuple:
    """
    Apply configured feature engineering steps to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    config : dict
        Feature engineering config (the 'feature_engineering' sub-dict).
    ohe_encoder : dict, optional
        Fitted one-hot encoder. Required if onehot_columns is specified.
    is_train : bool, optional
        If True and ohe_encoder is None, fits the encoder on df.
        Default is True.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        (transformed DataFrame, fitted ohe_encoder)
    """
    df = df.copy()

    # Family features
    if config.get("family_features"):
        df = add_family_features(df)

    # Log transformations
    for col in config.get("log_columns", []):
        df = log_transform_feature(df, col, new_name=f"{col}_log")

    # Polynomial features
    poly_cfg = config.get("polynomial", {})
    if poly_cfg.get("columns"):
        df = add_polynomial_features(
            df,
            columns=poly_cfg["columns"],
            degree=poly_cfg.get("degree", 2)
        )

    # One-hot encoding
    ohe_cols = config.get("onehot_columns", [])
    if ohe_cols:
        if is_train and ohe_encoder is None:
            ohe_encoder = fit_onehot_encoder(df, ohe_cols)
        df = apply_onehot_encoder(df, ohe_encoder)

    return df, ohe_encoder

def summarise_pipeline(pipeline: PreprocessingPipeline) -> None:
    """Print a human-readable summary of a fitted `PreprocessingPipeline`.

    Mirrors the helper used in the chapter scripts so `workflow.py` can call
    `summarise_pipeline(pipeline)` after fitting.
    """
    if not getattr(pipeline, "is_fitted_", False):
        print("Pipeline is not fitted.")
        return

    config = pipeline.config
    params = pipeline.get_params()

    print("=" * 50)
    print("PREPROCESSING PIPELINE SUMMARY")
    print("=" * 50)

    # Missing data
    missing_cfg = config.get("missing", {})
    strategies = missing_cfg.get("strategies", {})
    indicators = missing_cfg.get("indicator_columns", [])
    if strategies:
        print("\n[1] Missing Data Imputation")
        for col, strategy in strategies.items():
            fitted_val = params["fill_values"].get(col, strategy)
            val_str = f"{fitted_val:.4f}" if isinstance(fitted_val, float) else str(fitted_val)
            print(f"    {col}: strategy='{strategy}', fill_value={val_str}")
        if indicators:
            print(f"    Indicators added for: {indicators}")

    # Outlier capping
    outlier_cfg = config.get("outliers", {})
    if outlier_cfg.get("columns"):
        print(f"\n[2] Outlier Capping "
              f"(method={outlier_cfg.get('method','iqr')}, "
              f"multiplier={outlier_cfg.get('multiplier', 1.5)})")
        for col, (lo, hi) in params["iqr_bounds"].items():
            print(f"    {col}: [{lo:.2f}, {hi:.2f}]")

    # Scaling
    scaling_cfg = config.get("scaling", {})
    if scaling_cfg.get("columns"):
        print(f"\n[3] Feature Scaling (method={scaling_cfg.get('method','standard')})")
        for col, p in params["scaling_params"].items():
            if scaling_cfg.get("method") == "standard":
                print(f"    {col}: mean={p['mean']:.4f}, std={p['std']:.4f}")
            else:
                print(f"    {col}: min={p['min']:.4f}, max={p['max']:.4f}")

    print("=" * 50)
    