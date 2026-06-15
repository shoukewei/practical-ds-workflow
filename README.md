# Reusable Data Science Workflow

A modular, reproducible workflow for tabular machine learning projects — built from a small set of focused, reusable, and well-tested modules rather than a single monolithic notebook.

This project is the companion reusable workflow system developed throughout the book *Practical Data Science Engineering: Building Reusable Workflows and Pipelines in Python*, published by [Deepsim Press](https://press.deepsim.ca/?utm_source=chatgpt.com).

Each module owns one stage of the pipeline (loading, exploring, preprocessing, splitting, feature engineering, modelling) and follows a consistent **fit on training data, transform any split** pattern, so the same transformations are applied identically to training, validation, and test data. The workflow demonstrates how reusable data science systems can be assembled from modular components for:

* data loading and validation
* exploratory data analysis (EDA)
* preprocessing and feature engineering
* reproducible dataset splitting
* model training and evaluation
* workflow orchestration
* artifact organization and persistence

Rather than treating data science as a collection of isolated scripts or notebook experiments, this project emphasizes:

* modularity
* reproducibility
* maintainability
* composability
* workflow consistency

The repository is intended both as:

* a companion resource for readers of the book
* a reusable starter template for practical machine learning workflows

The complete workflow architecture evolves progressively throughout the book and culminates in this integrated reusable system.

## Project Structure

```text
project/
├── data/
│   └── splits/            # train/val/test CSVs written by save_split()
├── artifacts/
│   ├── models/             # trained models (save_model / load_model)
│   ├── reports/            # EDA and evaluation reports
│   ├── transformed_data/   # preprocessed feature sets
│   └── plots/
├── modules/
│   ├── data_io.py           # loading, validation, caching, saving
│   ├── eda.py                # summary, missing-value, skewness, correlation reports
│   ├── preprocessing.py      # imputation, outlier capping, scaling
│   ├── splitting.py          # train/test, three-way, stratified, time-based splits
│   ├── pipeline.py            # PreprocessingPipeline, run_feature_engineering, config I/O
│   ├── feature_engineering.py # encodings, ratios, polynomial terms, binning
│   └── modeling.py            # training, evaluation, diagnostics, ModelRegistry
├── notebooks/               # exploratory and prototyping notebooks (see below)
├── tests/                   # pytest smoke and unit tests
├── models/                   # saved config, pipeline, and final model
└── run_workflow.py           # orchestration entry point
```

## Modules at a Glance

| Module | Responsibility | Key functions / classes |
| --- | --- | --- |
| `data_io.py` | Load and validate data from CSV/Parquet/JSON/Excel, with caching | `load_dataset()`, `save_dataset()`, `validate_columns()`, `validate_dtypes()` |
| `eda.py` | Reusable exploratory reports | `describe_df()`, `missing_summary()`, `skewness_summary()`, `correlation_summary()` |
| `preprocessing.py` | Missing-value, outlier, and scaling logic, split into compute/apply pairs | `compute_fill_values()` / `fill_missing()`, `compute_iqr_bounds()` / `cap_with_bounds()`, `compute_scaling_params()` / `apply_scaling()`, `StandardScaler` |
| `splitting.py` | Reproducible dataset splitting and CV | `create_split()`, `create_three_way_split()`, `create_stratified_split()`, `create_time_split()`, `save_split()` / `load_split()`, `cross_validate_model()` |
| `pipeline.py` | Ties preprocessing and feature engineering into one fit/transform object | `PreprocessingPipeline`, `run_feature_engineering()`, `save_config()` / `load_config()`, `summarise_pipeline()` |
| `feature_engineering.py` | Derived features and categorical encodings | `add_family_features()`, `add_ratio_feature()`, `add_polynomial_features()`, one-hot / ordinal / target encoders, `log_transform_feature()`, `bin_feature()` |
| `modeling.py` | Training, evaluation, diagnostics, model comparison | `train_model()`, `evaluate_regression()` / `evaluate_classification()`, `compute_vif()`, `select_significant_features()`, `ModelRegistry`, `save_model()` / `load_model()` |

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- statsmodels
- joblib
- descripstats
- normscaler

Install everything with:

```bash
pip install pandas numpy scikit-learn statsmodels joblib descripstats normscaler
```

## Quickstart

The full end-to-end workflow — load, explore, split, preprocess, engineer features, train, compare,
and save — is wired up in `run_workflow.py`. Run it with:

```bash
python run_workflow.py
```

A minimal version of what it does:

```python
from modules.data_io import load_dataset
from modules.splitting import create_three_way_split, save_split
from modules.pipeline import PreprocessingPipeline, save_config
from modules.modeling import ModelRegistry, evaluate_regression, save_model
from sklearn.linear_model import LinearRegression

df = load_dataset("data/dataset.csv", required_columns=["target"])

split = create_three_way_split(df, target="target", test_size=0.2, val_size=0.1, random_state=42)
save_split(split, "data/splits/dataset")

config = {
    "missing":  {"strategies": {"feature_a": "median"}},
    "outliers": {"columns": ["feature_a"], "method": "iqr", "multiplier": 1.5},
    "scaling":  {"columns": ["feature_a", "feature_b"], "method": "standard"},
}
pipeline = PreprocessingPipeline(config)

X_train = pipeline.fit_transform(split.X_train)
X_val   = pipeline.transform(split.X_val)
X_test  = pipeline.transform(split.X_test)

registry = ModelRegistry(task="regression").register("linear_regression", LinearRegression())
registry.fit_all(X_train, split.y_train, X_val, split.y_val)

best_name, best_model = registry.get_best()
print(evaluate_regression(best_model, X_test, split.y_test, label="test"))

save_config(config, "models/pipeline_config.json")
pipeline.save("models/preprocessing_pipeline.pkl")
save_model(best_model, "models/best_model.joblib")
```

**The split always comes first.** Every `fit`-style call (`PreprocessingPipeline.fit_transform`,
`run_feature_engineering(..., is_train=True)`) runs only on `X_train`; `X_val` and `X_test` only
ever see `transform()`. This is what prevents data leakage between splits.

## Configuration

Preprocessing and feature-engineering behaviour is driven by a single config dictionary, which can
be saved and reloaded as JSON:

```python
from modules.pipeline import save_config, load_config

save_config(config, "models/pipeline_config.json")
config = load_config("models/pipeline_config.json")
```

Expected keys:

- `missing` — `{"strategies": {col: "mean" | "median" | "mode" | <constant>}, "indicator_columns": [...]}`
- `outliers` — `{"columns": [...], "method": "iqr", "multiplier": 1.5}`
- `scaling` — `{"columns": [...], "method": "standard" | "minmax"}`
- `feature_engineering` — `{"family_features": bool, "log_columns": [...], "polynomial": {...}, "onehot_columns": [...]}`

## Reloading for Inference

```python
from modules.pipeline import PreprocessingPipeline
from modules.modeling import load_model

pipeline = PreprocessingPipeline.load("models/preprocessing_pipeline.pkl")
model = load_model("models/best_model.joblib")

X_new_clean = pipeline.transform(X_new)
predictions = model.predict(X_new_clean)
```

## Notebooks

`notebooks/` is for exploration and communication, not for logic the workflow depends on:

- initial EDA before a pattern is promoted into `modules/eda.py`
- prototyping new features or models before they're added to `feature_engineering.py` or
  `ModelRegistry`
- interactive walkthroughs of `run_workflow.py` for demos and reviews

Anything the workflow needs to run correctly belongs in `modules/`, where it can be imported and
tested.

## Testing

```bash
pytest tests/
```

At minimum, `tests/` should include smoke tests that run the load → split → preprocess pipeline
end-to-end and assert on shapes and column consistency (see `test_pipeline_runs.py`).

## Author

**Shouke Wei, PhD**
Deepsim Intelligence Technology Inc., Canada
[https://deepsim.ca/](https://deepsim.ca/)

## License

MIT License.
