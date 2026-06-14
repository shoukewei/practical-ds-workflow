# test_pipeline_runs.py
import pandas as pd

from modules.data_io import load_dataset, save_dataset
from modules.splitting import create_split
from modules.pipeline import PreprocessingPipeline


CONFIG = {
    "missing":  {"strategies": {"TV": "median", "radio": "mean", "newspaper": 0.0},
                  "indicator_columns": []},
    "outliers": {"columns": ["TV", "newspaper"], "method": "iqr", "multiplier": 1.5},
    "scaling":  {"columns": ["TV", "radio", "newspaper"], "method": "standard"},
}


def test_pipeline_runs(tmp_path):
    """A small, clean DataFrame representative of the Advertising dataset."""
    sample = pd.DataFrame({
        "TV":        [230.1, 44.5, 17.2, 151.5, 180.8,
                       8.7, 57.5, 120.2, 8.6, 199.8],
        "radio":     [37.8, 39.3, 45.9, 41.3, 10.8,
                      48.9, 32.8, 19.6, 2.1, 2.6],
        "newspaper": [69.2, 45.1, 69.3, 58.5, 58.4,
                      75.1, 23.5, 11.6, 1.0, 21.2],
        "sales":     [22.1, 10.4, 12.0, 16.5, 17.9,
                      7.2, 11.8, 13.2, 4.8, 10.6],
    })

    sample_path = tmp_path / "sample.csv"
    save_dataset(sample, str(sample_path))

    df = load_dataset(str(sample_path), required_columns=["sales"])
    split = create_split(df, target="sales", test_size=0.2, random_state=42)

    pipeline = PreprocessingPipeline(CONFIG)
    X_train_clean = pipeline.fit_transform(split.X_train)
    X_test_clean = pipeline.transform(split.X_test)

    assert X_train_clean is not None
    assert X_train_clean.shape[0] == split.X_train.shape[0]
    assert set(X_test_clean.columns) == set(X_train_clean.columns)
