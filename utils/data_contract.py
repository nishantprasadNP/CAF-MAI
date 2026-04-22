"""Data contract object for dataset structure and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class DataContract:
    """
    Encapsulates validated dataset parts and metadata for ML workflows.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    target_col : str
        Name of the target column.
    bias_columns : list[str] | None, optional
        User-provided bias-sensitive columns.
    """

    data: pd.DataFrame
    target_col: str
    bias_columns: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate input and build all derived data structures."""
        self._validate_target_column()
        self._drop_missing_target_rows()
        self._build_xy()
        self._build_bias_columns()
        self._build_metadata()
        self.summary()

    def _validate_target_column(self) -> None:
        """Ensure the target column exists in the input DataFrame."""
        if self.target_col not in self.data.columns:
            available = ", ".join(map(str, self.data.columns.tolist()))
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame. "
                f"Available columns: [{available}]"
            )

    def _build_xy(self) -> None:
        """Split dataset into feature matrix X and target vector Y."""
        self.Y: pd.Series = self.data[self.target_col].copy()
        self.X: pd.DataFrame = self.data.drop(columns=[self.target_col]).copy()

    def _drop_missing_target_rows(self) -> None:
        """
        Remove rows where target is missing and print a warning.

        This keeps training labels aligned and avoids invalid samples.
        """
        initial_rows = len(self.data)
        cleaned = self.data.dropna(subset=[self.target_col]).copy()
        removed_rows = initial_rows - len(cleaned)

        if removed_rows > 0:
            print("Warning: Rows with missing target values were removed")

        self.data = cleaned

    def _build_bias_columns(self) -> None:
        """Store user bias columns and initialize other bias lists."""
        self.B_user: list[str] = list(self.bias_columns or [])
        self.B_suggested: list[str] = []
        self.B_hidden: list[str] = []

    def _build_metadata(self) -> None:
        """Create metadata such as feature column types and target type."""
        self.column_types: dict[str, str] = self._infer_column_types(self.X)
        self.target_type: str = self._infer_target_type(self.Y)
        self.metadata: dict[str, Any] = {
            "column_types": self.column_types,
            "target_type": self.target_type,
        }

    @staticmethod
    def _infer_column_types(features: pd.DataFrame) -> dict[str, str]:
        """
        Detect each feature column type as numeric or categorical.

        Non-numeric columns are treated as categorical by default.
        """
        column_types: dict[str, str] = {}
        for col in features.columns:
            column_types[col] = "numeric" if is_numeric_dtype(features[col]) else "categorical"
        return column_types

    @staticmethod
    def _infer_target_type(target: pd.Series) -> str:
        """
        Infer target type as binary, multiclass, or regression.

        Rules:
        - Non-numeric target -> classification
        - Numeric target with exactly 2 unique values -> binary
        - Numeric target with small integer-like cardinality -> multiclass
        - Otherwise -> regression
        """
        y_non_null = target.dropna()
        unique_count = y_non_null.nunique()

        # Binary classification is always a 2-class target.
        if unique_count == 2:
            return "binary"

        # Non-numeric targets are considered categorical classes.
        if not is_numeric_dtype(y_non_null):
            return "multiclass"

        # Numeric but integer-like with limited distinct values -> multiclass.
        is_integer_like = pd.api.types.is_integer_dtype(y_non_null) or (
            (y_non_null % 1 == 0).all() if len(y_non_null) > 0 else False
        )
        if is_integer_like and 3 <= unique_count <= 20:
            return "multiclass"

        return "regression"

    def get_data(self) -> dict[str, Any]:
        """
        Return all contract outputs in a single dictionary.

        Returns
        -------
        dict[str, Any]
            {
                "X": pd.DataFrame,
                "Y": pd.Series,
                "B_user": list[str],
                "B_suggested": list[str],
                "B_hidden": list[str],
                "metadata": dict[str, Any]
            }
        """
        return {
            "X": self.X,
            "Y": self.Y,
            "B_user": self.B_user,
            "B_suggested": self.B_suggested,
            "B_hidden": self.B_hidden,
            "metadata": {
                "column_types": self.column_types,
                "target_type": self.target_type,
            },
        }

    def summary(self) -> None:
        """
        Print a compact, user-friendly report of the contract.

        Report includes:
        - Number of rows
        - Number of feature columns
        - Inferred target type
        - User-provided bias columns
        """
        rows = len(self.data)
        feature_count = self.X.shape[1]
        target_type = self.target_type
        bias_cols = self.B_user if self.B_user else []

        print("\n=== Data Contract Summary ===")
        print(f"Rows           : {rows}")
        print(f"Feature columns: {feature_count}")
        print(f"Target type    : {target_type}")
        print(f"Bias columns   : {bias_cols}")
        print("=============================\n")
