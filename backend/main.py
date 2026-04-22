"""FastAPI backend for processing uploaded CSV datasets."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from utils.data_contract import DataContract

app = FastAPI(title="Unbiased AI Backend")


@app.post("/process-data")
async def process_data(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Process an uploaded CSV file and return a preview of contract outputs.

    Assumptions for now:
    - Target column: last column in CSV
    - Bias columns: empty list
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV file.")

    try:
        raw_content = await file.read()
        csv_text = raw_content.decode("utf-8")
        df = pd.read_csv(StringIO(csv_text))
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.") from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {error}") from error

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
    if len(df.columns) < 1:
        raise HTTPException(status_code=400, detail="CSV must contain at least one column.")

    target_col = df.columns[-1]
    bias_columns: list[str] = []

    try:
        contract = DataContract(data=df, target_col=target_col, bias_columns=bias_columns)
        data = contract.get_data()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    response = {
        "X": data["X"].head(5).to_dict(orient="records"),
        "Y": data["Y"].head(5).tolist(),
        "B_user": list(data["B_user"]),
        "B_suggested": list(data["B_suggested"]),
        "B_hidden": list(data["B_hidden"]),
        "metadata": dict(data["metadata"]),
    }
    return response
