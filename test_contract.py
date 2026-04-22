"""Simple manual test script for the DataContract class."""

import pandas as pd

from utils.data_contract import DataContract


def main() -> None:
    """Build sample data, initialize contract, and print outputs."""
    # Small sample dataset:
    # - age: numeric
    # - gender: categorical
    # - target: binary
    df = pd.DataFrame(
        {
            "age": [22, 35, 41, 29, 33, 48],
            "gender": ["F", "M", "F", "M", "F", "M"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    contract = DataContract(
        data=df,
        target_col="target",
        bias_columns=["gender"],
    )

    data = contract.get_data()

    print("=== DataContract: get_data() Output ===")
    print(f"Returned object type : {type(data)}")
    print(f"Dictionary keys      : {list(data.keys())}")
    print()

    print("X (first 5 rows):")
    print(data["X"].head())
    print()

    print("Y (first 5 rows):")
    print(data["Y"].head())
    print()

    print("Bias Columns:")
    print(f"  B_user      : {data['B_user']}")
    print(f"  B_suggested : {data['B_suggested']}")
    print(f"  B_hidden    : {data['B_hidden']}")
    print()

    print("Metadata:")
    for key, value in data["metadata"].items():
        print(f"  - {key}: {value}")
    print("=======================================\n")

    print("=== DataContract: Error Handling Test ===")
    try:
        DataContract(
            data=df,
            target_col="wrong_target",
            bias_columns=["gender"],
        )
    except ValueError as error:
        print("Caught ValueError (as expected):")
        print(f"  {error}")
    print("=========================================\n")

    print("=== DataContract: Missing Target Handling Test ===")
    df_with_missing_target = pd.DataFrame(
        {
            "age": [22, 35, 41, 29, 33, 48],
            "gender": ["F", "M", "F", "M", "F", "M"],
            "target": [0, 1, None, 1, 0, None],
        }
    )
    print(f"Original rows (with missing target): {len(df_with_missing_target)}")

    contract_missing = DataContract(
        data=df_with_missing_target,
        target_col="target",
        bias_columns=["gender"],
    )
    missing_data = contract_missing.get_data()

    print(f"Rows after cleanup              : {len(missing_data['Y'])}")
    print("X (after cleanup, first 5 rows):")
    print(missing_data["X"].head())
    print()
    print("Y (after cleanup, first 5 rows):")
    print(missing_data["Y"].head())
    print("===============================================\n")


if __name__ == "__main__":
    main()
