import pandas as pd
import mlflow
import json
import os


def validate_data():
    """
    Loads the iris dataset, performs basic validation checks,
    and logs the results to MLflow.
    """
    # Set the experiment name for this step
    mlflow.set_experiment("Iris - Data Validation")


    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")


        # 1. Load data as a Pandas DataFrame
        df = pd.read_csv(r"Iris.csv")
        print("Data loaded successfully.")


        # 2. Perform simple validation checks
        num_rows, num_cols = df.shape
        num_classes = df['Species'].nunique()
        missing_values = df.isnull().sum().sum()


        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")


        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("num_classes", num_classes)


        # Check if the data passes our defined criteria
        validation_status = "Success"
        if missing_values > 0 or num_classes < 3:
            validation_status = "Failed"


        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")
        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()
