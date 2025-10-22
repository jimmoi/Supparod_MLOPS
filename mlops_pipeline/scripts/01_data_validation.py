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
    mlflow.set_experiment("Data Validation")


    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data validation run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_validation")
        
        validation_data_dir = "validation_data"
        os.makedirs(validation_data_dir, exist_ok=True)


        def data_validation(data):
            all_df = []
            for section, path in data.items():
                if section == "root_dir":
                    continue
                df = pd.read_csv(path)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True) # <-- ADD THIS LINE
                df['sections'] = section
                all_df.append(df)
                
            merged_df = pd.concat(all_df, ignore_index=True)
            merged_df["category"] = merged_df["image:FILE"].apply(lambda x: os.path.split(x)[0].split("/")[-1]) # from directory structure
            merged_df.rename(columns={"image:FILE": "file_path", "category": "class"}, inplace=True)
            class_names = merged_df["class"].unique()
            merged_df["file_path"] = merged_df["file_path"].apply(lambda x: os.path.join(data["root_dir"], os.path.join(*x.split("/"))))
            
            number_per_class = {}
            for section in data.keys():
                if section == "root_dir":
                    continue
                section_df = merged_df[merged_df["sections"]==section].copy() 
                number_per_class[section] = len(section_df["class"])
                section_df.drop(columns=["sections"], inplace=True)
                section_df.reset_index(drop=True, inplace=True)
                section_df.to_csv(os.path.join(validation_data_dir, f"{section}.csv"), index=False)
            return {"class_names": class_names,
                    "Number of class": len(class_names),
                    "Number of data": len(merged_df),
                    "number_per_class": number_per_class
                }

        if os.path.exists("Dataset"):
            data_path = {
                "root_dir": r"Dataset",
                "train": "Dataset/train.csv",
                "val": "Dataset/val.csv",
                "test": "Dataset/test.csv"
            }
        else:
            data_path = {
                "root_dir": r"Dataset_github",
                "train": "Dataset_github/train.csv",
                "val": "Dataset_github/val.csv",
                "test": "Dataset_github/test.csv"
            }

        meta_data = data_validation(data_path)
        class_names = meta_data["class_names"]
        number_of_class = meta_data["Number of class"]
        number_of_data = meta_data["Number of data"]
        number_per_class = meta_data["number_per_class"]


        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", number_of_data)
        mlflow.log_metric("num_cols", number_of_class)
        mlflow.log_param("num_classes", number_of_class)
        mlflow.log_param("class_names", json.dumps(class_names.tolist()))
        mlflow.log_param("num_train", number_per_class.get("train", 0))
        mlflow.log_param("num_val", number_per_class.get("val", 0))
        mlflow.log_param("num_test", number_per_class.get("test", 0))


        validation_status = "passed" if number_of_data > 0 and number_of_class > 0 else "failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        mlflow.log_artifacts(validation_data_dir, artifact_path="validation_data")
        print("Logged validation data as artifacts in MLflow.")

        print("-" * 50)
        print(f"Data validation run finished. Please use the following Run ID for the next step:")
        print(f"Validation Run ID: {run_id}")
        print("-" * 50)
        
        print("Data validation run finished.")
        
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"validation_run_id={run_id}", file=f)

        else:
            with open("run_id.json", "w") as f:
                json.dump({"validation_run_id": run_id}, f)


if __name__ == "__main__":
    validate_data()
