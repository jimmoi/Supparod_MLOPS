import sys
import os # เพิ่ม import นี้สำหรับจัดการ file path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts # เพิ่ม import นี้


def train_evaluate_register(preprocessing_run_id, C=1.0):
    """
    Loads preprocessed data, trains a model, evaluates it, and
    registers the model in the MLflow Model Registry if it meets
    the performance threshold.
    """
    ACCURACY_THRESHOLD = 0.95
    mlflow.set_experiment("Iris - Model Training")


    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        print(f"Starting training run with C={C}...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)


        # 1. โหลดข้อมูลจาก Artifacts ของ Preprocessing Run
        try:
            # --- START: โค้ดส่วนที่แก้ไขสำหรับ Windows ---
            # 1.1 ใช้ MLflow ดาวน์โหลด Artifacts ลงมาที่ local path ชั่วคราว
            local_artifact_path = download_artifacts(
                run_id=preprocessing_run_id,
                artifact_path="processed_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")


            # 1.2 สร้างพาธไปยังไฟล์ CSV ที่ดาวน์โหลดมา
            train_path = os.path.join(local_artifact_path, "train.csv")
            test_path = os.path.join(local_artifact_path, "test.csv")
            
            # 1.3 อ่านไฟล์ CSV จาก local path ที่ถูกต้อง
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("Successfully loaded data from downloaded artifacts.")
            # --- END: โค้ดส่วนที่แก้ไข ---
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Please ensure the preprocessing_run_id is correct.")
            sys.exit(1)


        X_train = train_df.drop('Species', axis=1)
        y_train = train_df['Species']
        X_test = test_df.drop('Species', axis=1)
        y_test = test_df['Species']
        
        # 2. สร้าง Scikit-learn Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)


        # 3. ประเมินผลโมเดล
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")


        # 4. Log Parameters, Metrics, และ Model (Pipeline)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "iris_classifier_pipeline")


        # 5. ตรวจสอบและลงทะเบียนโมเดล
        if acc >= ACCURACY_THRESHOLD:
            print(f"Model accuracy ({acc:.4f}) meets the threshold. Registering model...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/iris_classifier_pipeline"
            registered_model = mlflow.register_model(model_uri, "iris-classifier-prod")
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
        else:
            print(f"Model accuracy ({acc:.4f}) is below the threshold. Not registering.")
        print("Training run finished.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_id = sys.argv[1]
        c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    elif len(sys.argv) == 2:
        if (sys.argv[1]=="local") and (os.path.exists("run_id.json")):
            with open("run_id.json", "r") as f:
                import json
                data = json.load(f)
                run_id = data.get("run_id")
                c_value = 0.666
                if not run_id:
                    print("run_id.json found but 'run_id' key is missing.")
                    sys.exit(1)
    else:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C_value]")
        sys.exit(1)
    
    train_evaluate_register(preprocessing_run_id=run_id, C=c_value)
