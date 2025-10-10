import os
# import random
import sys
# import numpy as np
import pandas as pd
import pickle
import json
from tqdm import tqdm
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data import random_split, SubsetRandomSampler
from torchvision import transforms
# from torchvision import datasets, models 

# from torchvision.datasets import ImageFolder
# from torchvision.transforms import ToTensor
# from torchvision.utils import make_grid

# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from PIL import Image

import mlflow
# import mlflow.sklearn
from mlflow.artifacts import download_artifacts # เพิ่ม import นี้

# Data Preprocessing
def preprocess_data(data_validation_id):
    # Set the experiment name
    mlflow.set_experiment("Data Preprocessing")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_param("validation_run_id", data_validation_id)

        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)


        try:
            local_artifact_path = download_artifacts(
                run_id=data_validation_id,
                artifact_path="validation_data"
            )
            print(f"Artifacts downloaded to: {local_artifact_path}")


            # 1.2 สร้างพาธไปยังไฟล์ CSV ที่ดาวน์โหลดมา
            train_path = os.path.join(local_artifact_path, "train.csv")
            val_path = os.path.join(local_artifact_path, "val.csv")
            
            # 1.3 อ่านไฟล์ CSV จาก local path ที่ถูกต้อง
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            print("Successfully loaded data from downloaded artifacts.")
            # --- END: โค้ดส่วนที่แก้ไข ---
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Please ensure the preprocessing_run_id is correct.")
            sys.exit(1)
        
        le = LabelEncoder()
        
        merged_df = pd.concat([train_df, val_df], ignore_index=True)
        le.fit(merged_df['class'])
        del merged_df
        train_df['label'] = le.transform(train_df['class'])
        val_df['label'] = le.transform(val_df['class'])

        def create_path_label_list(df):
            path_label_list = []
            for _, row in df.iterrows():
                path = row['file_path']
                label = row['label']
                path_label_list.append((path, label))
            return path_label_list

        transform=transforms.Compose([
                transforms.RandomRotation(10),      # rotate +/- 10 degrees
                transforms.RandomHorizontalFlip(),  # reverse 50% of images
                transforms.Resize(224),             # resize shortest side to 224 pixels
                transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])


        ## เก็บ artifact transform ที่ใช้
        with open(os.path.join(processed_data_dir, "preprocess_artifact.pkl"), "wb") as f:
            data = {"label_encoder": le,
                    "transform": transform}
            pickle.dump(data, f)

        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data as artifacts in MLflow.")
        #################################
        
        
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"processing_run_id={run_id}", file=f)
        else:
            with open("run_id.json", "r") as f:
                data = json.load(f)
                data["processing_run_id"] = run_id
            with open("run_id.json", "w") as f:
                json.dump(data, f)


        class CustomDataset(Dataset):
            def __init__(self, path_label, transform=None):
                self.path_label = path_label
                self.transform = transform

            def __len__(self):
                return len(self.path_label)

            def __getitem__(self, idx):
                path, label = self.path_label[idx]
                img = Image.open(path).convert('RGB')

                if self.transform is not None:
                    img = self.transform(img)

                label = torch.tensor(label, dtype=torch.long)  #LongTensor
                return img, label
            

        train_dataset = CustomDataset(create_path_label_list(train_df), transform=transform)
        val_dataset = CustomDataset(create_path_label_list(val_df), transform=transform)
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print("-" * 50)
        print(f"Data preprocessing run finished. Please use the following Run ID for the predict step:")
        print(f"preprocessing Run ID: {run_id}")
        print("-" * 50)
        
        print("Data validation run finished.")
        return le, train_loader, val_loader

# Model Training

def train_evaluate_register(le, train_loader, val_loader):
    """
    Trains a model, evaluates it, and
    registers the model in the MLflow Model Registry if it meets
    the performance threshold.
    """
    
    ACCURACY_THRESHOLD = 0.90
    mlflow.set_experiment("Model Training")
    
    with mlflow.start_run(run_name=f"GhostNet_Finetune"):
        print(f"Starting training")
        mlflow.set_tag("ml.step", "model_training_evaluation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_ghostnet = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)

    num_ftrs = pretrained_ghostnet.classifier.in_features
    pretrained_ghostnet.classifier = nn.Linear(num_ftrs, le.classes_.size)
    pretrained_ghostnet = pretrained_ghostnet.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pretrained_ghostnet.parameters(), lr=0.001)
    
    def train_model(model, train_loader, val_loader, criterion, optimizer, device):
        num_epochs = 1
        best_val_acc = 0.0
        best_model_wts = None

        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            # class_running_corrects = [0] * len(le.classes_)

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # class_running_corrects += torch.sum(preds.unsqueeze(1) == torch.arange(len(le.classes_)).to(device), dim=0).cpu().numpy()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            # epoch_class_acc = class_running_corrects / len(train_loader.dataset)

            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_running_corrects = 0

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data)
                    
            val_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Acc: {val_epoch_acc:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc.item(), step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_epoch_acc.item(), step=epoch)

            # Deep copy the model if it has the best validation accuracy so far
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                best_model_wts = model.state_dict().copy()

        print(f"Best Validation Acc: {best_val_acc:.4f}")

        # Load best model weights
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            
        return best_val_acc, model


    acc, model = train_model(pretrained_ghostnet, train_loader, val_loader, criterion, optimizer, device)

    mlflow.pytorch.log_model(model, "model", registered_model_name="GhostNet_Classifier")
    
    if acc >= ACCURACY_THRESHOLD:
        print(f"Validation accuracy {acc:.4f} meets the threshold of {ACCURACY_THRESHOLD}. Registering model...")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/GhostNet_Classifier"
        registered_model = mlflow.register_model(model_uri, "GhostNet-Classifier-Prod")
        print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
    else:
        print(f"Model accuracy ({acc:.4f}) is below the threshold. Not registering.")
    print("Training run finished.")
    
    
    
    
def main(run_id):
    le, train_loader, val_loader = preprocess_data(run_id)
    train_evaluate_register(le, train_loader, val_loader)
    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        if (sys.argv[1]=="local") and (os.path.exists("run_id.json")):
            with open("run_id.json", "r") as f:
                import json
                data = json.load(f)
                run_id = data.get("validation_run_id")
                if not run_id:
                    print("run_id.json found but 'run_id' key is missing.")
                    sys.exit(1)
        else:
            run_id = sys.argv[1]
    else:
        print("Usage: python scripts/03_train_evaluate_register.py <validation_run_id>")
        sys.exit(1)
        
    main(run_id)