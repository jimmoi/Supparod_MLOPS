import os
from copy import deepcopy
import sys
# import numpy as np
import pandas as pd
import pickle
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.jit as jit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models


from sklearn.preprocessing import LabelEncoder
from PIL import Image

import mlflow
from mlflow.artifacts import download_artifacts


from torchvision.models import ResNet50_Weights

# Data Preprocessing
def preprocess_data(data_validation_id, batch_size):
    # Set the experiment name
    mlflow.set_experiment(f"Data Preprocessing batch_size_{batch_size}")
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
        
        inference_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


        ## เก็บ artifact transform ที่ใช้
        with open(os.path.join(processed_data_dir, "preprocess_artifact.pkl"), "wb") as f:
            data = {
                "label_encoder": le,
                "transform": inference_transform
                }
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print("-" * 50)
        print(f"Data preprocessing run finished. Please use the following Run ID for the predict step:")
        print(f"preprocessing Run ID: {run_id}")
        print("-" * 50)
        
        print("Data validation run finished.")
        return le, train_loader, val_loader

# Model Training

def train_evaluate_register(le, train_loader, val_loader, lr, num_epochs):
    """
    Trains a model, evaluates it, and
    registers the model in the MLflow Model Registry if it meets
    the performance threshold.
    """
    
    param = {
        "experiment_name": f"Model Training lr_{lr}_epochs_{num_epochs}",
        "run_name": "GhostNetv3_Finetune",
        "accuracy_threshold": 0.85,
        "pretrained_model_url": "ghostnetv1_100",
        "model_registered_name": "GhostNetv1_Classifier",
        "model_name": "GhostNetv1-Classifier-Prod",
        "learning_rate": lr,
        "batch_size": train_loader.batch_size,
        "num_epochs": num_epochs,
        "model_architecture": "GhostNetv1+FC",
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
    }
    
    ACCURACY_THRESHOLD = param["accuracy_threshold"]
    mlflow.set_experiment(param["experiment_name"])

    with mlflow.start_run(run_name=param["run_name"]):
        print(f"Starting training")
        mlflow.set_tag("ml.step", "model_training_evaluation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 1. Load the pretrained ResNet model
    # pretrained_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # pretrained_model = timm.create_model(param["pretrained_model_url"], pretrained=True)
    pretrained_model =  torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)

    # # 2. FREEZE THE ENTIRE BACKBONE
    # for model_param in pretrained_model.parameters():
    #     model_param.requires_grad = False

    # 3. REPLACE THE CLASSIFIER HEAD
    num_ftrs = pretrained_model.classifier.in_features
    print(f"Number of features in the classifier: {num_ftrs}")

    pretrained_model.classifier = nn.Linear(num_ftrs, len(le.classes_))
    # pretrained_model.classifier = nn.Sequential(
    #     # nn.Dropout(p=0.25), 
    #     # nn.Linear(num_ftrs, 50),
    #     # nn.Linear(50, len(le.classes_)),
    #     nn.Linear(num_ftrs, len(le.classes_))
    # )

    # 4. Move the model to the device
    pretrained_model = pretrained_model.to(device)

    # 5. Initialize Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, pretrained_model.parameters()),
        lr=param["learning_rate"]
    )
    num_epochs = param["num_epochs"]
    # Log hyperparameters
    mlflow.log_param("learning_rate", param["learning_rate"])
    mlflow.log_param("batch_size", param["batch_size"])
    mlflow.log_param("num_epochs", param["num_epochs"])
    mlflow.log_param("model_architecture", param["model_architecture"])
    mlflow.log_param("optimizer", param["optimizer"])
    mlflow.log_param("loss_function", param["loss_function"])
    
    def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=1):
        best_val_acc = 0.0
        best_model_wts = None

        for epoch in range(epochs):
            running_loss = 0.0
            running_corrects = 0
            # class_running_corrects = [0] * len(le.classes_)
            
            # Training phase
            model.train()
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
                best_model_wts = deepcopy(model.state_dict())

        print(f"Best Validation Acc: {best_val_acc:.4f}")

        # Load best model weights
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            
        return best_val_acc, model


    acc, model = train_model(pretrained_model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    
    
    example_input, _ = next(iter(train_loader))
    traced_script_module = jit.trace(model, example_input.to(device))
    traced_script_module.save("test.pt")
    
    model_info = mlflow.pytorch.log_model(model, "Classifier")
    model_uri = model_info.model_uri
    
    if acc >= ACCURACY_THRESHOLD:
        print(f"Validation accuracy {acc:.4f} meets the threshold of {ACCURACY_THRESHOLD}. Registering model...")
        registered_model = mlflow.register_model(model_uri, "Classifier-Prod")
        print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
    else:
        print(f"Model accuracy ({acc:.4f}) is below the threshold. Not registering.")
    print("Training run finished.")
    
    


def main(run_id, batch_size, lr, num_epochs):
    print(f"Using run_id: {run_id}")
    print(f"Using batch_size: {batch_size}, learning_rate: {lr}, num_epochs: {num_epochs}")
    le, train_loader, val_loader = preprocess_data(run_id, batch_size)
    train_evaluate_register(le, train_loader, val_loader, lr, num_epochs)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/02_preprocess_and_training.py <validation_run_id> [batch_size] [lr] [num_epochs]")
        sys.exit(1)
        
    run_id = sys.argv[1] if len(sys.argv) > 1 else "local"
    if run_id == "local":
        with open("run_id.json", "r") as f:
            data = json.load(f)
            run_id = data.get("validation_run_id", -1)
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    num_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    
    main(run_id, batch_size, lr, num_epochs)
