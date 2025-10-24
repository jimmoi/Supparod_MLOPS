import torch
import torch.nn as nn
from torchvision import transforms, models
import pickle
import time
import io
from flask import Flask, request, jsonify
from PIL import Image
import os
import sys

# --- 1. CONFIGURATION ---
# General
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anomaly Detection Config
LATENT_DIM = 512
FINAL_THRESHOLD = 45.7011 # ค่า Threshold สำหรับตัดสิน Anomaly
AE_MODEL_PATH = 'autoencoder_anomaly_detector_ResNet50.pth'
FEATURE_DIM = None # จะถูกกำหนดค่าอัตโนมัติด้านล่าง

# Classification Config
CLASSIFIER_MODEL_PATH = "ghost_netv1.pt"
PREPROCESSOR_PATH = os.path.join("processed_data", "preprocess_artifact.pkl")

# --- 2. AUTOENCODER CLASS DEFINITION ---
# ต้องมี Class นี้อยู่เพื่อให้ torch.load ทำงานได้ถูกต้อง
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2), torch.nn.ReLU(True),
            torch.nn.Linear(input_dim // 2, latent_dim), torch.nn.ReLU(True)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, input_dim // 2), torch.nn.ReLU(True),
            torch.nn.Linear(input_dim // 2, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 3. HELPER FUNCTION FOR ANOMALY DETECTION ---
def load_resnet50_feature_extractor(device):
    """Loads the pre-trained ResNet-50 model and strips the classifier layer."""
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity() # แทนที่ชั้นสุดท้ายเพื่อเอาเฉพาะ feature
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ ERROR: ResNet-50 Feature Extractor setup failed. Error: {e}")
        return None

# --- 4. FLASK APP SETUP & MODEL LOADING ---
app = Flask(__name__)

# ตั้งค่า Seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้ง
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"--- Running on {str(DEVICE).upper()} ---")

# -- Load all models and artifacts at startup --
try:
    print("\n--- Loading Models & Artifacts ---")
    
    # 1. Load Anomaly Detection Models
    print("1/3) Loading Anomaly Detection components...")
    feature_extractor = load_resnet50_feature_extractor(DEVICE)
    if feature_extractor is None:
        sys.exit(1)

    #   Determine FEATURE_DIM dynamically
    dummy_input = torch.zeros(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        vector_output = feature_extractor(dummy_input)
    FEATURE_DIM = vector_output.view(vector_output.size(0), -1).shape[1]
    print(f"   - Feature dimension determined: {FEATURE_DIM}")
    
    #   Load Autoencoder
    ae_model = Autoencoder(input_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    ae_model.eval()
    print("   ✅ Anomaly models loaded successfully.")

    # 2. Load Classification Models
    print("2/3) Loading Classification components...")
    classifier_model = torch.jit.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE)
    classifier_model.eval()
    print("   ✅ Classifier model loaded successfully.")

    # 3. Load Preprocessor
    print("3/3) Loading preprocessor artifact...")
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    label_encoder = preprocessor['label_encoder']
    print("   ✅ Preprocessor loaded successfully.")
    
    print("\n--- All systems ready. Server is running. ---")

except Exception as e:
    print(f"❌ FATAL: Could not load all required models. Server cannot start. Error: {e}")
    feature_extractor = None
    ae_model = None
    classifier_model = None
    label_encoder = None

# Universal transform for both models
inference_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and ae_model is not None and classifier_model is not None:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Pre-process the image once
            input_tensor = inference_transform(image)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            # --- STEP 1: ANOMALY DETECTION ---
            with torch.no_grad():
                feature_vector = feature_extractor(input_batch)
                feature_vector = feature_vector.view(feature_vector.size(0), -1)
                reconstructed_vector = ae_model(feature_vector)
                anomaly_score = torch.linalg.norm(feature_vector - reconstructed_vector, dim=1).pow(2).item()

            # --- STEP 2: CONDITIONAL CLASSIFICATION ---
            if anomaly_score > FINAL_THRESHOLD:
                # If ANOMALY, return anomaly result and stop
                result = {
                    'status': 'anomaly_detected',
                    'anomaly_score': f"{anomaly_score:.4f}",
                    'threshold': f"{FINAL_THRESHOLD:.4f}",
                    'message': 'Image is considered an anomaly. Classification was not performed.'
                }
                return jsonify(result)
            
            else:
                # If NORMAL, proceed with classification
                with torch.no_grad():
                    start_time = time.time()
                    output = classifier_model(input_batch)
                    end_time = time.time()
                    inference_time = end_time - start_time

                output = output.cpu()
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top5_probs, top5_indices = torch.topk(probabilities, 5)

                predictions = []
                for i in range(len(top5_probs)):
                    prob = top5_probs[i].item()
                    class_idx = top5_indices[i].item()
                    class_label = label_encoder.inverse_transform([class_idx])[0]
                    predictions.append({
                        'class': class_label,
                        'confidence': f"{prob:.2%}"
                    })
                
                result = {
                    'status': 'normal',
                    'anomaly_score': f"{anomaly_score:.4f}",
                    'predictions': predictions,
                    'time_taken': f"{inference_time:.4f}"
                }
                return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

    return jsonify({'error': 'A required model was not loaded. Check server logs.'}), 500

# --- 6. RUN THE APP ---
if __name__ == '__main__':
    # Use threaded=False if you are using GPU models in Flask in a dev environment
    # to avoid potential CUDA initialization errors in threads.
    app.run(host='0.0.0.0', port=5001, threaded=False)