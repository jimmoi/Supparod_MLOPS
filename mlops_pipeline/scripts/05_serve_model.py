import torch
from torchvision import transforms

import pickle
import time
import io
from flask import Flask, request, jsonify
from PIL import Image
import os

# --- 1. SETUP ---

SEED = 42
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on {str(device).upper()} ---")

# --- 2. LOAD ARTIFACTS ---
def load_artifacts():
    try:
        model_path = "ghost_netv1.pt"
        preprocessor_path = os.path.join("processed_data", "preprocess_artifact.pkl")

        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        
        print("Model and preprocessor loaded successfully.")
        return model, preprocessor
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None

model, preprocessor = load_artifacts()
inference_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
label_encoder = preprocessor['label_encoder']


# --- 3. CREATE PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and model is not None:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            input_tensor = inference_transform(image)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)

            with torch.no_grad():
                if str(device) == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_event.record()
                    
                    output = model(input_batch)
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_time = start_event.elapsed_time(end_event) / 1000.0
                else:
                    start_time = time.time()
                    output = model(input_batch)
                    end_time = time.time()
                    inference_time = end_time - start_time

            output = output.cpu()
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # --- START: TOP-5 PREDICTION LOGIC ---
            
            # 1. ใช้ torch.topk เพื่อหา 5 อันดับแรก
            top5_probs, top5_indices = torch.topk(probabilities, 5)

            # 2. แปลงผลลัพธ์เป็น list ของ dictionary
            predictions = []
            for i in range(len(top5_probs)):
                prob = top5_probs[i].item()
                class_idx = top5_indices[i].item()
                class_label = label_encoder.inverse_transform([class_idx])[0]
                
                predictions.append({
                    'class': class_label,
                    'confidence': f"{prob:.2%}"
                })
            
            # 3. สร้าง JSON result ใหม่
            result = {
                'predictions': predictions,
                'time_taken': f"{inference_time:.4f}"
            }
            # --- END: TOP-5 PREDICTION LOGIC ---

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

    return jsonify({'error': 'Model not loaded'}), 500

# --- 4. RUN THE APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)