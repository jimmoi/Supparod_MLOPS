import numpy as np
import pickle
from flask import Flask, request, jsonify
import mlflow

# สร้าง Flask application
app = Flask(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

try:
    model_uri = "models:/Classifier-Prod/2"
    mlflow.pyfunc.get_model_dependencies(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully!")
    
except Exception as e: # แก้ไขการดักจับ Error ให้แสดงข้อความจริง
    print(f"An error occurred: {e}")
    model = None

# postman http://127.0.0.1:5001/predict : {"features":[1,2,3,4]}
# @app.route('/predict', methods=['POST'])
# def predict():
#     # รับข้อมูล JSON จาก request
#     data = request.get_json(force=True)
#     try:
#         # แปลงข้อมูล features ให้เป็น numpy array
#         features = np.array(data['features'])
        
#         # ตรวจสอบมิติของข้อมูล (ต้องเป็น 2D array)
#         if features.ndim == 1:
#             features = features.reshape(1, -1) # แปลง [f1, f2, f3, f4] ให้เป็น [[f1, f2, f3, f4]]


#         # ทำนายผล
#         prediction_index = model.predict(features)
        
#         # ดึงชื่อคลาสจากการทำนาย
#         predicted_class_name = iris_target_names[prediction_index[0]]
        
#         result = {
#             'input_features': data['features'],
#             'predicted_class': predicted_class_name
#         }
        
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
# if __name__ == '__main__':
    # # app.run(debug=True) # ใช้สำหรับตอนพัฒนา
    # app.run(host='0.0.0.0', port=5001) # ใช้สำหรับ production หรือให้เครื่องอื่นเรียกได้