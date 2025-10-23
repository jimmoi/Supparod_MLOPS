import streamlit as st
from PIL import Image
import requests
import pandas as pd # Import pandas for better table display

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Classifier by Supparod",
    page_icon="🍍",
    layout="centered"
)

# --- Title and Description ---
st.title("🍍 Plant Classifier by Supparod")
st.write(
    "Upload a plant image, and the model will predict the type of plant. "
    "This app sends the image to a backend API for processing."
)

# --- API Endpoint ---
API_URL = "http://127.0.0.1:5001/predict"


# --- Main Application Logic ---
uploaded_file = st.file_uploader(
    "Choose a plant image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    display_image = image.copy()
    display_image.thumbnail((400, 400))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.image(display_image, caption='Uploaded Image', use_container_width=True)

    with col2:
        if st.button('Predict'):
            with st.spinner('Sending image to the model API...'):
                try:
                    image_bytes = uploaded_file.getvalue()
                    files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                    response = requests.post(API_URL, files=files)
                    response.raise_for_status()
                    result = response.json()

                    # --- START: DISPLAY TOP-5 RESULTS ---
                    st.success("Prediction Complete!")

                    # 1. ดึง list ของ predictions ออกมา
                    predictions = result.get('predictions', [])
                    time_taken = result.get('time_taken', 'N/A')

                    if predictions:
                        # 2. แสดงผลการทำนายอันดับ 1 ให้เด่นชัด
                        top_prediction = predictions[0]
                        st.write(f"**Top Prediction:** `{top_prediction['class']}`")
                        st.write(f"**Confidence:** `{top_prediction['confidence']}`")
                        st.write(f"**Inference Time:** `{time_taken} seconds`")
                        
                        st.markdown("---")
                        st.subheader("All Top 5 Predictions")
                        
                        # 3. สร้าง DataFrame เพื่อแสดงผลเป็นตารางสวยงาม
                        df_predictions = pd.DataFrame(predictions)
                        df_predictions.columns = ["Predicted Class", "Confidence"]
                        st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                        
                    else:
                        st.warning("No predictions were returned from the API.")
                    # --- END: DISPLAY TOP-5 RESULTS ---

                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling the API: {e}")
                except KeyError:
                    st.error("Received an unexpected response from the API. Please check the API's output format.")