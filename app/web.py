import streamlit as st
from PIL import Image
import requests
import pandas as pd
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Classifier by Supparod",
    page_icon="🍍",
    layout="centered"
)

# --- Title and Description ---
st.title("🍍 Plant Classifier & Anomaly Detector")
st.write(
    "Upload a plant image. The system will first check if it's a valid plant type (not an anomaly) "
    "and then predict its species."
)

# --- API Endpoint ---
# Make sure your Flask API is running at this address.
API_URL = "http://127.0.0.1:5001/predict"

# --- Main Application Logic ---
uploaded_file = st.file_uploader(
    "Choose a plant image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image bytes for both displaying and sending
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Create a smaller version for display
    display_image = image.copy()
    display_image.thumbnail((400, 400))

    st.markdown("---")
    col1, col2 = st.columns([0.8, 1.2]) # Adjust column widths

    with col1:
        st.image(display_image, caption='Uploaded Image', use_container_width=True)

    with col2:
        if st.button('Analyze Image', type="primary"):
            with st.spinner('Sending image to the model API...'):
                try:
                    # Prepare the file for the POST request
                    files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                    
                    # Send the request
                    response = requests.post(API_URL, files=files, timeout=30) # Added a timeout
                    response.raise_for_status() # Raise an error for bad status codes (4xx or 5xx)
                    
                    # Get the JSON result
                    result = response.json()

                    # --- START OF MODIFIED SECTION ---
                    # Check the 'status' field returned by the API
                    status = result.get('status')

                    if status == 'normal':
                        st.success("✅ Analysis Complete: Image is Normal")
                        
                        predictions = result.get('predictions', [])
                        time_taken = result.get('time_taken', 'N/A')
                        anomaly_score = result.get('anomaly_score', 'N/A')

                        if predictions:
                            top_prediction = predictions[0]
                            st.write(f"**Top Prediction:** `{top_prediction['class']}`")
                            st.write(f"**Confidence:** `{top_prediction['confidence']}`")
                            st.write(f"**Inference Time:** `{time_taken} seconds`")
                            st.write(f"**Anomaly Score:** `{anomaly_score}` (Below threshold)")
                            
                            st.markdown("---")
                            st.subheader("All Top 5 Predictions")
                            
                            df_predictions = pd.DataFrame(predictions)
                            df_predictions.columns = ["Predicted Class", "Confidence"]
                            st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                        else:
                            st.warning("Status is 'normal', but no predictions were returned.")

                    elif status == 'anomaly_detected':
                        st.error("🚨 Analysis Complete: Anomaly Detected!")
                        
                        score = result.get('anomaly_score')
                        threshold = result.get('threshold')
                        message = result.get('message')
                        
                        st.metric(
                            label="Anomaly Score", 
                            value=f"{float(score):.2f}", 
                            delta=f"Threshold: {float(threshold):.2f}",
                            delta_color="inverse"
                        )
                        st.warning(message)
                        
                    else:
                        # Handle any other unexpected responses
                        st.error("Received an unknown response from the API.")
                        st.json(result)
                    
                    # --- END OF MODIFIED SECTION ---

                except requests.exceptions.RequestException as e:
                    st.error(f"API Connection Error: Could not connect to the model server. Please ensure the API is running. Details: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")