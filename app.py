# streamlit_app.py
import streamlit as st
import pandas as pd
from streamlit_cropper import st_cropper
from PIL import Image
from io import BytesIO

# The following imports and functions are from the original api.py
import os
import cv2
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# === Dictionaries for Shade Mappings ===
SUGGESTIVE_SHADES = {
    'B1': 'A1', 'A1': 'A1', 'B2': 'A1/A2', 'D2': 'A2', 'A2': 'A2',
    'C1': 'A2', 'C2': 'A2/A3', 'D4': 'A3', 'A3': 'A3', 'D3': 'A3',
    'B3': 'A3/A3.5', 'A3.5': 'A3.5', 'B4': 'A3.5', 'C3': 'A4',
    'A4': 'A4', 'C4': 'A4',
}

THREE_D_SHADES = {
    'A1': '1M2', 'A2': '2M2', 'A3': '3M3', 'A3.5': '3R2.5', 'A4': '4L1.4',
    'B1': '1M1', 'B2': '2L2.5', 'B3': '3L2.5', 'B4': '3L2.5', 'C1': '3M1',
    'C2': '3L1.5', 'C3': '4L1.5', 'C4': '5M1', 'D2': '3M2',
    'D3': '3M1 or 4L1.5', 'D4': '3L1.5',
}

# === Feature Extraction Functions ===
def compute_color_histogram(image):
    """Computes a 3-channel color histogram for an image."""
    chans = cv2.split(image)
    hist_values = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_values.extend(hist)
    return hist_values

def compute_color_moments(image):
    """Computes the mean and standard deviation for each color channel."""
    chans = cv2.split(image)
    moments = []
    for chan in chans:
        mean = np.mean(chan)
        std = np.std(chan)
        moments.extend([mean, std])
    return moments

def preprocess_image(image):
    """Resizes and extracts features from an image."""
    if image is None:
        return None
    
    # Resize to 128x128, as used in original training
    image = cv2.resize(image, (128, 128))
    
    hist_values = compute_color_histogram(image)
    color_moments = compute_color_moments(image)
    return np.array(hist_values + color_moments)

# === Load Model Components ===
@st.cache_resource
def load_model_components():
    """Loads the pre-trained model, SVD transformer, and label encoder."""
    try:
        model_path = os.path.join("trained_models", "svm_model.pkl")
        svd_path = os.path.join("trained_models", "svd_transformer.pkl")
        encoder_path = os.path.join("trained_models", "label_encoder.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(svd_path, 'rb') as f:
            svd = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, svd, label_encoder
    except FileNotFoundError as e:
        st.error(f"Error: A model file was not found. Please ensure all .pkl files are in the 'trained_models' directory. Error: {e}")
        return None, None, None

# Load the model components once when the app starts using a cache
model, svd, encoder = load_model_components()


# === Streamlit UI and Prediction Logic ===
def plot_predictions(predictions):
    """
    Creates and displays a horizontal bar chart for the top predictions.
    """
    shades = [p['shade'] for p in predictions]
    probabilities = [p['probability'] for p in predictions]
    
    # Sort the data in descending order for the chart
    sorted_data = sorted(zip(shades, probabilities), key=lambda x: x[1], reverse=False)
    sorted_shades = [item[0] for item in sorted_data]
    sorted_probs = [item[1] for item in sorted_data]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sorted_shades, sorted_probs, color='skyblue')
    ax.set_xlabel('Probability (%)')
    ax.set_title('Top 3 Predictions')
    ax.set_xlim(0, 1)  # Set x-axis limit to 0-1 for probability scale

    # Add percentages on the bars
    for bar, prob in zip(bars, sorted_probs):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{prob:.2%}', 
                va='center', ha='left', fontsize=12)
    
    st.pyplot(fig)


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="Dental Shade Classifier", layout="centered")
    st.title("🦷 Dental Shade Classification")
    st.write("Upload a tooth image, crop it, and get shade predictions.")

    if model is None or svd is None or encoder is None:
        return
    
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        st.subheader("🖼️ Crop Area of Interest")
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color='#FF4B4B',
            aspect_ratio=(1, 1),
        )
        
        st.image(cropped_img, caption="Square Cropped Image", use_container_width=True)

        if st.button("🔍 Predict"):
            if cropped_img is None:
                st.error("Please crop an area of the image first.")
                return

            try:
                # Convert PIL image to OpenCV format
                img_array = np.array(cropped_img)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Preprocess and extract features
                with st.spinner('Extracting features and predicting...'):
                    features = preprocess_image(img_cv)
                
                if features is None:
                    st.error("Image preprocessing failed.")
                    return

                features_array = np.array([features])
                X_sparse = csr_matrix(features_array)
                X_reduced = svd.transform(X_sparse)

                # Predict the shade probabilities
                probs = model.predict_proba(X_reduced)
                
                # Get the indices of the top 3 predictions
                top3_idx = np.argsort(probs[0])[::-1][:3]
                top3_classes = encoder.inverse_transform(top3_idx)
                top3_probs = probs[0][top3_idx]
                
                # Structure the top 3 predictions for the JSON response
                top_predictions = []
                for shade, prob in zip(top3_classes, top3_probs):
                    top_predictions.append({"shade": shade, "probability": float(prob)})

                # Get the single best prediction for the table
                best_predicted_shade = top3_classes[0]

                # Get suggestive and 3D shades from the dictionaries
                suggestive_shade = SUGGESTIVE_SHADES.get(best_predicted_shade, 'N/A')
                three_d_shade = THREE_D_SHADES.get(best_predicted_shade, 'N/A')
                
                # === Display the Top 3 Predictions as a Horizontal Bar Chart ===
                st.subheader("📊 Top Predictions")
                plot_predictions(top_predictions)

                # === Display the Final Table ===
                st.subheader("📋 Shade Results")
                
                table_data = {
                    "Predicted Shade": [best_predicted_shade],
                    "Suggested Shade": [suggestive_shade],
                    "3D Shade": [three_d_shade]
                }
                df_table = pd.DataFrame(table_data)
                st.table(df_table)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                
if __name__ == "__main__":
    main()