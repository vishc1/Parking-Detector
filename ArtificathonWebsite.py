import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = 'BestCarParkingModel2.pt'

ICONS = {
    'Disabled Parking': '♿',
    'No Parking Crosses': '❌',
    'No Parking Sign': '🚫',
    'cones': '🚧',
    'curbsides': '🛑',
    'normal parking': '🅿️'
}

@st.cache_resource
def load_model():
    """Load the YOLO model once and cache it"""
    try:
        print("Loading model..."+MODEL_PATH)
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_parking_spot(model, image):
    """Run prediction on the uploaded image"""
    try:
        results = model(image, verbose=False, conf=0.1)
        result = results[0]

        # Check if it's a classification model (has probs)
        if hasattr(result, 'probs') and result.probs is not None:
            probs_data = result.probs.data.cpu().numpy()
            class_names = list(model.names.values())

            probabilities_dict = {}
            for i, class_name in enumerate(class_names):
                if i < len(probs_data):
                    probabilities_dict[class_name] = float(probs_data[i])

            predicted_idx = int(result.probs.top1)
            predicted_class = class_names[predicted_idx]
            confidence = float(probs_data[predicted_idx])

            return predicted_class, confidence, probabilities_dict
        else:
            # Detection model - get bounding boxes
            boxes = result.boxes
            if len(boxes) > 0:
                best_box = boxes[0]
                class_id = int(best_box.cls[0])
                confidence = float(best_box.conf[0])
                predicted_class = model.names[class_id]

                probabilities_dict = {class_name: 0.0 for class_name in model.names.values()}
                probabilities_dict[predicted_class] = confidence

                return predicted_class, confidence, probabilities_dict
            else:
                probabilities_dict = {class_name: 0.0 for class_name in model.names.values()}
                return "No Detection", 0.0, probabilities_dict

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Parking Spot Classifier",
        page_icon="🅿️",
        layout="wide"
    )

    # Custom CSS styling
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .upload-box {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .result-box {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin-top: 1rem;
        }
        h1 {
            color: white !important;
            text-align: center;
            font-size: 3rem !important;
            margin-bottom: 0.5rem !important;
        }
        .subtitle {
            color: rgba(255,255,255,0.9);
            text-align: center;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and subtitle
    st.title("🅿️ Parking Spot Classifier")
    st.markdown('<p class="subtitle">Upload an image to detect parking spot type</p>', unsafe_allow_html=True)

    # Load model
    model = load_model()

    if model is None:
        st.error("❌ Failed to load the model. Please ensure model.pt is in the directory.")
        return

    st.success(f"✅ Model loaded successfully with {len(model.names)} classes")

    # File uploader
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a parking spot image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a parking spot (JPG, JPEG, or PNG)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Process uploaded image
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        # Display uploaded image
        with col1:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Display results
        with col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("🔍 Analysis Results")

            with st.spinner("Analyzing parking spot..."):
                predicted_class, confidence, probabilities = predict_parking_spot(model, image)

            if predicted_class:
                icon = ICONS.get(predicted_class, '📍')

                st.markdown(f"### {icon} {predicted_class}")
                st.metric("Confidence", f"{confidence * 100:.1f}%")

                if predicted_class == "No Detection":
                    st.warning("⚠️ No parking spot detected. Try a different image/angle")

                st.markdown("---")
                st.subheader("📊 All of the Predictions")

                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

                for class_name, prob in sorted_probs:
                    icon = ICONS.get(class_name, '📍')
                    st.write(f"{icon} **{class_name}**")
                    st.progress(prob)
                    st.caption(f"{prob * 100:.1f}%")

            st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a YOLO model to classify parking spots into different categories.

        **Classes:**
        """)

        for class_name in model.names.values():
            icon = ICONS.get(class_name, '📍')
            st.write(f"{icon} {class_name}")

        st.markdown("---")
        st.write("**How to use:**")
        st.write("1. Upload an image of a parking spot")
        st.write("2. Wait for the model to analyze it")
        st.write("3. View the predicted parking type and confidence score(s)")

        st.markdown("---")
        st.info("💡 For best results, try using clear images that show parking markings or signs.")

if __name__ == "__main__":
    main()
