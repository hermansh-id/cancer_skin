import streamlit as st
from PIL import Image
import numpy as np

# System append path abs + "../"

from counter_box.classifier import SkinCancerClassifier
from counter_box.preprocess import Preprocess
# Load the model
classifier = SkinCancerClassifier()
ps = Preprocess()
classifier.load_model('script/skin_cancer_model.pkl')

# Define function to classify an image
def classify_image(image):
    # Preprocess the image
    img_array = np.array(image)
    preprocessed_img = ps.process(img_array)
    
    # Calculate Hausdorff dimension
    slope, intercept, _ = classifier.cb.hausdorff(preprocessed_img)
    if np.isnan(slope):
        slope = np.zeros(1)
    hausdorff_dimension = slope
    
    # Flatten image and combine with Hausdorff dimension
    img_flat = img_array.reshape(1, -1)
    img_combined = np.hstack((img_flat, hausdorff_dimension.reshape(1, -1)))
    
    # Predict using the model
    prediction = classifier.model.predict(img_combined)
    return prediction[0]

# Streamlit app
st.title('Skin Cancer Classifier')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        prediction = classify_image(image)
        if prediction == 0:
            st.write('Prediction: Benign')
        else:
            st.write('Prediction: Malignant')
