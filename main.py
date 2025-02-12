import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import os

def verify_faces(img1_path, img2_path):
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name='Facenet')
        return result["verified"], result["distance"], result["threshold"]
    except Exception as e:
        return None, None, str(e)

st.title("🔍 Face Verification App")
st.write("Upload two images to check if they contain the same person.")

col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload Image 1 (Person to find)", type=["jpg", "jpeg", "png"])
    if img1:
        img1_pil = Image.open(img1)
        st.image(img1_pil, caption="Person to Find", use_column_width=True)

with col2:
    img2 = st.file_uploader("Upload Image 2 (Group photo)", type=["jpg", "jpeg", "png"])
    if img2:
        img2_pil = Image.open(img2)
        st.image(img2_pil, caption="Search in this Image", use_column_width=True)

if img1 and img2:
    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp1:
            img1_pil.save(temp1.name)
            img1_path = temp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp2:
            img2_pil.save(temp2.name)
            img2_path = temp2.name

        match, distance, threshold = verify_faces(img1_path, img2_path)

        os.remove(img1_path)
        os.remove(img2_path)

    if match is None:
        st.error(f"Error: {threshold}")
    elif match:
        st.success(f"✅ Person Found! (Distance: {distance:.4f}, Threshold: {threshold:.4f})")
    else:
        st.error(f"❌ Person Not Found (Distance: {distance:.4f}, Threshold: {threshold:.4f})")
