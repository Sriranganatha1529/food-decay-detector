import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Food Decay Detector", page_icon="🍎")

# Load TFLite model using tf (already available in environment)
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🍎 Food Decay Detector")
st.write("Upload an image to check if food is Fresh or Rotten")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    img = img.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    confidence = prediction[0][0]

    if confidence > 0.5:
        st.error(f"Rotten ❌ ({confidence:.2f})")
    else:
        st.success(f"Fresh ✅ ({1-confidence:.2f})")
