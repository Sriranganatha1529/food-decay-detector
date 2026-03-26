import streamlit as st
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(page_title="Food Decay Detector", page_icon="🍎")

# Load model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🍎 Food Decay Detector")
st.write("Upload an image to check if food is Fresh or Rotten")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], img_array.astype('float32'))
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    confidence = prediction[0][0]


    if confidence > 0.5:
        st.error(f"Rotten ❌ ({confidence:.2f})")
    else:
        st.success(f"Fresh ✅ ({1-confidence:.2f})")
