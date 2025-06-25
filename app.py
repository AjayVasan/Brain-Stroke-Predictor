from flask import Flask , render_template,request,send_from_directory
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import io

# Load model
model = load_model("Model/model.h5")

class_labels = ['Kanama', 'YarısmaVeriSeti_1_Oturum', 'YarısmaVeriSeti_2_Oturum', 'İnme Yok', 'İskemi']

img_siz = 225

st.set_page_config(page_title="Stroke MRI Dectector" , layout='centered')
st.title("🧠 Brain Stroke MRI Classifier")
st.write("Upload an MRI image to detect the type of stroke (or) no stroke")

upload = st.file_uploader("Upload an MRI imaage",type=['jpeg','png','jpg'])

if upload :
    try:
        img = Image.open(upload).convert('RGB')
        img_resized = img.resize((img_siz,img_siz))
        img_arr = img_to_array(img_resized) / 255.0
        img_arr = np.expand_dims(img_arr,axis=0)

        prediction = model.predict(img_arr)
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_label = class_labels[predicted_index]

        # Show output
        st.image(img, caption="Uploaded MRI Image", use_column_width=True)
        st.markdown(f"### 🧾 Prediction: **{predicted_label}**")
        st.markdown(f"### 🔍 Confidence: **{confidence * 100:.2f}%**")

    except Exception as e:
        st.error(f"Error in processing image: {str(e)}")