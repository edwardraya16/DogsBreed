# pip install tensorflow

import streamlit as st
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform

st.title('Shoe Prediction')
st.code('Using Transfer Learning - EfficientNetB0')
st.subheader('Upload an image of a shoe to predict between one of these classes below:')
st.caption('- Ballet Flat')
st.caption('- Boat')
st.caption('- Brogue')
st.caption('- Clog')
st.caption('- Sneaker')
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    np_image = np.array(img).astype('float32')
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    
    dict = {0: 'Ballet Flat',
    1: 'Boat',
    2: 'Brogue',
    3: 'Clog',
    4: 'Sneaker'}
    
    with st.spinner("Predicting..."):
        model = load_model('dogs_model.h5')
        y_pred = model.predict(np_image)
    y_class = [np.argmax(element) for element in y_pred]
    conf = y_pred[0][y_class[0]]*100
    res = "Prediction result: {} - Confidence: {}%".format(y_class[0], round(conf, 3))
    st.image(img)
    st.success(res)