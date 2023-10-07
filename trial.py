import streamlit as st
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import cv2
from PIL import Image, ImageOps


def click_button_yes():
    st.session_state["res"] = "yes"


def click_button_no():
    st.session_state["res"] = "no"


@st.cache_resource
def load_model():
    mod = VGG16(weights="imagenet", include_top=False)
    mod.trainable = False
    embedding_model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1.0 / 255),
            mod,
            tf.keras.layers.Normalization(mean=0, variance=1),
            tf.keras.layers.Flatten(),
        ],
        name="embedding_model",
    )
    return embedding_model


def preprocess_and_extract(img, model):
    size = (224, 224)
    img2 = ImageOps.fit(img, size, Image.ANTIALIAS)
    x = image.img_to_array(img2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return tf.squeeze(features)


model = load_model()

if "res" not in st.session_state:
    st.session_state["res"] = "no"


if st.session_state["res"] == "no":
    st.title("Search by Image")
    with st.container():
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, width=250)
            st.button("Search", type="primary", key="search", on_click=click_button_yes)
            feat = preprocess_and_extract(img, model)
            st.write(feat.shape)

if st.session_state["res"] == "yes":
    st.title("Similar Image Results")
    with st.container():
        st.image(["1.jpg", "2.jpg", "3.png"], width=250)
        st.button("Back", type="primary", key="back", on_click=click_button_no)
