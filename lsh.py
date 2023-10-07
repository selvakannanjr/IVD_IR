from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np


def load_model():
    mod = VGG16(
        weights="imagenet", include_top=False, pooling="avg"
    )  # if pooling arg is removed,the output shape is 20588
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


def preprocess_and_extract(filename, model):
    img = Image.open(filename)
    size = (224, 224)
    img2 = ImageOps.fit(img, size, Image.ANTIALIAS)
    x = image.img_to_array(img2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return tf.squeeze(features)


model = load_model()

feat = preprocess_and_extract("1.jpg", model)

print(feat)
