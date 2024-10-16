import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2


LABELS = ['glioma','meningioma','notumor','pituitary']

def load_model():
    model = tf.keras.models.load_model('my_model.keras')
    return model

model = load_model()

st.write("# Brain Tumor Detection")

file = st.file_uploader("Choose File", type=['jpg','png','jpeg'])


def import_and_predict(image_data, model):
    size = (224,224)

    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload an Image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    predicted_label = LABELS[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.header("This image most likely belongs to {} with a {:.2f} percent confidence.".format(predicted_label, confidence))