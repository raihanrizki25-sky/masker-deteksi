import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Deteksi Masker", page_icon="😷")
st.title("😷 Deteksi Masker (CNN + DNN Face Detector)")

# =====================
# LOAD MODEL CNN
# =====================
model = tf.keras.models.load_model("mask_detector.h5")

with open("class_indices.json") as f:
    class_idx = json.load(f)

# =====================
# LOAD FACE DETECTOR (DNN)
# =====================
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def predict_mask(face_img):
    face_img = cv2.resize(face_img, (128,128))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    pred = model.predict(face_img, verbose=0)[0][0]
    return pred

def process_image(image):
    img = np.array(image.convert("RGB"))
    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    face_found = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            face_found = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            pred = predict_mask(face)

            if pred < 0.5:
                label = "WITH MASK"
                color = (0,255,0)
            else:
                label = "WITHOUT MASK"
                color = (255,0,0)

            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                img, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

    if not face_found:
        st.warning("Wajah tidak terdeteksi")

    return img

# =====================
# STREAMLIT UI
# =====================
option = st.radio(
    "Pilih metode input:",
    ["📁 Upload Foto", "📷 Webcam"]
)

if option == "📁 Upload Foto":
    file = st.file_uploader("Upload foto", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file)
        result = process_image(image)
        st.image(result, use_column_width=True)

if option == "📷 Webcam":
    cam = st.camera_input("Ambil gambar")
    if cam:
        image = Image.open(cam)
        result = process_image(image)
        st.image(result, use_column_width=True)
