import streamlit as st
# @st.experimental_singleton
# def initialisation():
#     from streamlit_cropper import st_cropper
#     from PIL import Image
import cv2
import dlib
import numpy as np
# initialisation()
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer:
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    'Photo taken'
    @st.experimental_singleton
    def dlib_objs():
        fd = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        return fd, sp
    fd, sp = dlib_objs()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fd(img)
    if faces:
        st.balloons()
        st.success('Its Working')
        points = sp(gray, faces[0])
        cx, cy = [i.x for i in points.parts()[36:42]], [i.y for i in points.parts()[36:42]]
        x, y, radius = np.mean(cx, dtype = int), np.mean(cy, dtype = int), (max(cx) - min(cx)) // 2
        mask = np.zeros(img.shape, dtype="uint8")
        mask = cv2.circle(mask, center = (x, y), radius = radius, thickness = -1, color = (1, 1, 1))
        st.image((cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * mask)[y - radius:y + radius, x - radius:x + radius], width = 500)
    else:
        st.error('Where\'s your face?')

#     if img_file_buffer:
#         img = Image.open(img_file_buffer)
#         cropped_img = st_cropper(img)
#     st.image(cropped_img, width = 1000)
