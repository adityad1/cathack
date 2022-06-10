import streamlit as st
import cv2
import dlib
import numpy as np
# initialisation()
img_file_buffer = st.camera_input("Take a picture")
labels = ['normal', 'mild', 'severe']
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
    gray, img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = fd(img)
    if faces:
        st.balloons()
        st.success('Its Working')
        points = sp(gray, faces[0])
        cx, cy = [i.x for i in points.parts()[36:42]], [i.y for i in points.parts()[36:42]]
        x, y, radius = np.mean(cx, dtype = int), np.mean(cy, dtype = int), (max(cx) - min(cx)) // 2
        mask = np.zeros(img.shape, dtype="uint8")
        mask = cv2.circle(mask, center = (x, y), radius = radius, thickness = -1, color = (1, 1, 1))
        right_eye = (img * mask)[y - radius:y + radius, x - radius:x + radius]
        cx, cy = [i.x for i in points.parts()[42:48]], [i.y for i in points.parts()[42:48]]
        x, y, radius = np.mean(cx, dtype = int), np.mean(cy, dtype = int), (max(cx) - min(cx)) // 2
        mask = np.zeros(img.shape, dtype="uint8")
        mask = cv2.circle(mask, center = (x, y), radius = radius, thickness = -1, color = (1, 1, 1))
        left_eye = (img * mask)[y - radius:y + radius, x - radius:x + radius]
#         st.image((cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * mask)[y - radius:y + radius, x - radius:x + radius], width = 500)
        col1, col2 = st.columns(2)
        col1.header('Left Eye')
        col1.image(left_eye, width = 300)
        col2.header('Right Eye')
        col2.image(right_eye, width = 300)
        from tensorflow.keras.models import model_from_json
        with open('model.json', 'r') as f:
            js = f.read()
        model = model_from_json(js)
        model.load_weights('weights.h5')
        preds = model.predict(np.array([cv2.resize(left_eye, (224, 224)), cv2.resize(right_eye, (224, 224))]))
        col1.write('Result - ' + labels[np.argmax(preds[0])])
        col2.write('Result - ' + labels[np.argmax(preds[1])])
    else:
        st.error('Where\'s your face?')

#     if img_file_buffer:
#         img = Image.open(img_file_buffer)
#         cropped_img = st_cropper(img)
#     st.image(cropped_img, width = 1000)
