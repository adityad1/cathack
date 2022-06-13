# Loading necessary packages and models
import streamlit as st
@st.experimental_singleton
def initial():
    import cv2
    from dlib import get_frontal_face_detector, shape_predictor
    fd = get_frontal_face_detector()
    sp = shape_predictor('utils/shape_predictor_68_face_landmarks.dat')
    import numpy as np
    from xgboost import Booster, DMatrix
    from tensorflow.keras.models import model_from_json
    with open('utils/vgg.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('utils/vgg_weights.h5')
    xg = Booster()
    xg.load_model('utils/xgb.json')
    return cv2, fd, sp, np, DMatrix, model, xg
cv2, fd, sp, np, dmat, model, xg = initial()

# WebApp
st.subheader('Welcome to Ikshana')
st.image('utils/logo.png')
with st.expander('Additional Information 📝'):
    st.text('1.Please take an image that covers your entire face')
    st.text('2.Take the picture in good lighting setup')
    st.text('3.Avoid using flash while taking the picture')
st.selectbox(label = 'Photo Options', options = ['Camera', 'Upload'], key = 'upload')
if st.session_state['upload'] == 'Upload':
    img_file_buffer = st.file_uploader('Upload a picture')
else:
    img_file_buffer = st.camera_input('Take a picture')
labels = ['normal', 'mild', 'severe']
if img_file_buffer:
    if st.session_state['upload'] == 'Upload':
        st.image(img_file_buffer)
    img = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    gray, img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = fd(img)
    if faces:
        st.success('Face Detected 😄')
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
        col1, col2 = st.columns(2)
        col1.header('Left Eye')
        col1.image(left_eye, width = 200)
        col2.header('Right Eye')
        col2.image(right_eye, width = 200)
        pr = model.predict(np.array([cv2.resize(left_eye, (224, 224)), cv2.resize(right_eye, (224, 224))]))
        preds = xg.predict(dmat(pr))
        label = ['No Cataract Detected ☮', 'Mild Cataract Detected ⚠', 'Severe Cataract Detected 💀']
        col1.info(label[np.argmax(preds[0])])
        col2.info(label[np.argmax(preds[1])])
        st.warning('Ensure Quality for Accurate Results')
    else:
        st.error('Face Detection Failed 😶 Please Try Again!!!')