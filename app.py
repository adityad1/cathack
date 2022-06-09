import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer:
#     bytes_data = img_file_buffer.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    'Photo taken'

    if img_file_buffer:
        img = Image.open(img_file_buffer)
        cropped_img = st_cropper(img)
    st.image(cropped_img, width = 1000)
