import streamlit as st
import cv2
import numpy as np
# name = st.session_state.get('name')
name = 'cat'
if name:
    st.title(f'Hello {name}')
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2_img)
        'Photo taken'
        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>

        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
#         st.write(cv2_img.shape)
# else:
#     st.text_input(label = 'Enter your name', key = 'name')
