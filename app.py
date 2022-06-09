import streamlit as st
name = 'cat'
if name:
    st.title(f'Hello {name}')
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer:
        st.image(cv2_img)
        'Photo taken'
