import streamlit as st
from modules import register_face, identify_face, verify_faces, gallery, logs, delete_face

st.set_page_config(page_title="Face Recognition", layout="wide")

st.sidebar.title("📸 Face Recognition")
page = st.sidebar.radio("Navigasi", [
    "Dashboard", "Register", "Identify", "Verify", "Gallery", "Logs", "Hapus Wajah"
])

if page == "Dashboard":
    st.title("📊 Dashboard")
    st.write("Selamat datang di sistem Face Recognition Engine.")

elif page == "Register":
    register_face.render()

elif page == "Identify":
    identify_face.render()

elif page == "Verify":
    verify_faces.render()

elif page == "Gallery":
    gallery.render()

elif page == "Logs":
    logs.render()

elif page == "Hapus Wajah":
    delete_face.render()