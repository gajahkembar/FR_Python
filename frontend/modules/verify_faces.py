import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:8000/api/verify"

def render():
    st.title("✅ Verifikasi 1:1 Wajah")

    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader("Upload Gambar Wajah 1", type=["jpg", "jpeg", "png"], key="verify1")
        img1 = None
        if file1:
            file1.seek(0)
            img1 = file1.read()
            st.image(img1, caption="🖼️ Wajah 1", width=250)

    with col2:
        file2 = st.file_uploader("Upload Gambar Wajah 2", type=["jpg", "jpeg", "png"], key="verify2")
        img2 = None
        if file2:
            file2.seek(0)
            img2 = file2.read()
            st.image(img2, caption="🖼️ Wajah 2", width=250)

    if img1 and img2 and st.button("🔍 Verifikasi"):
        with st.spinner("Memproses di engine..."):
            response = requests.post(
                API_URL,
                files={
                    "file1": ("face1.jpg", img1, "image/jpeg"),
                    "file2": ("face2.jpg", img2, "image/jpeg"),
                }
            )

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Verifikasi berhasil diproses.")
                st.markdown(f"- 📈 Similarity: `{result['similarity']:.4f}`")
                st.markdown(f"- 📌 Hasil: **{result['result']}**")
            else:
                try:
                    err = response.json()
                    detail = err.get("detail", "")
                    if "No face detected" in detail:
                        st.warning("⚠️ Wajah tidak terdeteksi di salah satu atau kedua gambar. Silakan unggah foto wajah yang jelas dan menghadap kamera.")
                    else:
                        st.error(f"❌ Terjadi kesalahan: {detail}")
                except Exception:
                    st.error(f"❌ Gagal memproses verifikasi. Status: {response.status_code}")