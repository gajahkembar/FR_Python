import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

API_URL = "http://localhost:8000/api/identify"

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

def render():
    st.title("🔍 Identifikasi Multi-Wajah")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        uploaded_file = st.file_uploader("Upload Gambar Wajah (Max 4MB)", type=["jpg", "jpeg", "png"])

        image_bytes = None
        if uploaded_file:
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            st.image(image_bytes, caption="📷 Gambar Query", width=465)

        if image_bytes and st.button("🔎 Identifikasi"):
            with st.spinner("Memproses di engine..."):
                response = requests.post(
                    API_URL,
                    files={"file": ("query.jpg", image_bytes, "image/jpeg")}
                )
                if response.status_code == 200:
                    st.session_state.identify_result = response.json()
                else:
                    st.session_state.identify_result = {"error": response.text}

    with col2:
        result_data = st.session_state.get("identify_result")

        if result_data:
            if "error" in result_data:
                st.error(f"❌ Gagal: {result_data['error']}")
            else:
                faces = result_data.get("faces", [])
                st.success(f"✅ Ditemukan {len(faces)} wajah")

                for face in faces:
                    st.markdown(f"### 🧠 Wajah ke-{face['face_index'] + 1}")

                    if 'crop_image' in face:
                        query_crop_img = decode_base64_image(face['crop_image'])
                    else:
                        query_crop_img = None

                    for idx, match in enumerate(face["top_matches"], start=1):
                        st.markdown(f"**#{idx} Kandidat**")
                        cols = st.columns([1, 1, 3])  # [Query, Galeri, Metadata]

                        with cols[0]:
                            if query_crop_img:
                                st.image(query_crop_img, caption="🖼️ Query", width=150)

                        with cols[1]:
                            if 'gallery_image' in match:
                                match_crop_img = decode_base64_image(match['gallery_image'])
                                st.image(match_crop_img, caption="🖼️ Galeri", width=150)
                            else:
                                st.warning("❌")

                        with cols[2]:
                            st.markdown(f"- 🔗 UUID: `{match['user_id']}`")
                            st.markdown(f"- 👤 Nama: **{match['name']}**")
                            st.markdown(f"- 🏫 Asal: {match['origin']}")
                            st.markdown(f"- 📈 Similarity: `{match['similarity']:.4f}`")

                        st.markdown("---")