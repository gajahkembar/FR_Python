import streamlit as st
import requests

API_URL = "http://localhost:8000/api/register"

def render():
    st.title("📝 Daftarkan Wajah Baru")

    col1, col2 = st.columns([1.5, 1])  # kiri: form, kanan: preview

    image_bytes = None  # buffer untuk preview dan upload

    with col1:
        name = st.text_input("Nama Lengkap")
        origin = st.text_input("Asal / Instansi")
        uploaded_file = st.file_uploader("Upload Gambar Wajah (Max 4MB)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            if uploaded_file.size > 4 * 1024 * 1024:
                st.warning("Ukuran file melebihi 4MB. Silakan unggah gambar yang lebih kecil.")
                return
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()

        if image_bytes and name and origin:
            if st.button("Daftarkan"):
                with st.spinner("Mengirim ke engine..."):
                    response = requests.post(
                        API_URL,
                        files={"file": ("filename.jpg", image_bytes, "image/jpeg")},
                        data={"name": name, "origin": origin}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status")

                        if status == "success":
                            st.success(f"Wajah berhasil didaftarkan. UUID: {data.get('user_id')}")
                            if data.get("crop_url"):
                                col2.image(data.get("crop_url"), caption="Crop dari engine", width=300)

                        elif status == "duplicate_name_origin":
                            st.warning("Wajah dengan nama dan asal yang sama sudah terdaftar sebelumnya.")
                        else:
                            st.error(f"Gagal mendaftar: status tidak dikenal: {status}")

                    else:
                        try:
                            error_msg = response.json()
                            if "RESOURCE_EXHAUSTED" in str(error_msg):
                                st.error("Ukuran gambar terlalu besar. Silakan upload gambar di bawah 4MB.")
                            else:
                                st.error(f"Gagal mendaftar: {error_msg}")
                        except:
                            st.error("Terjadi kesalahan saat mendaftar. Silakan coba lagi.")

    with col2:
        if image_bytes:
            st.image(image_bytes, caption="Preview Gambar", width=500)