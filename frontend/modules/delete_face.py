import os
import streamlit as st
import requests

DATA_DIR = "data"
MIDDLEWARE_URL = "http://localhost:8000"

def parse_folder_name(folder):
    try:
        name_origin, uuid = folder.rsplit("_", 1)
        parts = name_origin.split("_")
        origin = parts[-1]
        name = " ".join(parts[:-1])
        return name, origin, uuid
    except Exception:
        return None, None, None

def load_gallery():
    gallery = []
    if not os.path.exists(DATA_DIR):
        return gallery

    for folder in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            name, origin, uuid = parse_folder_name(folder)
            if name and origin and uuid:
                gallery.append({
                    "folder": folder,
                    "name": name,
                    "origin": origin,
                    "uuid": uuid,
                    "folder_path": folder_path
                })
    return gallery

def delete_from_middleware(name, origin):
    try:
        url = f"{MIDDLEWARE_URL}/api/delete"
        payload = {"name": name, "origin": origin}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        resp = requests.post(url, data=payload, headers=headers, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        st.error(f"❌ Gagal menghubungi middleware: {e}")
        return False

def render():
    st.title("🗑️ Hapus Wajah dari Galeri")

    gallery = load_gallery()

    search_query = st.text_input("🔍 Cari nama yang ingin dihapus").strip()
    if not search_query:
        st.info("Masukkan nama untuk mencari data yang ingin dihapus.")
        return

    matches = [g for g in gallery if search_query.lower() in g["name"].lower()]

    if not matches:
        st.warning("Tidak ditemukan wajah dengan nama tersebut.")
        return

    st.write(f"Ditemukan {len(matches)} hasil:")

    for item in matches:
        with st.expander(f"{item['name']} ({item['origin']}) — UUID: {item['uuid']}"):
            st.image(os.path.join(item["folder_path"], f"{item['folder']}.jpg"), width=250)
            if st.button(f"🗑️ Hapus {item['name']}", key=item['uuid']):
                if delete_from_middleware(item["name"], item["origin"]):
                    st.success("✅ Data di middleware dan executor berhasil dihapus.")
                else:
                    st.warning("⚠️ Gagal hapus dari middleware.")

if __name__ == "__main__":
    render()