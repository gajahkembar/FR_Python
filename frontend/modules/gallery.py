import os
import streamlit as st
from PIL import Image

DATA_DIR = "data"

def render():
    show_gallery_page()

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
    for folder in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            name, origin, uuid = parse_folder_name(folder)
            if name and origin and uuid:
                img_path = os.path.join(folder_path, f"{folder}.jpg")
                if os.path.exists(img_path):
                    gallery.append({
                        "name": name,
                        "origin": origin,
                        "uuid": uuid,
                        "image_path": img_path
                    })
    return sorted(gallery, key=lambda x: x["name"].lower())

def search_gallery(gallery, keyword):
    return [g for g in gallery if keyword.lower() in g["name"].lower()]

def show_gallery_page():
    st.title("🖼️ Galeri Wajah")

    gallery = load_gallery()

    # Pencarian
    search_query = st.text_input("🔍 Cari nama", "").strip()
    if search_query:
        gallery = search_gallery(gallery, search_query)

    total = len(gallery)
    per_page = st.session_state.get("gallery_per_page", 5)
    page = st.session_state.get("gallery_page", 0)

    start = page * per_page
    end = start + per_page
    rows = gallery[start:end]
    total_pages = (total + per_page - 1) // per_page

    # Grid wajah (5 kolom)
    for i in range(0, len(rows), 5):
        cols = st.columns(5)
        for idx, data in enumerate(rows[i:i+5]):
            with cols[idx]:
                st.image(data["image_path"], width=300)
                st.markdown(f"<div style='font-weight:bold; font-size:16px'>{data['name']}</div>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:13px'>📍 <i>{data['origin']}</i></span>", unsafe_allow_html=True)
                st.code(data["uuid"], language="text")

        st.caption(f"Halaman {page+1} dari {total_pages} • Total wajah: {total}")

    # Navigasi & dropdown per halaman dalam satu baris
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        st.button("⬅️ Sebelumnya", on_click=lambda: st.session_state.update({"gallery_page": max(0, page - 1)}), disabled=page == 0)
    with nav2:
        st.selectbox(
            "📄 Per halaman",
            options=[1, 5, 10, 15, 20],
            index=[1, 5, 10, 15, 20].index(per_page),
            key="gallery_per_page",
            label_visibility="collapsed"
        )
    with nav3:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <form action="#">
                    <button type="submit" style="
                        background-color: #262730;
                        color: white;
                        padding: 0.5em 1.2em;
                        border: none;
                        border-radius: 0.5em;
                        cursor: pointer;
                        font-weight: bold;
                        opacity: {'0.5' if page >= total_pages - 1 else '1.0'};
                        pointer-events: {'none' if page >= total_pages - 1 else 'auto'};
                    ">
                        ➡️ Berikutnya
                    </button>
                </form>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tangani klik manual pakai hidden form submit
        if 'submit' in st.session_state and not page >= total_pages - 1:
            st.session_state.gallery_page = min(total_pages - 1, page + 1)
