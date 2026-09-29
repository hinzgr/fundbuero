import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import os
import base64
from io import BytesIO

DATA_FILE = "fundbuero_data.json"

# ──────────────────────────────────────────
# Seiten-Konfiguration
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Fundbuero",
    page_icon="🔍",
    layout="centered"
)

# ──────────────────────────────────────────
# CSS – Design wie in den Bildern
# ──────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: white; }

  html, body, [class*="css"] {
      font-family: 'Georgia', serif;
      color: black;
  }

  div.stButton > button {
      background-color: white;
      color: black;
      border: 2px solid black;
      font-family: 'Georgia', serif;
      font-style: italic;
      font-size: 1.2rem;
      width: 100%;
      padding: 18px;
      border-radius: 0px;
      margin-bottom: 12px;
  }
  div.stButton > button:hover {
      background-color: #f0f0f0;
      border: 2px solid black;
      color: black;
  }

  input, textarea {
      font-family: 'Georgia', serif !important;
      font-style: italic !important;
      border: 2px solid black !important;
      border-radius: 0px !important;
  }

  div[data-testid="stTextInput"] input {
      font-family: 'Georgia', serif;
      font-style: italic;
      border: 2px solid black;
      border-radius: 0px;
      font-size: 1.1rem;
  }

  .fund-card {
      border: 2px solid black;
      padding: 10px;
      margin-bottom: 12px;
      background-color: white;
      display: flex;
      align-items: flex-start;
      gap: 14px;
  }
  .fund-card-text h4 {
      font-family: 'Georgia', serif;
      font-style: italic;
      text-decoration: underline;
      margin: 0 0 4px 0;
      font-size: 1.1rem;
  }
  .fund-card-text p {
      font-family: 'Georgia', serif;
      font-style: italic;
      font-size: 0.88rem;
      margin: 2px 0;
  }

  .logo-container {
      position: fixed;
      top: 56px;
      right: 18px;
      z-index: 999;
      text-align: center;
      line-height: 1;
  }
  .logo-symbol {
      font-size: 1.6rem;
  }
  .logo-text {
      font-family: Arial, sans-serif;
      font-weight: bold;
      font-size: 0.75rem;
      color: red;
  }

  div[data-testid="stFileUploader"] {
      border: 2px solid black;
      padding: 10px;
  }

  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Logo (oben rechts, auf allen Seiten)
# ──────────────────────────────────────────
st.markdown("""
<div class="logo-container">
  <div class="logo-symbol">&#8853;</div>
  <div class="logo-text">TU<br>ES</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Session State initialisieren
# ──────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "start"

if "save_success" not in st.session_state:
    st.session_state.save_success = ""

# ──────────────────────────────────────────
# Datenverwaltung
# ──────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────
# KI-Modell (wird nur 1x geladen)
# ──────────────────────────────────────────
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    return processor, model

def describe_image(pil_image, processor, model):
    image = pil_image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def image_to_base64(pil_image, size=(70, 70)):
    pil_image.thumbnail(size)
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ──────────────────────────────────────────
# STARTSEITE
# ──────────────────────────────────────────
def page_start():
    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.session_state.save_success:
        st.success(st.session_state.save_success)
        st.session_state.save_success = ""

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        if st.button("Suchen  🔍"):
            st.session_state.page = "search"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Hochladen  \u2b06"):
            st.session_state.page = "upload"
            st.rerun()

# ──────────────────────────────────────────
# HOCHLADESEITE
# ──────────────────────────────────────────
def page_upload():
    processor, model = load_model()

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Foto hochladen  \u2b06",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible"
        )

        ai_desc = ""
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=120)
            with st.spinner("KI analysiert das Bild ..."):
                ai_desc = describe_image(image, processor, model)

        st.markdown("<br>", unsafe_allow_html=True)
        name = st.text_input("name_label", placeholder="Name des Objekts",
                             label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        desc = st.text_input(
            "desc_label",
            value=ai_desc,
            placeholder="Beschreibung",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Türkiser Fertig-Button
        st.markdown("""
        <style>
        div[data-testid="stButton"]:last-of-type > button {
            background-color: #4ECDC4 !important;
            color: white !important;
            border: none !important;
            width: auto !important;
            padding: 8px 50px !important;
            display: block;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([2, 2, 2])
        with col_b:
            if st.button("Fertig"):
                if not name.strip():
                    st.warning("Bitte einen Namen eingeben.")
                else:
                    img_b64 = ""
                    if uploaded_file:
                        uploaded_file.seek(0)
                        img = Image.open(uploaded_file)
                        img_b64 = image_to_base64(img)

                    data = load_data()
                    data.append({
                        "name": name.strip(),
                        "description": desc.strip(),
                        "image_b64": img_b64
                    })
                    save_data(data)
                    st.session_state.save_success = (
                        name.strip() + " wurde gespeichert!"
                    )
                    st.session_state.page = "start"
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("\u2190 Zurueck"):
            st.session_state.page = "start"
            st.rerun()

# ──────────────────────────────────────────
# SUCHSEITE
# ──────────────────────────────────────────
def page_search():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        query = st.text_input(
            "search_label",
            placeholder="Suchen 🔍",
            label_visibility="collapsed"
        )

        data = load_data()

        for item in data:
            name = item.get("name", "")
            desc = item.get("description", "")
            img_b64 = item.get("image_b64", "")

            q = query.strip().lower()
            if q and q not in name.lower() and q not in desc.lower():
                continue

            desc_lines = "".join(
                ["<p>- " + line.strip() + "</p>"
                 for line in desc.split(",") if line.strip()]
            ) if desc else ""

            if img_b64:
                img_html = (
                    '<img src="data:image/png;base64,' + img_b64 + '"'
                    ' width="60" height="60"'
                    ' style="object-fit:cover;margin-right:12px;">'
                )
            else:
                img_html = '<div style="width:60px;font-size:2rem;">🖼</div>'

            st.markdown(
                '<div class="fund-card">'
                + img_html
                + '<div class="fund-card-text">'
                + "<h4>" + name + "</h4>"
                + "<p>Beschreibung:</p>"
                + desc_lines
                + "</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("\u2190 Zurueck"):
            st.session_state.page = "start"
            st.rerun()

# ──────────────────────────────────────────
# Seitensteuerung
# ──────────────────────────────────────────
if st.session_state.page == "start":
    page_start()
elif st.session_state.page == "upload":
    page_upload()
elif st.session_state.page == "search":
    page_search()
