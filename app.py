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
    page_title="Fundbüro",
    page_icon="🔍",
    layout="centered"
)

# ──────────────────────────────────────────
# CSS – Design exakt wie in den Bildern
# ──────────────────────────────────────────
st.markdown("""
<style>
  /* Hintergrund weiß */
  .stApp { background-color: white; }

  /* Alle Texte in kursiver Schreibschrift */
  html, body, [class*="css"] {
      font-family: 'Georgia', serif;
      color: black;
  }

  /* Buttons: weiß mit schwarzem Rahmen */
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

  /* Fertig-Button türkis */
  div[data-testid="stButton"].fertig-btn > button {
      background-color: #4ECDC4;
      color: white;
      border: none;
      width: auto;
      padding: 8px 40px;
  }

  /* Eingabefelder: weiß mit schwarzem Rahmen */
  input, textarea {
      font-family: 'Georgia', serif !important;
      font-style: italic !important;
      border: 2px solid black !important;
      border-radius: 0px !important;
  }

  /* Suchfeld */
  div[data-testid="stTextInput"] input {
      font-family: 'Georgia', serif;
      font-style: italic;
      border: 2px solid black;
      border-radius: 0px;
      font-size: 1.1rem;
  }

  /* Karten auf der Suchseite */
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

  /* Logo oben rechts */
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

  /* Zurück-Button kleiner */
  .back-btn > button {
      width: auto !important;
      padding: 6px 20px !important;
      font-size: 0.9rem !important;
  }

  /* Uploadfeld anpassen */
  div[data-testid="stFileUploader"] {
      border: 2px solid black;
      padding: 10px;
  }

  /* Verstecke Streamlit-Standardelemente */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Logo (oben rechts, auf allen Seiten)
# ──────────────────────────────────────────
st.markdown("""
<div class="logo-container">
  <div class="logo-symbol">⊕</div>
  <div class="logo-text">TU<br>ES</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Session State initialisieren
# ──────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "start"

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
# KI-Modell (gecacht, wird nur 1x geladen)
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

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        if st.button("Suchen  🔍"):
            st.session_state.page = "search"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Hochladen  ⬆"):
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
            "Foto hochladen  ⬆",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible"
        )

        ai_desc = ""
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=120)
            with st.spinner("KI analysiert das Bild …"):
                ai_desc = describe_image(image, processor, model)

        st.markdown("<br>", unsafe_allow_html=True)
        name = st.text_input("", placeholder="Name des Objekts")

        st.markdown("<br>", unsafe_allow_html=True)
        desc = st.text_input(
            "",
            value=ai_desc,
            placeholder="Beschreibung"
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
                    # Bild als Base64 speichern
                    img_b64 = ""
                    if uploaded_file:
                        img = Image.open(uploaded_file)
                        img_b64 = image_to_base64(img)

                    data = load_data()
                    data.append({
                        "name": name.strip(),
                        "description": desc.strip(),
                        "image_b64": img_b64
                    })
                    save_data(data)
                    st.success(f"„{name}" wurde gespeichert!")
                    st.session_state.page = "start"
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Zurück"):
            st.session_state.page = "start"
            st.rerun()

# ──────────────────────────────────────────
# SUCHSEITE
# ──────────────────────────────────────────
def page_search():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        query = st.text_input("", placeholder="Suchen 🔍")

        data = load_data()

        for item in data:
            name = item.get("name", "")
            desc = item.get("description", "")
            img_b64 = item.get("image_b64", "")

            # Filter
            q = query.strip().lower()
            if q and q not in name.lower() and q not in desc.lower():
                continue

            # Beschreibungs-Zeilen
            desc_lines = "".join(
                [f"<p>- {line.strip()}</p>"
                 for line in desc.split(",") if line.strip()]
            ) if desc else ""

            # Bild-HTML
            if img_b64:
                img_html = (
                    f'<img src="data:image/png;base64,{img_b64}" '
                    f'width="60" height="60" '
                    f'style="object-fit:cover;margin-right:12px;">'
                )
            else:
                img_html = '<div style="width:60px;font-size:2rem;">🖼</div>'

            st.markdown(f"""
            <div class="fund-card">
              {img_html}
              <div class="fund-card-text">
                <h4>{name}</h4>
                <p>Beschreibung:</p>
                {desc_lines}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Zurück"):
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
