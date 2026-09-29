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
# CSS – Dancing Script Schriftart + Design
# ──────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;700&display=swap');

  /* Hintergrund weiß */
  .stApp { background-color: white; }

  /* Gesamte App in Dancing Script */
  html, body, [class*="css"], p, div, span, label, input {
      font-family: 'Dancing Script', cursive !important;
      color: black;
  }

  /* Streamlit Buttons: weiß, schwarzer Rahmen, Schreibschrift */
  div.stButton > button {
      background-color: white !important;
      color: black !important;
      border: 2.5px solid black !important;
      font-family: 'Dancing Script', cursive !important;
      font-size: 1.4rem !important;
      width: 100% !important;
      padding: 16px 20px !important;
      border-radius: 0px !important;
      margin-bottom: 14px !important;
      text-align: left !important;
      display: flex !important;
      justify-content: space-between !important;
  }
  div.stButton > button:hover {
      background-color: #f5f5f5 !important;
      border: 2.5px solid black !important;
      color: black !important;
  }

  /* Eingabefelder */
  div[data-testid="stTextInput"] input {
      font-family: 'Dancing Script', cursive !important;
      font-size: 1.3rem !important;
      border: 2.5px solid black !important;
      border-radius: 0px !important;
      background-color: white !important;
      padding: 14px 12px !important;
  }

  /* File Uploader */
  div[data-testid="stFileUploader"] {
      border: 2.5px solid black !important;
      border-radius: 0px !important;
      background-color: white !important;
      padding: 8px !important;
  }
  div[data-testid="stFileUploader"] label {
      font-family: 'Dancing Script', cursive !important;
      font-size: 1.3rem !important;
  }

  /* Karten auf der Suchseite */
  .fund-card {
      border: 2.5px solid black;
      padding: 10px 12px;
      margin-bottom: 14px;
      background-color: white;
      display: flex;
      align-items: flex-start;
      gap: 14px;
  }
  .fund-card-text h4 {
      font-family: 'Dancing Script', cursive;
      text-decoration: underline;
      margin: 0 0 4px 0;
      font-size: 1.25rem;
      font-weight: 700;
  }
  .fund-card-text p {
      font-family: 'Dancing Script', cursive;
      font-size: 1.0rem;
      margin: 2px 0;
  }

  /* Fertig-Button türkis – gezielt über Klasse */
  .fertig-btn > button {
      background-color: #4ECDC4 !important;
      color: white !important;
      border: none !important;
      width: auto !important;
      padding: 8px 50px !important;
      font-size: 1.2rem !important;
      text-align: center !important;
  }

  /* Verstecke Streamlit-Standard-UI */
  #MainMenu, footer, header { visibility: hidden; }

  /* Zurück-Button: kleiner */
  .back-btn > button {
      width: auto !important;
      padding: 6px 18px !important;
      font-size: 1.0rem !important;
      border: 1.5px solid black !important;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Session State
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
# KI-Modell
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
# Logo-Funktion (oben rechts, jede Seite)
# ──────────────────────────────────────────
def show_logo():
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        # Logo als Base64 einbetten → funktioniert auf Streamlit Cloud
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            '<div style="position:fixed;top:56px;right:14px;z-index:9999;">'
            '<img src="data:image/png;base64,' + logo_b64 + '" width="55">'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Fallback falls logo.png fehlt: Text-Version
        st.markdown(
            '<div style="position:fixed;top:56px;right:14px;z-index:9999;'
            'background:white;border:2px solid black;padding:2px 5px;'
            'text-align:center;line-height:1.1;">'
            '<span style="font-size:1.4rem;">&#8853;</span><br>'
            '<span style="font-family:Arial;font-weight:bold;'
            'font-size:0.75rem;color:red;">TU<br>ES</span>'
            '</div>',
            unsafe_allow_html=True
        )

# ──────────────────────────────────────────
# STARTSEITE
# ──────────────────────────────────────────
def page_start():
    show_logo()
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    if st.session_state.save_success:
        st.success(st.session_state.save_success)
        st.session_state.save_success = ""

    col1, col2, col3 = st.columns([0.5, 7, 0.5])
    with col2:
        if st.button("Suchen   🔍"):
            st.session_state.page = "search"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Hochladen   \u2b06"):
            st.session_state.page = "upload"
            st.rerun()

# ──────────────────────────────────────────
# HOCHLADESEITE
# ──────────────────────────────────────────
def page_upload():
    show_logo()
    processor, model = load_model()

    col1, col2, col3 = st.columns([0.5, 7, 0.5])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Foto hochladen   \u2b06",
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

        name = st.text_input(
            "name_label",
            placeholder="Name des Objekts",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        desc = st.text_input(
            "desc_label",
            value=ai_desc,
            placeholder="Beschreibung",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([2, 3, 2])
        with col_b:
            st.markdown('<div class="fertig-btn">', unsafe_allow_html=True)
            fertig = st.button("Fertig")
            st.markdown('</div>', unsafe_allow_html=True)

            if fertig:
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

        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("\u2190 Zurueck"):
            st.session_state.page = "start"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────
# SUCHSEITE
# ──────────────────────────────────────────
def page_search():
    show_logo()

    col1, col2, col3 = st.columns([0.5, 7, 0.5])
    with col2:
        query = st.text_input(
            "search_label",
            placeholder="Suchen  🔍",
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
                    ' style="object-fit:cover;margin-right:12px;'
                    'flex-shrink:0;">'
                )
            else:
                img_html = (
                    '<div style="width:60px;height:60px;font-size:2rem;'
                    'flex-shrink:0;">🖼</div>'
                )

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

        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("\u2190 Zurueck"):
            st.session_state.page = "start"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────
# Seitensteuerung
# ──────────────────────────────────────────
if st.session_state.page == "start":
    page_start()
elif st.session_state.page == "upload":
    page_upload()
elif st.session_state.page == "search":
    page_search()
