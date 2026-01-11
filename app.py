# app.py
# MILTRANS: Context-Aware AI Translation Engine for Military SOPs

import os
import re
import numpy as np
from io import BytesIO
from PIL import Image

import streamlit as st
import requests
from bs4 import BeautifulSoup

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import easyocr
from transformers import pipeline
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
from audio_recorder_streamlit import audio_recorder

from pymongo import MongoClient
from pymongo.server_api import ServerApi

# ---------------------------
# Configure pydub to use ffmpeg
# ---------------------------
ffmpeg_path = which("ffmpeg")
AUDIO_ENABLED = ffmpeg_path is not None

if AUDIO_ENABLED:
    AudioSegment.converter = ffmpeg_path
else:
    st.sidebar.warning("⚠ FFmpeg not found. Audio features disabled.")

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="MILTRANS: Hindi → Multi-Lang Translator",
    page_icon="🌐",
    layout="wide"
)

st.title("MILTRANS: Context-Aware AI Translation Engine for Military SOPs")

# ---------------------------
# SIMPLE LOGIN (Option A)
# ---------------------------
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def login_screen():
    st.subheader("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")


if not st.session_state.authenticated:
    login_screen()
    st.stop()

# ---------------------------
# MongoDB Connection
# ---------------------------
MONGODB_URI = None
try:
    MONGODB_URI = st.secrets["mongodb"]["uri"]
except Exception:
    MONGODB_URI = os.getenv("MONGODB_URI", None)

client = None
collection = None

if MONGODB_URI:
    try:
        client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
        client.admin.command("ping")
        st.sidebar.success("✅ Connected to MongoDB")
        # Use the same DB/collection as used in Colab upload
        db = client["translation_db"]
        collection = db["miltrans"]
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB connection failed: {e}")
else:
    st.sidebar.info("ℹ No MongoDB URI detected.")


def save_translation(source_text, translated_text, source_lang="hi", target_lang_code="en"):
    """Save a clean translation to MongoDB, skipping errors and duplicates."""
    if collection is None:
        return None

    # Don't save empty or obvious error translations
    if not translated_text:
        return None
    if isinstance(translated_text, str) and translated_text.startswith("[Error"):
        return None
    if isinstance(translated_text, str) and translated_text.startswith("Error:"):
        return None

    # Avoid duplicates
    existing = collection.find_one({
        "source_text": source_text,
        "source_lang": source_lang,
        "target_lang": target_lang_code
    })
    if existing:
        st.sidebar.info(f"ℹ {target_lang_code} already saved.")
        return existing

    new_doc = {
        "source_text": source_text,
        "translated_text": translated_text,
        "source_lang": source_lang,
        "target_lang": target_lang_code
    }
    collection.insert_one(new_doc)
    st.sidebar.success(f"✅ Saved translation ({target_lang_code})")
    return new_doc


def get_translation_from_db(source_text, source_lang="hi", target_lang_code="en"):
    """Fetch an existing translation from MongoDB, if present."""
    if collection is None:
        return None
    return collection.find_one({
        "source_text": source_text,
        "source_lang": source_lang,
        "target_lang": target_lang_code
    })

# ---------------------------
# Supported Languages (NLLB codes)
# ---------------------------
LANGUAGES = {
    "English": "eng_Latn",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Malayalam": "mal_Mlym",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Gujarati": "guj_Gujr",
    "Assamese": "asm_Beng",
    "Bhojpuri": "bho_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Nepali": "npi_Deva",
    "Manipuri (Meitei)": "mni_Beng"
}

# Language codes used in MongoDB dataset
LANG_CODES_DB = {
    "English": "en",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    # other languages will use key.lower() as fallback
}

# ---------------------------
# Sidebar Language Selection
# ---------------------------
target_languages = st.sidebar.multiselect(
    "Select target languages:",
    options=list(LANGUAGES.keys()),
    default=["English"]
)

if "English" not in target_languages:
    target_languages.insert(0, "English")

# ---------------------------
# OCR Reader
# ---------------------------
@st.cache_resource
def _init_reader():
    return easyocr.Reader(["hi"])


reader = _init_reader()

# ---------------------------
# Hindi Normalizer
# ---------------------------
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("hi")


def normalize_hindi(text):
    return normalizer.normalize(text or "")

# ---------------------------
# Translation Pipeline
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_nllb_translator(src, tgt):
    return pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang=src,
        tgt_lang=tgt
    )


def translate_text(text, tgt_lang):
    translator = load_nllb_translator("hin_Deva", tgt_lang)
    return translator(text)[0]["translation_text"]

# ---------------------------
# OCR Function
# ---------------------------
def extract_text_from_image(image_file):
    img = Image.open(image_file).convert("RGB")
    result = reader.readtext(np.array(img), detail=0)
    return " ".join(result).strip()

# ---------------------------
# Web Scraping
# ---------------------------
def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        return " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])
    except Exception as e:
        return f"Error: {e}"

# ---------------------------
# Audio Processing
# ---------------------------
def extract_text_from_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        segment = AudioSegment.from_file(audio_file)
        wav_io = BytesIO()
        segment.export(wav_io, format="wav")
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language="hi-IN")
    except Exception as e:
        return f"Audio error: {e}"

# ---------------------------
# Input UI
# ---------------------------
available_inputs = ["Text", "File (.txt)", "Image", "Web URL"]
if AUDIO_ENABLED:
    available_inputs.append("Audio")

input_type = st.radio("Choose input type:", available_inputs, horizontal=True)

hindi_text = ""

if input_type == "Text":
    hindi_text = st.text_area("Enter Hindi text:", height=150)

elif input_type == "File (.txt)":
    uploaded = st.file_uploader("Upload Hindi text file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="ignore")
        hindi_text = st.text_area("Edit text:", value=text, height=150)

elif input_type == "Image":
    img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if img:
        text = extract_text_from_image(img)
        hindi_text = st.text_area("Edit text:", value=text, height=150)

elif input_type == "Web URL":
    url = st.text_input("Enter URL:")
    if url:
        text = extract_text_from_url(url)
        hindi_text = st.text_area("Edit text:", value=text, height=150)

elif input_type == "Audio":
    st.info("Upload or record audio.")
    mode = st.radio("Audio input:", ["Upload File", "Record"], horizontal=True)

    if mode == "Upload File":
        audio = st.file_uploader("Upload audio", type=["mp3", "wav"])
        if audio:
            t = extract_text_from_audio(audio)
            hindi_text = st.text_area("Edit text:", value=t, height=150)

    elif mode == "Record":
        audio_bytes = audio_recorder()
        if audio_bytes:
            audio_file = BytesIO(audio_bytes)
            t = extract_text_from_audio(audio_file)
            hindi_text = st.text_area("Edit text:", value=t, height=150)

# ---------------------------
# Translate Button
# ---------------------------
if st.button("Translate"):
    if hindi_text.strip():
        normalized = normalize_hindi(hindi_text)
        translations = {}

        for lang in target_languages:
            # DB language code: use mapping if available, else fallback to lower-case name
            target_code_db = LANG_CODES_DB.get(lang, lang.lower())

            # 1️⃣ Try to fetch from MongoDB first
            existing = get_translation_from_db(normalized, "hi", target_code_db)

            if existing:
                tr = existing["translated_text"]
                st.sidebar.info(f"⚡ Loaded {lang} translation from database.")
            else:
                # 2️⃣ Only call model if not in DB
                with st.spinner(f"Translating to {lang}..."):
                    try:
                        tr = translate_text(normalized, LANGUAGES[lang])
                    except Exception as e:
                        tr = f"[Error: {e}]"

                # 3️⃣ Save to DB (function will skip errors)
                save_translation(normalized, tr, "hi", target_code_db)

            translations[lang] = tr

        st.subheader("Translated Texts")
        for lang, text in translations.items():
            st.markdown(f"**{lang}:** {text}")

        file_data = "Original Hindi:\n" + normalized + "\n\n"
        for lang, text in translations.items():
            file_data += f"{lang}:\n{text}\n\n"

        st.download_button(
            "⬇ Download Translations",
            data=file_data.encode("utf-8"),
            file_name="translations.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please enter text.")
