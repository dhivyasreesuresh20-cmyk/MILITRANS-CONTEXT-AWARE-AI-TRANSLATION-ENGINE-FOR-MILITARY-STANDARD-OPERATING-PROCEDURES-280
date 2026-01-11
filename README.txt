MILTRANS – Context-Aware AI Translation Engine for Military SOPs
📖 Project Overview

MILTRANS is an AI-powered multilingual translation system developed to address language barriers in military training environments. Military SOPs are often written in Hindi, while trainees come from diverse linguistic backgrounds. Generic translators fail to preserve military terminology and contextual meaning, which can lead to misinterpretation.

This project uses context-aware NLP models to translate Hindi SOP content into multiple Indian languages while preserving operational accuracy and intent.

🎯 Objectives

Translate Hindi military SOPs into multiple regional languages

Preserve domain-specific military terminology

Support multiple input formats (text, file, image, URL, audio)

Improve training efficiency and comprehension

Store translations to avoid repeated processing

🛠️ Technologies Used

Frontend & Backend: Streamlit

Programming Language: Python

AI / NLP Model: Facebook NLLB-200 (Hugging Face Transformers)

OCR: EasyOCR

Speech-to-Text: SpeechRecognition, PyDub

Database: MongoDB Atlas

Text Normalization: Indic NLP Library

🔄 System Workflow

User logs in and selects input type

Hindi text is collected from text, file, image, URL, or audio

Text is normalized for consistency

Context-aware AI model translates the content

Translations are stored in MongoDB

Output is displayed and downloadable

🌐 Supported Input Types

Direct text input

.txt file upload

Image upload (OCR)

Web URL scraping

Audio upload or live recording

🌍 Supported Output Languages

English

Tamil

Telugu

Kannada

Malayalam

Marathi

Gujarati

Punjabi

Urdu

Assamese

Bhojpuri

Maithili

Nepali

Manipuri (Meitei)
(and more via NLLB-200)

🗄️ Database Usage

MongoDB is used to:

Store original Hindi text

Store translated outputs

Prevent duplicate translations

Improve response time by reusing stored results

⚠️ Limitations

Limited to predefined SOP formats

Translation quality depends on available training data

Audio translation accuracy depends on speech clarity

🚀 Future Enhancements

Real-time voice-to-voice translation

Mobile application deployment

Expanded military domain dataset

Role-based authentication

Integration with live training systems

▶️ How to Run the Project
# Create environment
conda create -n miltrans python=3.10
conda activate miltrans

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

👩‍💻 Author

Dhivyasree Suresh
MCA | Data Science | AI & NLP Enthusiast
