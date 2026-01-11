
MILTRANS Streamlit App — Step-by-Step

1) Create a Python env (recommended - Anaconda)
   - Open Anaconda Prompt
   - conda create -n militrans python=3.10 -y
   - conda activate militrans

2) Install packages
   - pip install -r requirements.txt

3) Install FFmpeg (system tool, needed for audio)
   - Windows (Chocolatey): choco install ffmpeg
   - macOS (Homebrew): brew install ffmpeg
   - Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg

4) Add your MongoDB URI (two options)
   A) Secrets file (recommended):
      - Create a folder ".streamlit" next to app.py
      - Create a file ".streamlit/secrets.toml" with content:
        [mongodb]
        uri = "mongodb+srv://<user>:<pass>@<cluster-url>/?retryWrites=true&w=majority&appName=<AppName>"
   B) Environment variable:
      - set MONGODB_URI="mongodb+srv://<user>:<pass>@<cluster-url>/?retryWrites=true&w=majority&appName=<AppName>"

5) Run the app
   - streamlit run app.py
   - A browser tab will open with the app.

6) Use the app
   - Choose input type: Text, File (.txt), Image, Web URL, or Audio.
   - Click Translate to get outputs for selected languages.
   - If MongoDB is configured, each translation is saved (one doc per target language).

7) Verify saved data (optional quick check in Python)
   from pymongo import MongoClient
   client = MongoClient("<your-uri>")
   docs = list(client["translations_db"]["translations"].find().limit(5))
   for d in docs: print(d)

Troubleshooting
- ModuleNotFoundError: sentencepiece / torch / etc.
  Run: pip install -r requirements.txt (inside your activated env)
- FFmpeg not found
  Install FFmpeg and restart the terminal; ensure 'ffmpeg' works on the command line.
- MongoDB auth failed or network error
  Double-check your MongoDB username/password, IP allowlist, and connection string format.
- Model download slow
  The first run downloads "facebook/nllb-200-distilled-600M"; wait until it completes.
