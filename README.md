# 🐍 RAG-Based Python Tutor Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for learning Python, powered by ChromaDB, Sentence Transformers, and Google Gemini 1.5 Flash LLM. Ask questions about your Python tutorial PDF and get contextual, AI-powered answers!

---

## 🚀 Technologies Used

- **Python 3.8+**
- **Streamlit** — Interactive web UI
- **ChromaDB** — Vector database for document retrieval
- **Sentence Transformers** — Text embedding (all-MiniLM-L6-v2)
- **Google Gemini 1.5 Flash** — Large Language Model (LLM) via API
- **PyPDF2** — PDF reading
- **LangChain** — Text chunking and metadata

---

## 🧠 Techniques Used

- **Retrieval-Augmented Generation (RAG):**
  - Embeds and stores PDF chunks in ChromaDB
  - Retrieves relevant context for user queries
  - Augments LLM prompt with retrieved context
- **Semantic Embedding:**
  - Uses Sentence Transformers for high-quality text embeddings
- **Secure API Key Handling:**
  - Reads Gemini API key from environment variable (never hard-coded)
- **Chunking & Metadata:**
  - Splits PDF into overlapping chunks for better retrieval

---

## 📦 Project Structure

```
RAG_Assistant/
├── app.py                # Streamlit chatbot app
├── ingest.py             # PDF ingestion & embedding script
├── prompts.py            # Prompt templates
├── utils.py              # Embedding utility
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── chroma_db/            # ChromaDB persistent storage
│   └── ...
├── data/
│   └── Python_Tutorial_EDIT.pdf  # Example PDF
└── .gitignore            # Excludes .env, venv, etc.
```

---

## ⚡️ How to Run

### 1. Clone the repository
```sh
git clone https://github.com/Hars99/RAG_Based_Assistant_for_python_learning.git
cd RAG_Assistant
```

### 2. Set up a Python environment
```sh
python -m venv venv
venv\Scripts\activate  # On Windows
# Or
source venv/bin/activate  # On Mac/Linux
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Add your PDF
Place your Python tutorial PDF in the `data/` folder. Update the path in `ingest.py` if needed.

### 5. Ingest the PDF
```sh
python ingest.py
```

### 6. Set your Gemini API key
Set your API key as an environment variable:
```sh
$env:GEMINI_API_KEY="your-gemini-api-key"  # PowerShell (Windows)
# Or
export GEMINI_API_KEY="your-gemini-api-key"  # Bash (Mac/Linux)
```

### 7. Run the Streamlit app
```sh
streamlit run app.py
```

---

## 🔒 Security Notes
- **Never commit your `.env` or API keys to GitHub.**
- `.gitignore` is set up to exclude secrets and unnecessary files.

---

## 📝 Usage
- Ask questions about your PDF in the chat UI.
- The bot retrieves relevant context and answers using Gemini LLM.
- View retrieved context in the sidebar for transparency.

---

## 📚 References
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)

---

## 👤 Author
- [Hars99](https://github.com/Hars99)

---

## 🏷 License
MIT
