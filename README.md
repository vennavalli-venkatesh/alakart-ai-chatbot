# Alakart AI Chatbot Backend

Production-oriented AI chatbot backend for the Alakart healthcare/wellness application.

## Project Structure

```
alakart-ai-chatbot/
│
├── app/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   └── groq_service.py
│   └── main.py
│
├── data/
│   ├── navigation/
│   │   └── Alakart_App_Navigation_Guide.docx
│   ├── otc/
│   │   └── OTC Medicine Data.pdf
│   ├── products/
│   │   └── ABP Product Portfolio V7.pdf
│   └── wellness/
│       └── Wellness Data-1.pdf
│
├── tests/
│   ├── __init__.py
│   └── test_document_loader.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9+ installed

### Installation & Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Environment Variables:
   Ensure your `.env` file contains:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

### Running the Document Loader Test

```powershell
.\.venv\Scripts\python.exe tests/test_document_loader.py
```

### Running the Server

Start the FastAPI development server using Uvicorn:

```bash
uvicorn app.main:app --reload
```
