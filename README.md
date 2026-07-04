# Streamlit Local RAG Template

A local-first Streamlit template for document question answering, objection handling, and quiz generation.

This repository is a sanitized template. It does not include private documents, model files, vector indexes, API keys, or local runtime folders.

## License And Attribution

This project is shared for noncommercial use only.

You may use, copy, modify, and share this project for learning, research, or other noncommercial purposes. Commercial use is not permitted without prior written permission from the copyright holder.

If you use, copy, modify, or share this project, please retain the license notice and clearly credit the original project:

```text
Original project: https://github.com/Lori-33/streamlit-local-rag-template
Author: Lori-33
License: PolyForm Noncommercial License 1.0.0
```

## Features

- Local document loading for Markdown, TXT, PDF, and PPTX files
- Vector search through an OpenAI-compatible LM Studio embedding endpoint
- Chat generation through an OpenAI-compatible LM Studio chat endpoint
- Knowledge Q&A page with citations
- Objection-handling workflow with query expansion and evidence filtering
- Quiz generation from indexed documents

## Project Structure

```text
streamlit-local-rag-template/
├─ app.py
├─ config.py
├─ rag_engine.py
├─ llm_api.py
├─ objection_engine.py
├─ preprocess.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ .env.example
├─ start_app.bat
├─ pages/
│  ├─ 01_knowledge_qa.py
│  ├─ 02_objection_handling.py
│  └─ 03_quiz.py
├─ sample_docs/
│  ├─ .gitkeep
│  └─ example_product_faq.md
├─ sample_objections/
│  ├─ .gitkeep
│  └─ example_price_objection.md
└─ vector_store/
   └─ .gitkeep
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Start LM Studio and load:

```text
Chat model: qwen3-8b or any OpenAI-compatible local chat model
Embedding model: text-embedding-bge-m3 or any compatible embedding model
```

The default LM Studio base URL is:

```text
http://localhost:1234
```

## Configuration

Copy `.env.example` to `.env` for local use if you want to override defaults. Do not commit `.env`.

Supported environment variables:

```env
LMSTUDIO_URL=http://localhost:1234
LMSTUDIO_MODEL=qwen3-8b
EMBED_MODEL=text-embedding-bge-m3
LOCAL_RAG_DOCS_FOLDER=sample_docs
LOCAL_RAG_OBJECTION_DOCS_FOLDER=sample_objections
LOCAL_RAG_VECTOR_STORE_DIR=vector_store
```

## Add Your Documents

Put safe test documents in `sample_docs/` and safe objection-handling examples in `sample_objections/`, or point the environment variables to your own local folders.

Do not commit private PDFs, decks, spreadsheets, internal notes, generated indexes, or model files.

## Build The Index

Run the app:

```powershell
streamlit run app.py
```

Open the Knowledge Q&A page and click `Rebuild index`.

You can also use the starter script on Windows:

```powershell
.\start_app.bat
```

## Privacy Notes

- The template is designed for local document workflows.
- Source documents and vector indexes are ignored by default when they may contain private data.
- If you enable a cloud API endpoint, review your privacy and data handling requirements first.

## Safety Checklist Before Publishing

Run a repository scan before pushing:

```powershell
rg "private|secret|api_key|password|token|absolute_user_path|local_private_path" .
Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 30 FullName,Length
```

Confirm that the repository contains no private documents, local runtime folders, generated indexes, model files, or personal paths.
