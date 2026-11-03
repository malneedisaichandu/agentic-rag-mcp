# Agentic RAG

A small document-question-answering project that exposes an MCP server. It searches a PDF index first and uses DuckDuckGo results when the answer is not in the indexed documents.

## Features

- Retrieves relevant passages from a local Chroma database.
- Answers questions using the indexed PDF with OpenAI's `gpt-4.1-mini`.
- Falls back to DuckDuckGo search when the PDF does not contain the answer.
- Provides the `retrieve_documents` and `answer_question` MCP tools.

## Requirements

- Python
- An OpenAI API key
- Dependencies listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project folder and add your API key:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Index a PDF

The project includes `RPSAS INDIA FRESHER.pdf` and a pre-existing `chroma_db` directory. To create or rebuild the index, update `file_path` in `inject.py` to the location of your PDF, then run:

```bash
python inject.py
```

The script splits the PDF into text chunks, creates embeddings, and stores them in `chroma_db`.

## Run

Start the MCP server with:

```bash
python Rag_Bot.py
```

Configure your MCP client to launch this command from the project folder. The server exposes:

- `retrieve_documents(question)` — returns relevant PDF passages.
- `answer_question(question)` — answers from the PDF first, then tries a web search.
