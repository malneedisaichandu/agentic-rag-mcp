
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

# 1. Validate and load the PDF
file_path = r"D:\GenAI_Projects\AgenticAI Projects\AgenticRAG\RPSAS INDIA FRESHER.pdf"

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"PDF file not found: {file_path}")

if os.path.getsize(file_path) == 0:
    raise ValueError("The PDF file is empty.")

with open(file_path, "rb") as f:
    header = f.read(8)

print("File header:", repr(header))

if not header.startswith(b"%PDF-"):
    raise ValueError(
        "This file does not have a standard PDF header. "
        "Verify that it is a genuine, uncorrupted PDF."
    )

loader = PyPDFLoader(file_path)
documents = loader.load()

if not documents:
    raise ValueError("No pages were extracted from the PDF.")

print(f"PDF loaded successfully: {len(documents)} pages")

# 2. Split PDF text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

if not chunks:
    raise ValueError(
        "No text chunks were extracted. The PDF may contain "
        "scanned images that require OCR."
    )

print("PDF split into chunks:", len(chunks))

# 3. Generate embeddings and store in Chroma
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv('OPENAI_API_KEY')
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="pdf_knowledge_base"
)

print("PDF injected into vector database successfully!")