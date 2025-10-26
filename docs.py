import faiss
from app import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import re


def clean_text(t):
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


text = ""
doc = fitz.open("IPCC_AR6_WGII_Chapter16.pdf")
for page in doc:
    text += page.get_text("text")

text = clean_text(text)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = text_splitter.split_text(text)
vectors = embedding_model.encode(texts)
index = faiss.IndexHNSWFlat(384, 4)
index.add(vectors)
