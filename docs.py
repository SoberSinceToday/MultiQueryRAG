# import faiss
# from app import embedding_model
#
# ipcc_docs = [
#     "Climate change has caused substantial damages and increasingly irreversible losses in terrestrial, freshwater, and coastal ecosystems.",
#     "Human influence has warmed the atmosphere, ocean, and land.",
#     "Key risks are projected to increase with every increment of warming.",
#     "Extreme weather events are becoming more frequent and intense across regions.",
#     "Limiting global warming to 1.5°C requires rapid and far-reaching transitions.",
#     "Vulnerability to climate change differs substantially among regions and populations.",
#     "Mitigation and adaptation can reduce risks but cannot eliminate them.",
#     "Ecosystem degradation exacerbates climate-related hazards and risks.",
#     "Transitioning to sustainable energy reduces long-term climate risks.",
#     "Financial systems face significant climate-related risks and opportunities.",
# ]
#
# noise_docs = [
#     "The stock market fluctuated significantly during the fiscal year.",
#     "Artificial intelligence models require large-scale datasets.",
#     "The new iPhone was released with an improved camera system.",
#     "Global tourism recovered after travel restrictions were lifted.",
#     "Machine learning can improve customer recommendations."
# ]
#
# data = noise_docs + ipcc_docs
# vectors = embedding_model.encode(data)
# index = faiss.IndexHNSWFlat(384, 4)
# index.add(vectors)
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
