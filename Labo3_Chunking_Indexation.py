#  import what needed
import os

from PyPDF2 import PdfReader 
#  pdfreader is not so efficient, give a try with pdfdocument
from pypdfium2 import PdfDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
#  chroma , that will play DB role
from langchain_chroma import Chroma
# from langchain.embeddings import ollamaEmbeddings
from langchain_ollama import OllamaEmbeddings



#  set the model // no memory for local treatment 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)



#  read a pdf, i downloaded a pdf to the folder
reader = PdfDocument("Agent_With_RAG/Dessert_recipes.pdf") 
# text = "".join(page.extract_text() for page in reader.pages)
text = ""
for page in reader:
    text += page.get_textpage().get_text_range()
# split the text into chunks of 500 token, let the chunks overlap one another over 100 tokens
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
chunks = splitter.split_text(text)




#  index the chuns , and reate a db
vectordb = Chroma.from_texts(chunks, embedding=embeddings)
# print number of chunks
print('----------- nombre de chunks -----------------')
print(f"Nombre de chunks indexés : {len(chunks)}")
print("----------------------------------------------")
# Snapshot of the DB
print("-------Snapshot of the DB---------------")
results = vectordb.similarity_search("", k=5)
for i, r in enumerate(results):
    print("------------------------------------")
    print(f"\nChunk {i+1}:\n{r.page_content}")
    print("------------------------------------")

