# splitter.py
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

def token_len(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=token_len
)
