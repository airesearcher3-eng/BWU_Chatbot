from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents


def filter_to_minimul_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of document
    objects containing only 'source' in metadata and the original page content.
    """
    minimul_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimul_docs.append(
            Document(
                metadata={"source": src},
                page_content=doc.page_content
            )
        )
    return minimul_docs


def text_split(minimul_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
        keep_separator=True,
    )
    texts_chunk = text_splitter.split_documents(minimul_docs)
    return texts_chunk

def download_embeddings():
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings