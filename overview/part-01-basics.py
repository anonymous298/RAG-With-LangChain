import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'Documents', 'lord_of_the_rings.txt')
persistent_dir = os.path.join(current_dir, 'db', 'chroma_db')

if not os.path.exists(persistent_dir):
    print('Persistent Directory Does not exists. Initializing Vector Store. ')

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory = persistent_dir
    )

    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")

