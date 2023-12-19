import os
import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone

# Constants
PINECONE_API_KEY = "1c7be374-afe2-4eec-9076-67d6dfabb479"
# OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"


# Function to load a document
def load_document(file):
    if file is None:
        st.warning("Please upload a document.")
        return None

    # Save the uploaded file to a temporary location
    with open("temp_file.pdf", "wb") as temp_file:
        temp_file.write(file.read())

    name, ext = os.path.splitext(file.name)

    if ext == ".pdf" or ext == "":
        from langchain.document_loaders import PyPDFLoader
        st.write(f"Loading {file.name}")
        loader = PyPDFLoader("temp_file.pdf")
    elif ext == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        st.write(f"Loading {file.name}")
        loader = Docx2txtLoader("temp_file.pdf")
    else:
        st.warning(f"Document type {ext} not supported")
        return None

    data = loader.load()

    # Remove the temporary file
    os.remove("temp_file.pdf")

    return data


# Function to chunk data
def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


# Function to insert or fetch Pinecone index
def insert_or_fetch_index(chunks, file_name):
    first_word = file_name[:4].lower()
    index_name = f"{first_word}"

    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY,environment="gcp-starter")

    if index_name not in pinecone.list_indexes():
        delete_pine_index(pinecone.list_indexes()[0])
        print(f"Creating index {index_name}.... It may take a while")
        pinecone.create_index(index_name, metric='cosine', dimension=1536, pods=1, pod_type='p1.x2')
    else:
        print(f"Index {index_name} already exists.")

    vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    print("Created Successfully")

    return vectorstore, index_name


# Function to delete Pinecone index
def delete_pine_index(index_name):
    pinecone.init(api_key=PINECONE_API_KEY,environment="gcp-starter")
    if index_name in pinecone.list_indexes():
        print(f"Deleting index {index_name}......")
        pinecone.delete_index(index_name)
        print("Deleted Successfully")
    else:
        print("Index does not Exist")


# Function to ask and get an answer
def ask_and_get_answer(query, vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(query)
    return answer


# Function to summarize text
def summarize_text(text):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    print(llm.get_num_tokens(text))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    chunks = text_splitter.create_documents([text])
    print(len(chunks))
    chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=False)
    output_summary = chain.run(chunks)
    return output_summary


# Main Streamlit app
def main():
    st.title("Document Summarization System with Q&A ability")
    st.write("Upload a document, get summary, and ask questions about it.")

    document_file = st.file_uploader("Upload Document", type=["pdf", "docx"])
    query = st.text_input("Enter Your Question", "Write the summary?")

    if st.button("Submit"):
        if document_file is not None:
            data = load_document(document_file)
            chunks = chunk_data(data)

            # Create a new Pinecone index for the new document
            vectorstore, index_name = insert_or_fetch_index(chunks, document_file.name)
            text = " ".join([page.page_content for page in data])

            if 'summary' in query.lower():
                output_summary = summarize_text(text)
                st.write(output_summary)
            else:
                answer = ask_and_get_answer(query, vectorstore)
                st.write(answer)


if __name__ == "__main__":
    main()
