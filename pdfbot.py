import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say "answer is not available in the context".
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template,   input_variables=["context", "question"])

    def call_huggingface(input_value):
        prompt_text = str(input_value)

        client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        response = client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ],

            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message["content"]

    chain = prompt | call_huggingface
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    # response = chain(
    #     {"input_documents":docs, "question": user_question}
    #     , return_only_outputs=True)

    # print(response)
    # st.write("Reply: ", response["output_text"])
    context = "\n\n".join([d.page_content for d in docs])

    response = chain.invoke({"context": context, "question": user_question})
    st.write("Reply:", response)





def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()