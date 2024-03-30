import streamlit as st
from dotenv import load_dotenv
import pickle
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title('💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    
    #### Made by [Bhavini Thakkar](https://www.linkedin.com/in/thakkarbhavini/)
    ''')
load_dotenv()
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    if pdf is not None:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

    # Embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)


        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        VectorStore.save_local("faiss_index")

    # Accept user question
        query = st.text_input("Ask questions about your PDF file:")

    #Answer the question
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
if __name__ == '__main__':
    main()