import streamlit as st
import PyPDF2
import docx
import os
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  # Updated import
from langchain.prompts import PromptTemplate
import tempfile
import re
# Configure page settings
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="🤖",
    layout="wide"
)
def read_pdf(file) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file) -> str:
    """Read text file"""
    return file.getvalue().decode('utf-8')

def process_file(uploaded_file) -> str:
    """Process different file formats and return text content"""
    if uploaded_file is None:
        return ""
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            text = read_pdf(tmp_file_path)
        elif file_extension in ['docx', 'doc']:
            text = read_docx(tmp_file_path)
        elif file_extension == 'txt':
            text = read_txt(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            text = ""
    finally:
        os.unlink(tmp_file_path)
    
    return text

def create_vector_db(text: str) -> FAISS:
    """Create FAISS vector database from text"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    texts = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

def get_gemini_response(query: str, context: str = None, api_key: str = None) -> str:
    """Get response from Gemini API"""
       # Setup Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        api_key=api_key
    )
    
    if context:
        # Define the prompt template with clear sections for different query types
        template = """
        You are a helpful assistant with access to specific context information. Follow these rules to answer:

        1. If the question is a greeting (e.g., "hi", "hello", "good morning"):
        - Respond politely with an appropriate greeting

        2. If the question is about the provided context:
        - Read the context carefully
        - Answer specifically based on the given information
        - Be concise and accurate
        - Include relevant quotes or references from the context when appropriate
        - Example: "Based on the provided information, [answer]"

        3. If the question cannot be answered using the context:
        - Clearly state that the information is not available
        - Example: "I apologize, but I cannot find information about [topic] in the provided documents."

        Context:
        {context}

        Question: {question}

        Answer: """
                
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        response = llm.predict(prompt.format(context=context, question=query))
    else:
        # Template for general conversation
        template = """You are a helpful assistant. Respond naturally to the following:
        
        User: {question}
        Assistant: """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"]
        )
        
        response = llm.predict(prompt.format(question=query))
    
    return response

def initialize_session_state():
    """Initialize session state variables"""
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    initialize_session_state()
    
    st.title("🤖 Intelligent Document Q&A System")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        if st.button("Save API Key"):
            if api_key:
                st.success("API Key saved successfully!")
            else:
                st.error("Please enter an API Key")
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, DOCX, or TXT)", 
            type=['pdf', 'docx', 'doc', 'txt']
        )
        
        if uploaded_file:
            with st.spinner('Processing document...'):
                text_content = process_file(uploaded_file)
                if text_content:
                    st.session_state.vectordb = create_vector_db(text_content)
                    st.success(f'✅ {uploaded_file.name} processed successfully!')
    
    
    
    with col2:
        st.subheader("💬 Chat Interface")
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Initial welcome message
        if not st.session_state.messages:
            boot = "السَّلاَمُ عَلَيْكُمْ وَرَحْمَةُ اللهِ وَبَرَكَاتُهُ\n\nWelcome! I'm here to help answer your questions.\nHow may I assist you today?"
            with st.chat_message("assistant"):
                st.markdown(boot)
                
            st.session_state.messages.append({"role": "assistant", "content": boot})


        
        # React to user input
        if prompt := st.chat_input("Ask me something..."):
            if not api_key:
                st.error("Please enter your Gemini API key in the sidebar.")
            else:
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
               
                if st.session_state.vectordb:
                    results = st.session_state.vectordb.similarity_search_with_score(prompt, k=2)
                    
                    context = "\n".join([doc.page_content for doc, _ in results])
                    response = get_gemini_response(prompt, context=context, api_key=api_key)

                else:
                    response = "Please upload a document first to ask document-related questions."

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
