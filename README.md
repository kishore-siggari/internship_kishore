# internship_kishore


""" RAG Engine - Core Retrieval-Augmented Generation Logic 
Handles PDF parsing, chunking, embedding, and question answering. """

import os
from typing import Optional
from dotenv import load_dotenv

# ✅ Correct imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate


load_dotenv()

SYSTEM_PROMPT = """You are an intelligent assistant that answers questions 
based strictly on the provided PDF document context. Be accurate, concise, 
and cite specific parts of the document when possible. 

Context from document: {context} 
Question: {question} 

Answer based only on the above context. 
If the answer is not found in the context, say 
"I couldn't find that information in the uploaded document." 

Answer:"""


class RAGEngine:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")

        # ✅ Embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=self.api_key,   # Correct param name
            model="text-embedding-ada-002"
        )

        # ✅ LLM
        self.llm = ChatOpenAI(
            api_key=self.api_key,   # Correct param name
            model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1024
        )

        # ✅ Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def build_vectorstore(self, pdf_path: str) -> FAISS:
        """Load a PDF, split into chunks, embed, and store in FAISS."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("PDF appears to be empty or unreadable.")

        chunks = self.text_splitter.split_documents(documents)
        print(f"[RAG] Split PDF into {len(chunks)} chunks.")

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print(f"[RAG] Vector store created with {len(chunks)} embeddings.")

        return vectorstore

    def answer_question(self, vectorstore: FAISS, question: str) -> dict:
        """Retrieve relevant chunks and generate an answer using LLM."""
        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        result = qa_chain.invoke({"query": question})

        sources = []
        for doc in result.get("source_documents", []):
            page = doc.metadata.get("page", "?")
            source = f"Page {page + 1}"
            if source not in sources:
                sources.append(source)

        return {
            "answer": result["result"],
            "sources": sources
        }
