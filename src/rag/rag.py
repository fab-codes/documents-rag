from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config.config import GOOGLE_MODEL_ID

def create_rag_chain(vector_store):
    """
    Create complete RAG chain using Gemini.
    """
    retriever = vector_store.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL_ID)

    prompt_template = """
    Answer the user's question based EXCLUSIVELY on the following context.
    Be clear and concise. Cite the page number and the name of the source if possible.
    If the information is not in the context, reply: "I have not found this information among the resources provided."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain