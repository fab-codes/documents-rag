from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.config.config import GOOGLE_MODEL_ID

def create_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model=GOOGLE_MODEL_ID)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5})

    # 1. Prompt to create a contextualized question
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    DO NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}"),
    ])
    
    # 2. Chain to create the contextualized question
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. Final prompt to answer the question using the context
    qa_system_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    Be clear and concise. Cite the source if possible.
    If you don't know the answer, just say that you have not found the information in the provided resources.

    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 4. Chain that passes the documents to the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 5. Final RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain