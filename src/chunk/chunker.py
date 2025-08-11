from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    # We use create_documents as it's a good practice for potentially adding metadata later.
    return text_splitter.create_documents([text])