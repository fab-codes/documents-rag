from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_pages(pages: list[tuple[int, str]], pdf_path: str) -> list[Document]:
    """
    Splits the text into chunks while preserving the page metadata.
    """
    print("Starting chunking process...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # TODO: Handle right separators
        separators=["\n\n", "\n", ". ", " ", ""], # Gives priority to paragraphs and sentences
        length_function=len,
    )

    all_chunks = []
    for page_num, page_text in pages:
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            # Create a LangChain Document object with text and metadata
            doc = Document(
                page_content=chunk,
                metadata={"source": pdf_path, "page": page_num}
            )
            all_chunks.append(doc)

    print(f"Chunking complete. Created {len(all_chunks)} chunks.")
    return all_chunks