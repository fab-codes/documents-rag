# Documents RAG

## Overview

This project provides a simple RAG (Retrieval-Augmented Generation) pipeline.
It takes a document from the `files` directory, splits it into chunks, and generates embeddings for each chunk using an external embedding service (for example, the [fab-codes/embedding-service](https://github.com/fab-codes/embedding-service)).

The embeddings are then stored in a vector database, allowing semantic search or retrieval over the document contents.

---

## How to run with Docker

In the project root, run:

```bash
docker-compose up --build -d
```

This will build and start the service together with its dependencies.

---

## How to run locally

1. Set the document path in the `.env` file (for example, `PDF_FILE_PATH=files/mydocument.pdf`).
2. Open a shell inside the container and go to the `app` directory:

   ```bash
   docker exec -it documents-rag bash
   cd app
   ```

3. Run the main script:

   ```bash
   python -m src.main
   ```

---

## Environment Variables

Main variables to configure in `.env`:

- `PDF_FILE_PATH` → path to the file you want to process inside the `files/` directory.
- `EMBEDDING_SERVICE_URL` → URL of the external embedding service (default: `http://embedding-service:8000`).
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` → connection details for the vector database.

---

## How it works

1. Load a document (just PDF in this moment) from the `files` folder.
2. Split the document into smaller text chunks.
3. Call the external embedding service to generate embeddings for each chunk.
4. Store the embeddings in Qdrant.
5. Enable retrieval and semantic search over the indexed document.
