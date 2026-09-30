"""
Setup Knowledge Base - Load documents into vector store
"""

from pathlib import Path
from loguru import logger
from rag.vector_store import VectorStore

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def load_knowledge_base():
    """Load knowledge base into vector store"""
    
    logger.info("=" * 80)
    logger.info("Loading Knowledge Base into Vector Store")
    logger.info("=" * 80)
    
    # Initialize vector store
    vector_db_path = "knowledge_base/vector_db"
    vector_store = VectorStore(vector_db_path)
    
    # Load document
    doc_path = Path("knowledge_base/personal_loans.txt")
    
    if not doc_path.exists():
        logger.error(f"Document not found: {doc_path}")
        return
    
    logger.info(f"Loading: {doc_path}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Chunk the document
    chunks = chunk_text(content, chunk_size=400, overlap=50)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Prepare for vector store
    ids = [f"personal_loans_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"document": "personal_loans", "chunk_id": i} for i in range(len(chunks))]
    
    # Add to vector store
    vector_store.add_documents(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info("=" * 80)
    logger.info(f"✅ Knowledge base loaded successfully!")
    logger.info(f"Total chunks in store: {vector_store.count()}")
    logger.info("=" * 80)
    
    # Test query
    logger.info("\nTesting retrieval...")
    results = vector_store.query("What interest rates do you offer?", n_results=2)
    
    if results:
        logger.info(f"Test query returned {len(results)} results")
        logger.info(f"Top result distance: {results[0]['distance']:.3f}")
        logger.info(f"Top result preview: {results[0]['text'][:150]}...")
    else:
        logger.warning("Test query returned no results")

if __name__ == "__main__":
    load_knowledge_base()
