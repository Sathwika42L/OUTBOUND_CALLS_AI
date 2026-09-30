"""
📄 DOCUMENT UPLOAD & PROCESSING SYSTEM
=======================================

Upload ANY document (PDF, DOCX, TXT) → automatically processes into RAG

Usage:
    python upload_document.py path/to/document.pdf
    python upload_document.py path/to/document.txt
"""

import sys
from pathlib import Path
from loguru import logger
from rag.vector_store import VectorStore

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF"""
    try:
        import PyPDF2
        text = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    except ImportError:
        logger.error("PyPDF2 not installed. Run: pip install PyPDF2")
        return ""

def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from DOCX"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return '\n'.join(text)
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return ""

def process_document(file_path: str):
    """Process and upload document to vector store"""
    
    logger.info("=" * 80)
    logger.info("📄 DOCUMENT UPLOAD & PROCESSING")
    logger.info("=" * 80)
    
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Processing: {path.name}")
    
    # Extract text based on file type
    suffix = path.suffix.lower()
    
    if suffix == '.txt':
        text = extract_text_from_txt(path)
    elif suffix == '.pdf':
        text = extract_text_from_pdf(path)
    elif suffix == '.docx':
        text = extract_text_from_docx(path)
    else:
        logger.error(f"Unsupported file type: {suffix}")
        logger.info("Supported: .txt, .pdf, .docx")
        return
    
    if not text or len(text) < 50:
        logger.error("No text extracted from document")
        return
    
    logger.info(f"Extracted {len(text)} characters")
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Initialize vector store
    vector_db_path = "knowledge_base/vector_db"
    vector_store = VectorStore(vector_db_path)
    
    # Get current count
    before_count = vector_store.count()
    logger.info(f"Current chunks in store: {before_count}")
    
    # Prepare metadata
    doc_name = path.stem  # filename without extension
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"document": doc_name, "chunk_id": i, "source": path.name} for i in range(len(chunks))]
    
    # Add to vector store
    logger.info("Uploading to vector store...")
    vector_store.add_documents(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    after_count = vector_store.count()
    logger.info("=" * 80)
    logger.info(f"✅ Document processed successfully!")
    logger.info(f"   Document: {path.name}")
    logger.info(f"   Chunks added: {len(chunks)}")
    logger.info(f"   Total in store: {after_count} (was {before_count})")
    logger.info("=" * 80)
    
    # Test query
    logger.info("\nTesting retrieval...")
    results = vector_store.query("campaign", n_results=2, document_filter=doc_name)
    
    if results:
        logger.info(f"✅ Test successful!")
        logger.info(f"   Retrieved {len(results)} chunks from '{doc_name}'")
        logger.info(f"   Top result distance: {results[0]['distance']:.3f}")
    else:
        logger.warning("⚠️  Test query returned no results")
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 Ready to use! Run: python dynamic_bot.py")
    logger.info("=" * 80)

def list_documents():
    """List all documents in vector store"""
    logger.info("=" * 80)
    logger.info("📚 DOCUMENTS IN KNOWLEDGE BASE")
    logger.info("=" * 80)
    
    vector_db_path = "knowledge_base/vector_db"
    vector_store = VectorStore(vector_db_path)
    
    docs = vector_store.list_documents()
    
    if not docs:
        logger.info("No documents found. Upload one with:")
        logger.info("  python upload_document.py path/to/file.pdf")
    else:
        logger.info(f"Total documents: {len(docs)}")
        logger.info(f"Total chunks: {vector_store.count()}")
        logger.info("\nDocuments:")
        for i, doc in enumerate(docs, 1):
            logger.info(f"  {i}. {doc}")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Upload document:  python upload_document.py path/to/document.pdf")
        print("  List documents:   python upload_document.py --list")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        list_documents()
    else:
        process_document(sys.argv[1])
