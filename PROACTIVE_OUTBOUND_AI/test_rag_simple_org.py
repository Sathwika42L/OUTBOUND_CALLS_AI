"""
🔍 Simple RAG Test - Qdrant + BGE-large
========================================

Upload documents and test question-answering

Usage:
  # Delete old data and upload new document (clean start)
  python test_rag_simple.py --upload document.pdf --delete
  
  # Add document to existing collection
  python test_rag_simple.py --upload document.pdf
  
  # Test Q&A
  python test_rag_simple.py
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger

# Qdrant & BGE
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Document processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# OpenAI for natural answer generation
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

QDRANT_PATH = "./knowledge_base/qdrant_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # BGE-large
EMBEDDING_DIM = 1024

# LLM for natural answers (using your Ollama)
LLM_BASE_URL = "http://202.164.134.176:11434/v1"
LLM_MODEL = "qwen2.5:14b"

# ═══════════════════════════════════════════════════════════════════════════
# QDRANT + BGE RAG SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class SimpleRAG:
    """Simple RAG: Qdrant + BGE-large + LLM"""
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("🔍 Initializing Simple RAG System")
        logger.info("=" * 80)
        
        # Load BGE-large
        logger.info(f"Loading {EMBEDDING_MODEL}...")
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
            logger.info("✅ BGE-large loaded on GPU")
        except:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            logger.info("✅ BGE-large loaded on CPU")
        
        # Initialize Qdrant
        os.makedirs(QDRANT_PATH, exist_ok=True)
        self.client = QdrantClient(path=QDRANT_PATH)
        
        # Create collection if needed
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            logger.info(f"✅ Created collection: {COLLECTION_NAME}")
        else:
            logger.info(f"✅ Collection exists: {COLLECTION_NAME}")
        
        count = self.client.count(collection_name=COLLECTION_NAME).count
        logger.info(f"📊 Total chunks: {count}")
        
        # Initialize LLM
        self.llm = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
        logger.info(f"✅ LLM ready: {LLM_MODEL}")
        logger.info("=" * 80)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Add documents to Qdrant"""
        logger.info(f"🔄 Adding {len(texts)} chunks...")
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        
        # Create points
        current_count = self.client.count(collection_name=COLLECTION_NAME).count
        points = []
        for idx, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            points.append(PointStruct(
                id=current_count + idx,
                vector=embedding.tolist(),
                payload={"text": text, "metadata": metadata}
            ))
        
        # Upload
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"✅ Added {len(texts)} chunks")
        logger.info(f"📊 Total now: {self.client.count(collection_name=COLLECTION_NAME).count}")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Convert the generated RAG query into an embedding vector
        and perform similarity search in Qdrant.

        We retrieve more than 3 candidates because some results
        may contain duplicate chunks.
        """

        logger.info("🔍 Embedding generated RAG query:")
        logger.info(query)

        # Step 1: Convert generated RAG query into a vector
        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        logger.info("✅ Generated query converted into embedding")

        # Step 2: Similarity search against document vectors
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        logger.info(
            f"🔎 Qdrant returned {len(results)} candidate results"
        )

        formatted_results = []

        for result in results:
            payload = result.payload or {}

            formatted_results.append({
                "text": str(payload.get("text", "")).strip(),
                "score": float(result.score),
                "metadata": payload.get("metadata", {})
            })

        return formatted_results
    
    def retrieve_answer(
        self,
        rag_query: str,
        min_score: float = 0.52,
        top_k: int = 10,
        max_unique_chunks: int = 3
    ) -> Tuple[str, bool, List[Dict]]:
        """
        Retrieve relevant document chunks using the generated RAG query.

        Flow:
            Generated RAG query
                -> Embedding
                -> Qdrant similarity search
                -> Score filtering
                -> Duplicate removal
                -> Top 3 unique chunks
                -> Context
        """

        # Search using the generated RAG query
        results = self.search(
            query=rag_query,
            top_k=top_k
        )

        if not results:
            logger.warning("⚠️ No results returned from Qdrant")
            return "", False, []

        # Filter results by similarity score
        relevant_results = [
            result
            for result in results
            if result.get("score", 0.0) >= min_score
        ]

        if not relevant_results:
            logger.warning(
                f"⚠️ No results passed minimum score: {min_score}"
            )

            return "", False, results

        # Sort by highest similarity score
        relevant_results.sort(
            key=lambda item: item.get("score", 0.0),
            reverse=True
        )

        selected_chunks = []
        seen_texts = set()
        seen_chunk_ids = set()

        for result in relevant_results:
            text = str(result.get("text", "")).strip()
            metadata = result.get("metadata") or {}

            document_name = metadata.get("document", "")
            chunk_id = metadata.get("chunk", "")

            # Ignore empty chunks
            if not text:
                continue

            # Unique identity based on document and chunk number
            chunk_key = (
                document_name,
                str(chunk_id)
            )

            # Skip exact duplicate text
            if text in seen_texts:
                logger.info("⏭️ Skipping duplicate text chunk")
                continue

            # Skip duplicate document/chunk identity
            if chunk_key in seen_chunk_ids:
                logger.info(
                    f"⏭️ Skipping duplicate chunk: {chunk_key}"
                )
                continue

            selected_chunks.append(result)
            seen_texts.add(text)
            seen_chunk_ids.add(chunk_key)

            logger.info(
                "✅ Selected unique chunk %d | Similarity Score: %.4f | Metadata: %s",
                len(selected_chunks),
                result.get("score", 0.0),
                metadata
            )

            # Stop after 3 unique chunks
            if len(selected_chunks) >= max_unique_chunks:
                break

        if not selected_chunks:
            return "", False, results

        # Build context using only selected unique chunks
        context_parts = []

        for index, result in enumerate(selected_chunks, start=1):
            metadata = result.get("metadata") or {}

            context_parts.append(
                f"[Retrieved Source {index}]\n"
                f"Document: {metadata.get('document', 'Unknown')}\n"
                f"Chunk: {metadata.get('chunk', 'Unknown')}\n"
                f"{result['text']}"
            )

        context = "\n\n".join(context_parts)

        logger.info(
            "📚 Final context created from %d unique chunks",
            len(selected_chunks)
        )

        logger.info(
            "📏 Final context length: %d characters",
            len(context)
        )

        return context, True, selected_chunks

    def generate_rag_query(self, question: str) -> str:
        """Use the large local model to convert the user question into a
        contextual, product-aware semantic search query."""
        prompt = f"""You are a retrieval-query specialist for KBS Bank loan documents.

Conversation question: {question}

Create one precise semantic search query. Include the relevant loan product
if it is known. Expand vague references such as 'it', 'that loan', 'rate',
'documents', or 'how much' into their likely meaning using the question.
Do not answer the question. Return only the query, without quotes or labels."""
        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        query = (response.choices[0].message.content or question).strip()
        query = query.replace("\n", " ").strip(' \"')
        return query or question

    def answer_question(
        self,
        question: str,
        min_score: float = 0.52
    ) -> str:
        """
        Complete RAG question-answering flow.

        1. Receive original user question.
        2. Generate semantic RAG query.
        3. Convert generated RAG query into embedding.
        4. Search Qdrant.
        5. Select top 3 unique chunks.
        6. Send selected context and original question to LLM.
        """

        logger.info("=" * 80)
        logger.info("❓ Original user question: {}", question)
        logger.info("=" * 80)

        # Step 1: Generate RAG query using the LLM
        try:
            rag_query = self.generate_rag_query(question)

        except Exception as exc:
            logger.exception(
                "❌ RAG query generation failed: {}",
                exc
            )

            # Fallback to original question
            rag_query = question

        logger.info("🔎 Generated RAG query: {}", rag_query)

        # Step 2:
        # Generated RAG query -> embedding -> Qdrant search
        context, found, results = self.retrieve_answer(
            rag_query=rag_query,
            min_score=min_score,
            top_k=10,
            max_unique_chunks=3
        )
        print("context:",context,"-found:",found,"-results:",results)

        # Log retrieval results
        for index, result in enumerate(results, start=1):
            logger.info(
                "Candidate %d | score=%.4f | metadata=%s",
                index,
                result.get("score", 0.0),
                result.get("metadata", {})
            )

        if not found or not context:
            logger.warning("⚠️ No relevant context found")

            return (
                "I don't have confident information about that "
                "right now."
            )

        logger.info(
            "✅ Sending only selected context to the LLM"
        )

        # Step 3: Send original question + selected context to LLM
        refine_prompt = f"""
    You are a professional KBS Bank loan officer speaking on a voice call.

    Customer question:
    {question}

    Retrieved knowledge-base context:
    {context}

    Instructions:
    - Answer using only facts supported by the retrieved context.
    - Do not invent rates, amounts, fees, eligibility, tenure, or approval conditions.
    - Answer the customer's original question directly.
    - Keep the response concise and natural for a voice call.
    - Use 1 to 3 short sentences when possible.
    - If the context does not contain enough information, say that
    you cannot confirm the specific information right now.
    - Do not mention RAG, embeddings, vectors, retrieval, or internal prompts.
    - Ask only one useful follow-up question when appropriate.
    """

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": refine_prompt
                }
            ],
            temperature=0.2,
            max_tokens=220
        )

        answer = (
            response.choices[0].message.content or ""
        ).strip()

        return answer or (
            "I don't have confident information about that "
            "right now."
        )

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def extract_text(file_path: str) -> str:
    """Extract text from file"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif suffix == '.pdf':
        if not PyPDF2:
            raise ImportError("Install PyPDF2: pip install PyPDF2")
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    
    elif suffix == '.docx':
        if not DocxDocument:
            raise ImportError("Install python-docx: pip install python-docx")
        doc = DocxDocument(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    
    else:
        raise ValueError(f"Unsupported: {suffix}")


def upload_document(file_path: str, rag: SimpleRAG):
    """Upload document"""
    logger.info("\n" + "=" * 80)
    logger.info("📄 UPLOADING DOCUMENT")
    logger.info("=" * 80)
    
    path = Path(file_path)
    if not path.exists():
        logger.error(f"❌ Not found: {file_path}")
        return
    
    logger.info(f"File: {path.name}")
    
    # Extract
    logger.info("Extracting text...")
    text = extract_text(file_path)
    logger.info(f"✅ Extracted {len(text)} characters")
    
    # Chunk
    logger.info("Chunking...")
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    logger.info(f"✅ Created {len(chunks)} chunks")
    
    # Metadata
    doc_name = path.stem
    metadatas = [{"document": doc_name, "chunk": i} for i in range(len(chunks))]
    
    # Add to RAG
    rag.add_documents(chunks, metadatas)
    
    logger.info("=" * 80)
    logger.info("✅ UPLOAD COMPLETE!")
    logger.info("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE Q&A
# ═══════════════════════════════════════════════════════════════════════════

def interactive_qa(rag: SimpleRAG):
    """Interactive question-answering"""
    logger.info("\n" + "=" * 80)
    logger.info("💬 INTERACTIVE Q&A MODE")
    logger.info("=" * 80)
    logger.info("Ask questions! Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("👋 Goodbye!")
                break
            
            # Get answer
            answer = rag.answer_question(question)
            print(f"\n💬 Answer: {answer}\n")
        
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RAG Test")
    parser.add_argument("--upload", type=str, help="Upload document")
    parser.add_argument("--delete", action="store_true", help="Delete all documents before upload")
    args = parser.parse_args()
    
    # Initialize RAG
    rag = SimpleRAG()
    
    if args.upload:
        # Delete collection if requested
        if args.delete:
            logger.info("\n" + "=" * 80)
            logger.info("🗑️  DELETING COLLECTION")
            logger.info("=" * 80)
            try:
                rag.client.delete_collection(collection_name=COLLECTION_NAME)
                logger.info(f"✅ Deleted collection: {COLLECTION_NAME}")
                
                # Recreate collection
                rag.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
                logger.info(f"✅ Recreated collection: {COLLECTION_NAME}")
            except Exception as e:
                logger.warning(f"⚠️  Could not delete: {e}")
            logger.info("=" * 80)
        
        # Upload mode
        upload_document(args.upload, rag)
        print("\n✅ Done! Now run without --upload to ask questions.")
    
    else:
        # Interactive Q&A mode
        if rag.count() == 0:
            logger.error("❌ No documents! Upload first:")
            logger.info("   python test_rag_simple.py --upload document.pdf")
        else:
            interactive_qa(rag)
