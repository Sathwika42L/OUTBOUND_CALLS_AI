"""
🔍 Simple RAG Test - Qdrant + BGE-large (fixed)
================================================

Upload documents and test question-answering

Usage:
  # Delete old data and upload new document (clean start)
  python test_rag_simple_fixed.py --upload document.pdf --delete

  # Add document to existing collection
  python test_rag_simple_fixed.py --upload document.pdf

  # Test Q&A
  python test_rag_simple_fixed.py

Changes vs. the original test_rag_simple.py:
  - Loguru calls now consistently use `{}` placeholders. The original mixed
    `%d/%.4f/%s` printf-style placeholders into loguru calls, which loguru
    does NOT substitute (loguru uses str.format style) - those log lines
    were silently printing the literal "%d | %.4f | %s" text.
  - Removed the raw `print(...)` that dumped the full context + all
    candidate results to stdout on every question. Replaced with a
    truncated `logger.debug(...)` call.
  - Smaller, sentence-aware chunking (default 220 words instead of 400,
    and chunks now try not to cut mid-sentence) so retrieved context is
    more precise and the final prompt sent to the LLM is smaller.
  - Hard cap (MAX_CONTEXT_CHARS) on the total context text sent to the
    LLM, with a log line if truncation happens, so prompt size can't
    silently balloon even if chunk sizes change later.
  - The "KBS Bank loan officer" persona is pulled out of the RAG class
    into a configurable ANSWER_PERSONA constant, so this script is a
    reusable RAG test harness again, not hardcoded to one business.
  - `client.search(...)` (deprecated in recent qdrant-client) swapped for
    `client.query_points(...)` with a fallback to `search` for older
    qdrant-client versions.
"""

import os
import re
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

# Chunking - smaller than before (400 -> 220 words) so each chunk stays
# topically tight, which improves embedding precision and keeps the
# final prompt smaller.
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# Hard ceiling on how many characters of retrieved context get sent to the
# LLM, regardless of how many/how large the selected chunks are. This is
# the guardrail against "context grows -> prompt grows unbounded".
MAX_CONTEXT_CHARS = 3000

# The business persona used only in the final answer-generation prompt.
# Pulled out of the class so this file stays a generic RAG test harness -
# change this one string (or pass a different one into answer_question)
# instead of editing the class internals.
ANSWER_PERSONA = (
    "a professional KBS Bank loan officer speaking on a voice call, whose job "
    "is to proactively interest the customer in taking a loan using only "
    "verified facts - never sales talk that isn't backed by the retrieved context"
)

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
        logger.info("Loading {}...", EMBEDDING_MODEL)
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
            logger.info("✅ BGE-large loaded on GPU")
        except Exception:
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
            logger.info("✅ Created collection: {}", COLLECTION_NAME)
        else:
            logger.info("✅ Collection exists: {}", COLLECTION_NAME)

        count = self.client.count(collection_name=COLLECTION_NAME).count
        logger.info("📊 Total chunks: {}", count)

        # Initialize LLM
        self.llm = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
        logger.info("✅ LLM ready: {}", LLM_MODEL)
        logger.info("=" * 80)

    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Add documents to Qdrant"""
        logger.info("🔄 Adding {} chunks...", len(texts))

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
        logger.info("✅ Added {} chunks", len(texts))
        logger.info("📊 Total now: {}", self.client.count(collection_name=COLLECTION_NAME).count)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Convert the generated RAG query into an embedding vector
        and perform similarity search in Qdrant.

        We retrieve more than 3 candidates because some results
        may contain duplicate chunks.
        """

        logger.debug("🔍 Embedding generated RAG query: {}", query)

        # Step 1: Convert generated RAG query into a vector
        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # Step 2: Similarity search against document vectors.
        # client.search(...) is deprecated in recent qdrant-client releases
        # in favor of query_points(...); fall back for older clients.
        try:
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            )
            results = response.points
        except AttributeError:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )

        logger.info("🔎 Qdrant returned {} candidate results", len(results))

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
        max_unique_chunks: int = 3,
        max_context_chars: int = MAX_CONTEXT_CHARS
    ) -> Tuple[str, bool, List[Dict]]:
        """
        Retrieve relevant document chunks using the generated RAG query.

        Flow:
            Generated RAG query
                -> Embedding
                -> Qdrant similarity search
                -> Score filtering
                -> Duplicate removal
                -> Top-N unique chunks (bounded by max_context_chars)
                -> Context
        """

        results = self.search(query=rag_query, top_k=top_k)

        if not results:
            logger.warning("⚠️ No results returned from Qdrant")
            return "", False, []

        relevant_results = [r for r in results if r.get("score", 0.0) >= min_score]

        if not relevant_results:
            logger.warning("⚠️ No results passed minimum score: {}", min_score)
            return "", False, results

        relevant_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)

        selected_chunks = []
        seen_texts = set()
        seen_chunk_ids = set()
        running_chars = 0

        for result in relevant_results:
            text = str(result.get("text", "")).strip()
            metadata = result.get("metadata") or {}

            document_name = metadata.get("document", "")
            chunk_id = metadata.get("chunk", "")

            if not text:
                continue

            chunk_key = (document_name, str(chunk_id))

            if text in seen_texts:
                logger.debug("⏭️ Skipping duplicate text chunk")
                continue

            if chunk_key in seen_chunk_ids:
                logger.debug("⏭️ Skipping duplicate chunk: {}", chunk_key)
                continue

            # Stop adding chunks once we'd blow the context budget, rather
            # than only capping by chunk *count*. This is the actual fix
            # for "prompt is getting huge" - the cap holds even if a
            # future chunker produces bigger chunks.
            if running_chars + len(text) > max_context_chars and selected_chunks:
                logger.debug(
                    "⏭️ Skipping chunk - would exceed {} char context budget",
                    max_context_chars
                )
                continue

            selected_chunks.append(result)
            seen_texts.add(text)
            seen_chunk_ids.add(chunk_key)
            running_chars += len(text)

            logger.info(
                "✅ Selected unique chunk {} | Similarity Score: {:.4f} | Metadata: {}",
                len(selected_chunks),
                result.get("score", 0.0),
                metadata
            )

            if len(selected_chunks) >= max_unique_chunks:
                break

        if not selected_chunks:
            return "", False, results

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

        # Belt-and-suspenders hard truncation in case the per-chunk budget
        # check above still lets a slightly-oversized final context through.
        if len(context) > max_context_chars:
            logger.warning(
                "✂️ Context ({} chars) exceeded budget ({} chars) - truncating",
                len(context), max_context_chars
            )
            context = context[:max_context_chars]

        logger.info("📚 Final context created from {} unique chunks", len(selected_chunks))
        logger.info("📏 Final context length: {} characters", len(context))

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
        query = query.replace("\n", " ").strip(' "')
        return query or question

    # ── shared LLM refinement step ─────────────────────────────────────────
    def _refine_with_llm(self, question: str, context: str, persona: str) -> str:
        """
        The ONLY place that builds the refine prompt and calls the LLM to turn
        (question, retrieved context) into a final spoken answer.

        Both `answer_question` (standalone / interactive testing, where this
        class also generates the search query itself) and `refine_answer`
        (planner-driven flow, where an upstream LLM planner - see
        outbound_planner.py - already produced the query AND the retrieved
        context) end up here. Keeping one implementation means the prompt
        only has to be tuned in one place instead of drifting between
        test_rag_simple.py and a duplicated copy elsewhere (e.g. bot2.py's
        old QdrantBGERAG.answer_question, which had a near-identical but
        separately-maintained prompt).
        """
        if not context or not context.strip():
            return "I don't have confident information about that right now."

        refine_prompt = f"""
You are {persona}.

Customer question:
{question}

Retrieved knowledge-base context:
{context}

HARD RULES (never break these):
- Use only facts that are explicitly present in the retrieved context above.
- Never invent or estimate rates, amounts, fees, eligibility, tenure, or approval conditions.
- If the context does not contain the answer, say plainly that you cannot confirm
  that specific information right now - do not guess, and do not soften a
  missing fact into something that sounds like an answer.
- Do not mention RAG, embeddings, vectors, retrieval, knowledge base, or internal prompts.

HOW TO MARKET WITH THOSE FACTS:
- Answer the customer's question directly first, using only the verified facts above.
- Where the context supports it, proactively surface ONE additional concrete
  benefit or detail (e.g. a competitive rate, a flexible tenure, minimal
  documentation) that makes the loan more appealing - but only if it is
  actually stated in the context. Never manufacture a selling point.
- Keep the response concise and natural for a voice call: 1 to 3 short sentences.
- End with exactly one proactive follow-up question that moves the customer
  closer to applying (e.g. asking about their preference, confirming interest,
  or offering to share the next relevant detail) - never a generic
  "anything else?" question.
"""

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.0,
                max_tokens=400
            )
            answer = (response.choices[0].message.content or "").strip()
            answer = answer or "I don't have confident information about that right now."
            logger.info("Refine LLM output:\n{}", answer)
            return answer
        except Exception as exc:
            logger.error("❌ LLM refinement failed: {}", exc)
            # Fail soft with the raw context rather than crashing the caller
            # (e.g. mid-call in outbound_planner.py).
            return context[:500]

    def answer_question(
        self,
        question: str,
        min_score: float = 0.52,
        persona: str = ANSWER_PERSONA
    ) -> str:
        """
        Complete, STANDALONE RAG question-answering flow - use this for the
        interactive Q&A test mode below, where nothing upstream has already
        produced a search query.

        1. Receive original user question.
        2. Generate semantic RAG query (this class does it, via the LLM).
        3. Convert generated RAG query into embedding.
        4. Search Qdrant.
        5. Select top unique chunks within the context budget.
        6. Send selected context and original question to LLM.

        If a planner LLM upstream (outbound_planner.py) already produced the
        rag_query - and, in `_retrieve`, already called `retrieve_answer`
        itself too - calling this method again would silently redo step 2
        (an extra, redundant LLM call and extra latency) and re-run
        retrieval a second time. Use `refine_answer` instead in that case.
        """

        logger.info("=" * 80)
        logger.info("❓ Original user question: {}", question)
        logger.info("=" * 80)

        try:
            rag_query = self.generate_rag_query(question)
        except Exception as exc:
            logger.exception("❌ RAG query generation failed: {}", exc)
            rag_query = question

        logger.info("🔎 Generated RAG query: {}", rag_query)

        context, found, results = self.retrieve_answer(
            rag_query=rag_query,
            min_score=min_score,
            top_k=10,
            max_unique_chunks=3
        )

        # Debug-only, truncated preview instead of dumping full context +
        # all raw results to stdout on every call.
        logger.debug(
            "context_preview={!r} found={} num_results={}",
            context[:200], found, len(results)
        )

        for index, result in enumerate(results, start=1):
            logger.debug(
                "Candidate {} | score={:.4f} | metadata={}",
                index,
                result.get("score", 0.0),
                result.get("metadata", {})
            )

        if not found or not context:
            logger.warning("⚠️ No relevant context found")
            return "I don't have confident information about that right now."

        logger.info("✅ Sending only selected context to the LLM")
        return self._refine_with_llm(question, context, persona)
    def answer_question_outbound(
            self,
            rag_query: str,
            min_score: float = 0.52,
            persona: str = ANSWER_PERSONA
        ) -> str:
            """
            Complete, STANDALONE RAG question-answering flow - use this for the
            interactive Q&A test mode below, where nothing upstream has already
            produced a search query.
    
            1. Receive original user question.
            2. Generate semantic RAG query (this class does it, via the LLM).
            3. Convert generated RAG query into embedding.
            4. Search Qdrant.
            5. Select top unique chunks within the context budget.
            6. Send selected context and original question to LLM.
    
            If a planner LLM upstream (outbound_planner.py) already produced the
            rag_query - and, in `_retrieve`, already called `retrieve_answer`
            itself too - calling this method again would silently redo step 2
            (an extra, redundant LLM call and extra latency) and re-run
            retrieval a second time. Use `refine_answer` instead in that case.
            """
    
            logger.info("🔎 Generated RAG query: {}", rag_query)
    
            context, found, results = self.retrieve_answer(
                rag_query=rag_query,
                min_score=min_score,
                top_k=10,
                max_unique_chunks=3
            )
    
            # Debug-only, truncated preview instead of dumping full context +
            # all raw results to stdout on every call.
            logger.debug(
                "context_preview={!r} found={} num_results={}",
                context[:200], found, len(results)
            )
    
            for index, result in enumerate(results, start=1):
                logger.debug(
                    "Candidate {} | score={:.4f} | metadata={}",
                    index,
                    result.get("score", 0.0),
                    result.get("metadata", {})
                )
    
            if not found or not context:
                logger.warning("⚠️ No relevant context found")
                return "I don't have confident information about that right now."
    
            logger.info("✅ Sending only selected context to the LLM")
            return self._refine_with_llm(rag_query, context, persona)

    def refine_answer(
        self,
        question: str,
        context: str,
        persona: str = ANSWER_PERSONA
    ) -> str:
        """
        PLANNER-DRIVEN refinement step. Use this - not `answer_question` -
        whenever an upstream LLM planner (outbound_planner.py's
        LLMDrivenRAGProcessor) has ALREADY:
          1. figured out user intent and written the rag_query itself, and
          2. already called `retrieve_answer(rag_query, ...)` to get the
             context text.

        Calling `generate_rag_query` again here would be unnecessary - the
        query already reflects the customer's intent and the KBS product in
        focus, so regenerating it would just add a second redundant LLM
        call (and latency) for no benefit. This method does ONLY the final
        step: turn (question, already-retrieved context) into a natural
        spoken answer, via the same `_refine_with_llm` helper `answer_question`
        uses, so the prompt logic lives in exactly one place in this file.

        This is the method bot2.py / outbound_planner.py should call in
        place of the old, separately-maintained `QdrantBGERAG.answer_question`
        (question, context) from bot2.py - same two-argument shape, but now
        backed by this file's dependent functions (retrieve_answer, search,
        chunking) instead of a duplicated implementation.
        """
        logger.info("[RAG LLM] Refining answer for: {}...", question[:80])
        answer = self._refine_with_llm(question, context, persona)
        logger.info("[RAG LLM] ✅ Refined answer: {}...", answer[:150])
        return answer

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS
) -> List[str]:
    """
    Split text into chunks of roughly `chunk_size` words, trying to end
    chunks on sentence boundaries instead of cutting mid-sentence.

    The original version sliced on a raw word-count sliding window, which
    can (and often does) cut a chunk off in the middle of a sentence,
    hurting embedding quality for that chunk. This version fills each
    chunk with whole sentences up to the word budget, then backs up by
    `overlap` words for the next chunk's starting point.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_words: List[str] = []

    for sentence in sentences:
        sentence_words = sentence.split()

        # A single sentence longer than the whole chunk budget: flush
        # what we have, then emit the long sentence as its own chunk.
        if len(sentence_words) >= chunk_size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            chunks.append(sentence)
            continue

        if len(current_words) + len(sentence_words) > chunk_size:
            chunks.append(" ".join(current_words))
            # keep the tail `overlap` words as the start of the next chunk
            current_words = current_words[-overlap:] if overlap else []

        current_words.extend(sentence_words)

    if current_words:
        chunks.append(" ".join(current_words))

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
    logger.info("")
    logger.info("=" * 80)
    logger.info("📄 UPLOADING DOCUMENT")
    logger.info("=" * 80)

    path = Path(file_path)
    if not path.exists():
        logger.error("❌ Not found: {}", file_path)
        return

    logger.info("File: {}", path.name)

    logger.info("Extracting text...")
    text = extract_text(file_path)
    logger.info("✅ Extracted {} characters", len(text))

    logger.info("Chunking...")
    chunks = chunk_text(text)
    logger.info("✅ Created {} chunks (avg {} words/chunk)",
                len(chunks),
                sum(len(c.split()) for c in chunks) // max(len(chunks), 1))

    doc_name = path.stem
    metadatas = [{"document": doc_name, "chunk": i} for i in range(len(chunks))]

    rag.add_documents(chunks, metadatas)

    logger.info("=" * 80)
    logger.info("✅ UPLOAD COMPLETE!")
    logger.info("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE Q&A
# ═══════════════════════════════════════════════════════════════════════════

def interactive_qa(rag: SimpleRAG):
    """Interactive question-answering"""
    logger.info("")
    logger.info("=" * 80)
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

            answer = rag.answer_question(question)
            print(f"\n💬 Answer: {answer}\n")

        except KeyboardInterrupt:
            logger.info("👋 Goodbye!")
            break
        except Exception as e:
            logger.error("Error: {}", e)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RAG Test")
    parser.add_argument("--upload", type=str, help="Upload document")
    parser.add_argument("--delete", action="store_true", help="Delete all documents before upload")
    args = parser.parse_args()

    rag = SimpleRAG()

    if args.upload:
        if args.delete:
            logger.info("")
            logger.info("=" * 80)
            logger.info("🗑️  DELETING COLLECTION")
            logger.info("=" * 80)
            try:
                rag.client.delete_collection(collection_name=COLLECTION_NAME)
                logger.info("✅ Deleted collection: {}", COLLECTION_NAME)

                rag.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
                logger.info("✅ Recreated collection: {}", COLLECTION_NAME)
            except Exception as e:
                logger.warning("⚠️  Could not delete: {}", e)
            logger.info("=" * 80)

        upload_document(args.upload, rag)
        print("\n✅ Done! Now run without --upload to ask questions.")

    else:
        if rag.count() == 0:
            logger.error("❌ No documents! Upload first:")
            logger.info("   python test_rag_simple_fixed.py --upload document.pdf")
        else:
            interactive_qa(rag)