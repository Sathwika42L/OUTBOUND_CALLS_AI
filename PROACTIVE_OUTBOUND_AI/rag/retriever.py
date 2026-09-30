"""
RAG Retriever - Query knowledge base and generate natural answers
"""

from typing import Tuple, Optional
from loguru import logger
from rag.vector_store import VectorStore


class RAGRetriever:
    """Retrieve relevant information and generate answers"""
    
    def __init__(self, vector_store: VectorStore, fallback_message: str = ""):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: Vector store instance
            fallback_message: Message when information not found
        """
        self.vector_store = vector_store
        self.fallback_message = fallback_message or (
            "I don't have that specific information right now. "
            "Would you like me to arrange for a specialist to provide those details?"
        )
    
    def retrieve_and_answer(
        self, 
        question: str, 
        document_filter: Optional[str] = None,
        n_results: int = 3,
        confidence_threshold: float = 1.5
    ) -> Tuple[str, bool, list]:
        """
        Retrieve relevant chunks and generate answer
        
        Args:
            question: Customer's question
            document_filter: Optional filter by document name
            n_results: Number of chunks to retrieve
            confidence_threshold: Maximum distance for relevant results (lower = more strict)
            
        Returns:
            (answer, found_relevant_info, retrieved_chunks)
        """
        
        logger.info(f"[RAG] Question: {question}")
        
        # Query vector store
        results = self.vector_store.query(
            query_text=question,
            n_results=n_results,
            document_filter=document_filter
        )
        
        if not results:
            logger.warning("[RAG] No results from vector store")
            return self.fallback_message, False, []
        
        # Check relevance (lower distance = more similar)
        top_result = results[0]
        distance = top_result.get('distance', 999)
        
        logger.info(f"[RAG] Top result distance: {distance:.3f}")
        
        # If distance too high, not relevant
        if distance > confidence_threshold:
            logger.warning(f"[RAG] Distance {distance:.3f} > threshold {confidence_threshold}")
            return self.fallback_message, False, results
        
        # Combine top results
        context_chunks = [r['text'] for r in results[:2]]
        combined_context = "\n\n".join(context_chunks)
        
        logger.info(f"[RAG] Retrieved {len(results)} chunks")
        logger.debug(f"[RAG] Context: {combined_context[:200]}...")
        
        # Extract relevant answer from context
        answer = self._extract_relevant_answer(question, combined_context)
        
        return answer, True, results
    
    def _extract_relevant_answer(self, question: str, context: str) -> str:
        """
        Extract the most relevant part of context for the question
        
        This is a simple extraction. In production, use an LLM for synthesis.
        
        Args:
            question: Customer's question
            context: Retrieved context
            
        Returns:
            Extracted answer
        """
        question_lower = question.lower()
        lines = context.split('\n')
        relevant_lines = []
        
        # Detect question type and extract relevant section
        if any(word in question_lower for word in ['loan', 'provide', 'offer', 'available', 'type']):
            # Looking for loan types/purposes
            in_section = False
            for i, line in enumerate(lines):
                if 'LOAN PURPOSE' in line.upper() or 'can be used for' in line.lower():
                    in_section = True
                    continue
                elif in_section and line.strip():
                    if line.strip().startswith('-') or 'including' in line.lower():
                        relevant_lines.append(line.strip())
                    elif line.strip().isupper() and len(line.strip()) > 10:
                        break
        
        elif any(word in question_lower for word in ['eligibility', 'eligible', 'qualify', 'criteria']):
            # Extract eligibility criteria
            in_section = False
            for i, line in enumerate(lines):
                if 'ELIGIBILITY CRITERIA' in line.upper():
                    in_section = True
                    continue
                elif in_section and line.strip():
                    if line.strip().startswith('-'):
                        relevant_lines.append(line.strip())
                    elif 'REQUIRED DOCUMENTS' in line.upper():
                        break
        
        elif any(word in question_lower for word in ['interest', 'rate', 'percentage']):
            # Extract interest rates
            in_section = False
            for i, line in enumerate(lines):
                if 'INTEREST RATES' in line.upper():
                    in_section = True
                    continue
                elif in_section and line.strip():
                    relevant_lines.append(line.strip())
                    if 'LOAN AMOUNTS' in line.upper():
                        break
        
        elif any(word in question_lower for word in ['document', 'documents', 'paperwork', 'need']):
            # Extract documents
            in_section = False
            for i, line in enumerate(lines):
                if 'REQUIRED DOCUMENTS' in line.upper():
                    in_section = True
                    continue
                elif in_section and line.strip():
                    if line.strip().startswith('-') or 'Copy of' in line or 'Proof of' in line:
                        relevant_lines.append(line.strip())
                    elif line.strip().isupper() and len(line.strip()) > 10:
                        break
        
        elif any(word in question_lower for word in ['amount', 'how much', 'range']):
            # Extract loan amounts
            in_section = False
            for i, line in enumerate(lines):
                if 'LOAN AMOUNTS' in line.upper():
                    in_section = True
                    continue
                elif in_section and line.strip():
                    relevant_lines.append(line.strip())
                    if 'ELIGIBILITY' in line.upper():
                        break
        
        elif any(word in question_lower for word in ['repay', 'payment', 'term', 'monthly']):
            # Extract repayment info
            in_section = False
            for i, line in enumerate(lines):
                if 'REPAYMENT' in line.upper():
                    in_section = True
                    continue
                elif in_section and line.strip():
                    relevant_lines.append(line.strip())
                    if 'FEES' in line.upper():
                        break
        
        # If found relevant lines, use them
        if relevant_lines:
            answer = ' '.join(relevant_lines[:5])  # Max 5 lines
            # Clean up
            answer = answer.replace('PERSONAL LOAN INFORMATION - ABC BANK', '').strip()
            answer = answer.replace(':', '', 1).strip()
            return answer
        
        # Fallback: return first 300 chars
        return context[:300].strip() + "..."
    
    def check_document_availability(self, document_name: str) -> bool:
        """Check if a document exists in vector store"""
        documents = self.vector_store.list_documents()
        return document_name in documents
    
    def get_available_documents(self) -> list:
        """Get list of available documents"""
        return self.vector_store.list_documents()
