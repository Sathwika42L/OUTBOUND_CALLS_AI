"""
Vector Store - Manages embeddings and retrieval using ChromaDB
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Vector store for document embeddings and retrieval"""
    
    def __init__(self, persist_directory: str):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Use sentence-transformers for embeddings
        logger.info("Using local sentence-transformers for embeddings")
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store initialized: knowledge_base")
        logger.info(f"Total documents: {self.collection.count()}")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """
        Add documents to vector store
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts
            ids: List of unique IDs
        """
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def query(
        self,
        query_text: str,
        n_results: int = 3,
        document_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Query vector store for similar documents
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            document_filter: Optional document name filter
            
        Returns:
            List of result dicts with text, metadata, and distance
        """
        # Build where clause for filtering
        where = None
        if document_filter:
            where = {"document": document_filter}
        
        # Query
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        logger.debug(f"Query: '{query_text}' returned {len(formatted_results)} results")
        
        return formatted_results
    
    def list_documents(self) -> List[str]:
        """List all document names in the store"""
        results = self.collection.get()
        if results and results['metadatas']:
            docs = set(meta.get('document', '') for meta in results['metadatas'])
            return list(docs)
        return []
    
    def count(self) -> int:
        """Get total number of chunks in store"""
        return self.collection.count()
