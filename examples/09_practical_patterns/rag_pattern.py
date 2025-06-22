"""RAG Pattern - Retrieval-Augmented Generation system.

Complete RAG implementation showing document chunking, similarity search,
and context-aware generation using composable operators.

Example:
    >>> rag = RAGPipeline(documents)
    >>> answer = rag(query="What is deep learning?")
    >>> print(answer["answer"])
"""

import sys
from pathlib import Path
from typing import List, Dict
import math

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import operators


def main():
    """Example demonstrating the simplified XCS architecture."""
    """Build a complete RAG system with Ember operators."""
    print_section_header("RAG Pattern Implementation")
    
    # Sample documents (in practice, load from files/DB)
    documents = [
        {
            "id": "doc1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data, identify patterns, and make decisions with minimal human intervention."
        },
        {
            "id": "doc2", 
            "title": "Types of Machine Learning",
            "content": "There are three main types of machine learning: supervised learning uses labeled data to train models, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through interaction with an environment using rewards and penalties."
        },
        {
            "id": "doc3",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is a subset of machine learning based on artificial neural networks. These networks consist of multiple layers that progressively extract higher-level features from raw input. Deep learning has revolutionized fields like computer vision and natural language processing."
        },
        {
            "id": "doc4",
            "title": "Machine Learning Applications",
            "content": "Machine learning powers many modern applications including recommendation systems, fraud detection, autonomous vehicles, medical diagnosis, and natural language processing. Its ability to find patterns in large datasets makes it invaluable for data-driven decision making."
        }
    ]
    
    # Part 1: Document Chunking
    print("🔍 Part 1: Document Processing\n")
    
    class DocumentChunker(operators.Operator):
        """Chunks documents into smaller pieces for indexing."""
        
        specification = operators.Specification()
        
        def __init__(self, *, chunk_size: int = 100, overlap: int = 20):
            self.chunk_size = chunk_size
            self.overlap = overlap
        
        def forward(self, *, inputs):
            documents = inputs.get("documents", [])
            chunks = []
            
            for doc in documents:
                content = doc["content"]
                words = content.split()
                
                # Create overlapping chunks
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk_words = words[i:i + self.chunk_size]
                    if len(chunk_words) > 10:  # Minimum chunk size
                        chunks.append({
                            "doc_id": doc["id"],
                            "title": doc["title"],
                            "chunk_id": f"{doc['id']}_chunk_{i // (self.chunk_size - self.overlap)}",
                            "text": " ".join(chunk_words),
                            "word_count": len(chunk_words)
                        })
            
            return {"chunks": chunks, "total_chunks": len(chunks)}
    
    # Chunk the documents
    chunker = DocumentChunker(chunk_size=50, overlap=10)
    chunked = chunker(documents=documents)
    
    print(f"Created {chunked['total_chunks']} chunks from {len(documents)} documents")
    print_example_output("Sample chunk", chunked['chunks'][0]['text'][:100] + "...")
    
    # Part 2: Simple Embedding and Indexing
    print("\n" + "="*50)
    print("🗂️ Part 2: Indexing with Simple Embeddings")
    print("="*50 + "\n")
    
    class SimpleEmbedder(operators.Operator):
        """Creates simple embeddings for demonstration."""
        
        specification = operators.Specification()
        
        def forward(self, *, inputs):
            chunks = inputs.get("chunks", [])
            
            # Simple embedding: word frequency features
            # In practice, use real embeddings (OpenAI, Sentence Transformers, etc.)
            embeddings = []
            
            for chunk in chunks:
                text_lower = chunk["text"].lower()
                words = text_lower.split()
                
                # Create simple feature vector
                features = {
                    "length": len(words),
                    "ml_terms": sum(1 for w in words if w in ["machine", "learning", "ml", "ai"]),
                    "deep_terms": sum(1 for w in words if w in ["deep", "neural", "network"]),
                    "data_terms": sum(1 for w in words if w in ["data", "dataset", "pattern"]),
                    "algo_terms": sum(1 for w in words if w in ["algorithm", "model", "train"])
                }
                
                # Convert to vector
                embedding = [features[k] for k in sorted(features.keys())]
                
                embeddings.append({
                    "chunk_id": chunk["chunk_id"],
                    "embedding": embedding,
                    "chunk": chunk
                })
            
            return {"embeddings": embeddings, "dimension": len(embedding)}
    
    embedder = SimpleEmbedder()
    indexed = embedder(chunks=chunked["chunks"])
    
    print(f"Created {len(indexed['embeddings'])} embeddings")
    print_example_output("Embedding dimension", indexed['dimension'])
    
    # Part 3: Retrieval
    print("\n" + "="*50)
    print("🔎 Part 3: Semantic Retrieval")
    print("="*50 + "\n")
    
    class SemanticRetriever(operators.Operator):
        """Retrieves relevant chunks based on query."""
        
        specification = operators.Specification()
        
        def __init__(self, *, top_k: int = 3):
            self.top_k = top_k
            self.embedder = SimpleEmbedder()
        
        def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        
        def forward(self, *, inputs):
            query = inputs.get("query", "")
            embeddings = inputs.get("embeddings", [])
            
            # Embed the query
            query_embedded = self.embedder(chunks=[{"text": query, "chunk_id": "query"}])
            query_vec = query_embedded["embeddings"][0]["embedding"]
            
            # Calculate similarities
            similarities = []
            for emb in embeddings:
                similarity = self.cosine_similarity(query_vec, emb["embedding"])
                similarities.append({
                    "chunk": emb["chunk"],
                    "similarity": similarity
                })
            
            # Sort by similarity and get top-k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            top_chunks = similarities[:self.top_k]
            
            return {
                "query": query,
                "retrieved_chunks": top_chunks,
                "num_retrieved": len(top_chunks)
            }
    
    # Test retrieval
    retriever = SemanticRetriever(top_k=2)
    query = "What are the different types of machine learning?"
    
    retrieved = retriever(query=query, embeddings=indexed["embeddings"])
    
    print(f"Query: {query}\n")
    print("Retrieved chunks:")
    for i, result in enumerate(retrieved["retrieved_chunks"], 1):
        print(f"\n{i}. From: {result['chunk']['title']} (similarity: {result['similarity']:.3f})")
        print(f"   {result['chunk']['text'][:150]}...")
    
    # Part 4: Generation with Context
    print("\n" + "="*50)
    print("💡 Part 4: Context-Aware Generation")
    print("="*50 + "\n")
    
    class ContextualGenerator(operators.Operator):
        """Generates answers using retrieved context."""
        
        specification = operators.Specification()
        
        def forward(self, *, inputs):
            query = inputs.get("query", "")
            retrieved_chunks = inputs.get("retrieved_chunks", [])
            
            # Build context from retrieved chunks
            context_parts = []
            for chunk_info in retrieved_chunks:
                chunk = chunk_info["chunk"]
                context_parts.append(f"[{chunk['title']}]: {chunk['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Simulate generation (in practice, use LLM)
            # For demo, we'll create a response based on the context
            if "types" in query.lower() and "machine learning" in query.lower():
                answer = "Based on the provided context, there are three main types of machine learning: supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through environment interaction with rewards)."
            else:
                # Generic response construction
                answer = f"Based on the context about {retrieved_chunks[0]['chunk']['title']}, {query.lower().replace('?', '.')}"
            
            return {
                "query": query,
                "answer": answer,
                "context_used": context,
                "num_sources": len(retrieved_chunks)
            }
    
    generator = ContextualGenerator()
    response = generator(
        query=query,
        retrieved_chunks=retrieved["retrieved_chunks"]
    )
    
    print("Generated Answer:")
    print(f"\n{response['answer']}")
    print(f"\nBased on {response['num_sources']} sources")
    
    # Part 5: Complete RAG Pipeline
    print("\n" + "="*50)
    print("🚀 Part 5: Complete RAG Pipeline")
    print("="*50 + "\n")
    
    class RAGPipeline(operators.Operator):
        """Complete RAG system in a single operator."""
        
        specification = operators.Specification()
        
        def __init__(self, *, documents: List[Dict], chunk_size: int = 50, top_k: int = 3):
            self.chunker = DocumentChunker(chunk_size=chunk_size)
            self.embedder = SimpleEmbedder()
            self.retriever = SemanticRetriever(top_k=top_k)
            self.generator = ContextualGenerator()
            
            # Pre-index documents
            chunks = self.chunker(documents=documents)
            self.index = self.embedder(chunks=chunks["chunks"])
        
        def forward(self, *, inputs):
            query = inputs.get("query", "")
            
            # Retrieve relevant chunks
            retrieved = self.retriever(
                query=query,
                embeddings=self.index["embeddings"]
            )
            
            # Generate answer
            generated = self.generator(
                query=query,
                retrieved_chunks=retrieved["retrieved_chunks"]
            )
            
            return {
                "query": query,
                "answer": generated["answer"],
                "sources": [
                    {
                        "title": chunk["chunk"]["title"],
                        "relevance": chunk["similarity"]
                    }
                    for chunk in retrieved["retrieved_chunks"]
                ],
                "metadata": {
                    "chunks_searched": len(self.index["embeddings"]),
                    "chunks_retrieved": retrieved["num_retrieved"]
                }
            }
    
    # Create and test the complete pipeline
    rag = RAGPipeline(documents=documents, chunk_size=50, top_k=2)
    
    test_queries = [
        "What are the different types of machine learning?",
        "How does deep learning work?",
        "What are some applications of ML?"
    ]
    
    print("RAG Pipeline Results:\n")
    for q in test_queries:
        result = rag(query=q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(s['title'] for s in result['sources'])}")
        print(f"Metadata: Searched {result['metadata']['chunks_searched']} chunks\n")
    
    print("="*50)
    print("✅ Key Takeaways")
    print("="*50)
    print("\n1. RAG separates retrieval from generation")
    print("2. Operators handle each stage independently")
    print("3. Clean composition creates powerful systems")
    print("4. Each component is testable and reusable")
    print("5. Real implementations would use:")
    print("   - Vector databases (Pinecone, Weaviate, etc.)")
    print("   - Real embeddings (OpenAI, Sentence Transformers)")
    print("   - LLMs for generation")
    
    print("\nNext: Explore more patterns in other examples!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())