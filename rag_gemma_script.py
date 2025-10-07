#!/usr/bin/env python3
import sys
import json
import os
import re
from typing import List, Dict, Any

try:
    # Set environment - UPDATE TOKEN HERE
    
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Disable MPS (Mac working setup)
    if torch.backends.mps.is_available():
        torch.backends.mps.is_built = lambda: False

    # Load Gemma model - UPDATE YOUR PATH HERE
    model_id = "/Users/nethrashri/Desktop/Testing_local_gemma_workings/gemma3-270m-i"

    print(f"Loading Gemma from: {model_id}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=torch.float32
    )

    # Load sentence transformer for embeddings from LOCAL PATH
    embedding_model_path = "/Users/nethrashri/Desktop/Testing_local_gemma_workings/all-MiniLM-L6-v2"
    print(f"Loading embedding model from: {embedding_model_path}", file=sys.stderr)
    embedding_model = SentenceTransformer(embedding_model_path)
    
    print("Models loaded successfully!", file=sys.stderr)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_embeddings(chunks: List[str]) -> np.ndarray:
        """Create embeddings for document chunks"""
        return embedding_model.encode(chunks)
    
    def is_topic_relevant(query: str) -> bool:
        """Check if query is related to WiFi, networking, or home automation"""
        wifi_keywords = [
            'wifi', 'wi-fi', 'wireless', 'router', 'network', 'internet', 'connection', 'signal',
            'bandwidth', 'speed', 'latency', 'modem', 'ethernet', 'ip', 'dns', 'firewall',
            'security', 'password', 'wpa', 'encryption', 'access point', 'mesh', 'range',
            'interference', 'channel', 'frequency', 'ghz', 'mbps', 'gbps', 'streaming',
            'gaming', 'qos', 'iot', 'smart home', 'automation', 'device', 'connect',
            'disconnect', 'dropping', 'slow', 'lag', 'buffering', 'youtube', 'netflix',
            'streaming', 'online', 'offline', 'troubleshoot', 'fix', 'problem', 'issue',
            'setup', 'configure', 'install', 'update', 'upgrade', 'vpn', 'vlan',
            'port', 'protocol', 'tcp', 'udp', 'ping', 'traceroute', 'wan', 'lan',
            'access', 'website', 'loading', 'timeout', 'error', 'failed', 'broken'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in wifi_keywords)
    
    def find_relevant_context(query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> tuple:
        """Find most relevant document chunks for query and return similarity score"""
        query_embedding = embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        max_similarity = similarities[top_indices[0]] if len(top_indices) > 0 else 0
        
        # Lower threshold - more inclusive
        relevant_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.1]
        
        return relevant_chunks, max_similarity
    
    # Get command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    
    if mode == "process_document":
        # Process and store document
        document_text = sys.argv[2] if len(sys.argv) > 2 else ""
        
        print("Processing document for RAG...", file=sys.stderr)
        chunks = chunk_document(document_text)
        embeddings = create_embeddings(chunks)
        
        result = {
            "success": True,
            "chunks": chunks,
            "embeddings": embeddings.tolist(),
            "chunk_count": len(chunks)
        }
        print(json.dumps(result))
        
    elif mode == "smart_generate":
        # Smart generation: RAG + fallback to general AI
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        chunks_json = sys.argv[3] if len(sys.argv) > 3 else "[]"
        embeddings_json = sys.argv[4] if len(sys.argv) > 4 else "[]"
        max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 250
        
        # Check if topic is relevant
        if not is_topic_relevant(query):
            response = "I can only answer questions related to WiFi management, network security, home automation, and network troubleshooting. Your question appears to be outside these topics."
            context_used = 0
            mode_used = "out_of_scope"
        else:
            chunks = json.loads(chunks_json)
            embeddings = np.array(json.loads(embeddings_json)) if chunks else np.array([])
            
            # Try to find relevant context if we have documents
            if len(chunks) > 0 and len(embeddings) > 0:
                relevant_context, max_similarity = find_relevant_context(query, chunks, embeddings)
            else:
                relevant_context = []
                max_similarity = 0
            
            if relevant_context and max_similarity > 0.15:
                # Use RAG with document context
                context_text = "\n\n".join(relevant_context)
                
                prompt = f"""You are a WiFi and network expert. Answer the user's question based ONLY on the following technical documentation.

Technical Documentation:
{context_text}

User Question: {query}

Answer the question directly using information from the documentation above. Be specific and reference the relevant details:"""

                print(f"Using RAG mode with {len(relevant_context)} chunks (similarity: {max_similarity:.3f})", file=sys.stderr)
                context_used = len(relevant_context)
                mode_used = "rag"
                
            else:
                # Fall back to general networking knowledge
                prompt = f"""You are a WiFi and network troubleshooting expert. The user has a networking issue or question.

User's Issue/Question: {query}

Provide specific troubleshooting steps and recommendations for this WiFi/networking issue. Include practical solutions like checking router settings, restarting devices, checking cables, examining interference, or configuration steps:"""

                print(f"Using general AI mode (low similarity: {max_similarity:.3f})", file=sys.stderr)
                context_used = 0
                mode_used = "general"

            messages = [{"role": "user", "content": prompt}]
            
            outputs = pipe(
                messages,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.3,
                disable_compile=True
            )
            
            result = outputs[0]['generated_text']
            
            if isinstance(result, list) and len(result) > 1:
                assistant_msg = result[-1]
                if isinstance(assistant_msg, dict) and assistant_msg.get('role') == 'assistant':
                    response = assistant_msg.get('content', '')
                else:
                    response = str(assistant_msg)
            else:
                response = str(result)
        
        output = {
            "success": True,
            "response": response,
            "context_used": context_used,
            "mode": mode_used,
            "topic_relevant": is_topic_relevant(query),
            "model": "google/gemma-3-270m-it"
        }
        print(json.dumps(output))
        
    elif mode == "rag_generate":
        # Original RAG mode (backward compatibility)
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        chunks_json = sys.argv[3] if len(sys.argv) > 3 else "[]"
        embeddings_json = sys.argv[4] if len(sys.argv) > 4 else "[]"
        max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 250
        
        chunks = json.loads(chunks_json)
        embeddings = np.array(json.loads(embeddings_json))
        
        # Find relevant context
        relevant_context, max_similarity = find_relevant_context(query, chunks, embeddings)
        
        if not relevant_context:
            response = "I can only answer questions related to the provided WiFi management, network security, and home automation documentation. Your question appears to be outside the scope of the reference material."
        else:
            context_text = "\n\n".join(relevant_context)
            
            prompt = f"""Based ONLY on the following technical documentation about WiFi management, network security, and home automation, answer the user's question. If the question cannot be answered using the provided documentation, respond that it's outside the scope of the reference material.

Documentation Context:
{context_text}

User Question: {query}

Answer based solely on the documentation above:"""

            messages = [{"role": "user", "content": prompt}]
            
            print(f"Generating RAG response...", file=sys.stderr)
            
            outputs = pipe(
                messages,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.3,
                disable_compile=True
            )
            
            result = outputs[0]['generated_text']
            
            if isinstance(result, list) and len(result) > 1:
                assistant_msg = result[-1]
                if isinstance(assistant_msg, dict) and assistant_msg.get('role') == 'assistant':
                    response = assistant_msg.get('content', '')
                else:
                    response = str(assistant_msg)
            else:
                response = str(result)
        
        output = {
            "success": True,
            "response": response,
            "context_used": len(relevant_context),
            "model": "google/gemma-3-270m-it"
        }
        print(json.dumps(output))
        
    else:
        # Regular generation (backward compatibility)
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
        max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        
        messages = [{"role": "user", "content": prompt}]
        
        print(f"Generating regular response...", file=sys.stderr)
        
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            disable_compile=True
        )
        
        result = outputs[0]['generated_text']
        
        if isinstance(result, list) and len(result) > 1:
            assistant_msg = result[-1]
            if isinstance(assistant_msg, dict) and assistant_msg.get('role') == 'assistant':
                response = assistant_msg.get('content', '')
            else:
                response = str(assistant_msg)
        else:
            response = str(result)
        
        output = {
            "success": True,
            "response": response,
            "model": "google/gemma-3-270m-it"
        }
        print(json.dumps(output))

except Exception as e:
    import traceback
    error = {
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }
    print(json.dumps(error))

    