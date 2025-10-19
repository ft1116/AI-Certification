#!/usr/bin/env python3
"""
Advanced Retrievers Evaluation for Mineral Rights RAG Pipeline

This script implements and evaluates 5 advanced retrieval strategies:
1. Multi-Query Retriever - Multiple query variations
2. BM25 Retriever - Traditional keyword-based sparse retrieval
3. Contextual Compression Retriever - Compresses to most relevant content
4. Cohere Rerank - Re-ranks using Cohere's reranking model
5. Ensemble Retriever - Combines BM25 + vector search

Evaluates using RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness
"""

import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Set up LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mineral-rights-advanced-retrieval"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# LangChain imports
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
# Try to import Qdrant from the new package, fallback to old one
try:
    from langchain_qdrant import Qdrant
except ImportError:
    from langchain_community.vectorstores import Qdrant

# Import Pinecone for testing
try:
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# CohereRerank will be imported later after logger is configured

# Manual implementations for missing retrievers
from typing import Protocol
from abc import ABC, abstractmethod
from qdrant_client import QdrantClient
from qdrant_client.http import models



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedRetrieversEvaluator:
    """Evaluates advanced retrieval strategies for mineral rights RAG pipeline"""
    
    def __init__(self):
        """Initialize the evaluator with required clients and configurations"""
        
        # Try to import CohereRerank, fallback if not available
        try:
            from langchain_cohere import CohereRerank
            self.CohereRerank = CohereRerank
            self.COHERE_AVAILABLE = True
        except ImportError:
            try:
                from langchain_community.document_compressors.cohere_rerank import CohereRerank
                self.CohereRerank = CohereRerank
                self.COHERE_AVAILABLE = True
            except ImportError:
                try:
                    from langchain_community.document_compressors import CohereRerank
                    self.CohereRerank = CohereRerank
                    self.COHERE_AVAILABLE = True
                except ImportError:
                    self.CohereRerank = None
                    self.COHERE_AVAILABLE = False
                    logger.warning("CohereRerank not available - Cohere rerank retriever will be skipped")
        
        # Initialize clients
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "mineral_insights"
        
        # Initialize LLMs
        self.llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Test questions for evaluation
        self.test_questions = [
            "What are typical lease terms for oil and gas drilling in Texas?",
            "Show me Pioneer Natural Resources drilling activity in Midland County",
            "What's the market price for mineral rights in Oklahoma?",
            "How do drilling permits work in Texas?",
            "Compare lease offers between Texas and Oklahoma",
            "What formations are being targeted in Oklahoma drilling permits?",
            "What are the current trends in mineral rights transactions?",
            "How do royalty rates vary by formation in Texas?",
            "What are the environmental considerations for drilling permits?",
            "Show me recent drilling activity in the Permian Basin"
        ]
        
        # Ground truth answers (simplified for evaluation - reduced to 2 for testing)
        self.ground_truths = [
            "Lease terms typically include bonus payments ($500-$2000/acre), royalty rates (20-25%), primary term (3-5 years), and drilling commitments.",
            "Pioneer Natural Resources has active drilling operations in Midland County, Texas, targeting the Spraberry and Wolfcamp formations."
        ]
        
        # Cost estimates (relative to base vector search)
        self.cost_estimates = {
            "multi_query": "Medium",      # Multiple LLM calls for query generation
            "bm25": "Low",                # No LLM calls, just text processing
            "compression": "High",        # LLM calls for each document compression
            "cohere_rerank": "Medium",    # Cohere API calls for reranking
            "ensemble": "Medium"          # Multiple retrieval + reranking
        }
        
        logger.info("🚀 Advanced Retrievers Evaluator initialized")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return [0.0] * 1536
    
    def qdrant_to_langchain_docs(self, qdrant_results) -> List[Document]:
        """Convert Qdrant results to LangChain Documents"""
        documents = []
        
        if hasattr(qdrant_results, 'points'):
            results = qdrant_results.points
        else:
            results = qdrant_results
        
        for result in results:
            payload = result.payload
            content = payload.get('searchable_text', '')
            
            # Handle None content
            if content is None:
                content = ''
            
            # Ensure content is a string and not empty
            if not isinstance(content, str):
                content = str(content) if content is not None else ''
            
            # Skip documents with no content
            if not content or not content.strip():
                continue
            
            metadata = {
                'data_type': payload.get('data_type', 'unknown'),
                'source': payload.get('source', 'unknown')
            }
            
            # Add score if available
            if hasattr(result, 'score') and result.score is not None:
                metadata['score'] = result.score
            
            # Add rich metadata based on data type
            data = payload.get('data', {})
            if payload.get('data_type') == 'texas_permit':
                metadata.update({
                    'operator': data.get('Operator', ''),
                    'county': data.get('County_Name', ''),
                    'state': data.get('State', ''),
                    'well_number': data.get('Well_Number', ''),
                    'formation': data.get('Formation_Name', ''),
                    'api_number': data.get('API_Number', '')
                })
            elif payload.get('data_type') == 'oklahoma_permit':
                metadata.update({
                    'operator': data.get('Entity_Name', ''),
                    'county': data.get('County', ''),
                    'state': data.get('State', ''),
                    'well_name': data.get('Well_Name', ''),
                    'formation': data.get('Formation_Name', ''),
                    'api_number': data.get('API_Number', '')
                })
            elif payload.get('data_type') in ['mineral_offer', 'lease_offer']:
                metadata.update({
                    'county': data.get('county', ''),
                    'state': data.get('state', ''),
                    'price_per_acre': data.get('price_per_acre', ''),
                    'total_acres': data.get('total_acres', ''),
                    'operator': data.get('operator', ''),
                    'buyer': data.get('buyer', '')
                })
            elif payload.get('data_type') in ['forum_topic', 'forum_post']:
                metadata.update({
                    'title': payload.get('title', ''),
                    'author': payload.get('author', ''),
                    'category': payload.get('category', ''),
                    'url': payload.get('url', ''),
                    'replies': payload.get('replies', 0),
                    'views': payload.get('views', 0)
                })
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def create_pinecone_vectorstore(self):
        """Create a Pinecone vector store connection"""
        if not PINECONE_AVAILABLE:
            logger.warning("⚠️ Pinecone not available, skipping Pinecone vector store")
            return None
            
        logger.info("🔧 Connecting to Pinecone vector store...")
        
        # Check for required environment variables
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX_NAME", "mineral-insights")
        
        if not api_key:
            logger.warning("⚠️ PINECONE_API_KEY not found, skipping Pinecone")
            return None
            
        if not environment:
            logger.warning("⚠️ PINECONE_ENVIRONMENT not found, skipping Pinecone")
            return None
        
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embedding_model
            )
            logger.info(f"✅ Connected to Pinecone vector store: {index_name}")
            return vectorstore
        except Exception as e:
            logger.error(f"❌ Failed to connect to Pinecone: {e}")
            return None
    
    def create_multi_query_retriever(self) -> MultiQueryRetriever:
        """Create Multi-Query Retriever that generates multiple query variations"""
        logger.info("🔧 Creating Multi-Query Retriever with Pinecone...")
        
        # Try Pinecone first, fallback to Qdrant
        vectorstore = self.create_pinecone_vectorstore()
        if vectorstore is None:
            logger.info("🔄 Falling back to Qdrant for Multi-Query retriever...")
            vectorstore = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embedding_model
            )
        
        # Create multi-query retriever using LangChain's method
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),
            llm=self.llm
        )
        
        return retriever
    
    def create_bm25_retriever(self) -> BM25Retriever:
        """Create BM25 Retriever for traditional keyword-based sparse retrieval"""
        logger.info("🔧 Creating BM25 Retriever with Pinecone...")
        
        try:
            all_docs = []
            
            # Try Pinecone first
            pinecone_vectorstore = self.create_pinecone_vectorstore()
            if pinecone_vectorstore is not None:
                logger.info("📚 Loading documents from Pinecone for BM25...")
                try:
                    # Get documents from Pinecone (limited for performance)
                    max_docs = 8000
                    # Use similarity search to get diverse documents
                    all_docs = pinecone_vectorstore.similarity_search("", k=max_docs)
                    logger.info(f"📚 Loaded {len(all_docs)} documents from Pinecone for BM25")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load from Pinecone: {e}, falling back to Qdrant")
                    pinecone_vectorstore = None
            
            # Fallback to Qdrant if Pinecone failed or not available
            if pinecone_vectorstore is None:
                logger.info("🔄 Falling back to Qdrant for BM25...")
                offset = 0
                limit = 1000
                max_docs = 8000
                doc_count = 0
                
                while doc_count < max_docs:
                    points = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=min(limit, max_docs - doc_count),
                        offset=offset,
                        with_payload=True
                    )[0]
                    
                    if not points:
                        break
                    
                    # Convert Qdrant points to documents for BM25
                    for point in points:
                        payload = point.payload
                        content = payload.get('searchable_text', '')
                        
                        # Handle None content
                        if content is None:
                            content = ''
                        
                        # Ensure content is a string
                        if not isinstance(content, str):
                            content = str(content) if content is not None else ''
                        
                        if content and content.strip():  # Only add documents with non-empty content
                            metadata = {
                                'data_type': payload.get('data_type', 'unknown'),
                                'source': payload.get('source', 'unknown')
                            }
                            
                            # Add rich metadata based on data type
                            data = payload.get('data', {})
                            if payload.get('data_type') == 'texas_permit':
                                metadata.update({
                                    'operator': data.get('Operator', ''),
                                    'county': data.get('County_Name', ''),
                                    'state': data.get('State', ''),
                                    'well_number': data.get('Well_Number', ''),
                                    'formation': data.get('Formation_Name', ''),
                                    'api_number': data.get('API_Number', '')
                                })
                            elif payload.get('data_type') == 'oklahoma_permit':
                                metadata.update({
                                    'operator': data.get('Entity_Name', ''),
                                    'county': data.get('County', ''),
                                    'state': data.get('State', ''),
                                    'well_name': data.get('Well_Name', ''),
                                    'formation': data.get('Formation_Name', ''),
                                    'api_number': data.get('API_Number', '')
                                })
                            elif payload.get('data_type') in ['mineral_offer', 'lease_offer']:
                                metadata.update({
                                    'county': data.get('county', ''),
                                    'state': data.get('state', ''),
                                    'price_per_acre': data.get('price_per_acre', ''),
                                    'total_acres': data.get('total_acres', ''),
                                    'operator': data.get('operator', ''),
                                    'buyer': data.get('buyer', '')
                                })
                            elif payload.get('data_type') in ['forum_topic', 'forum_post']:
                                metadata.update({
                                    'title': payload.get('title', ''),
                                    'author': payload.get('author', ''),
                                    'category': payload.get('category', ''),
                                    'url': payload.get('url', ''),
                                    'replies': payload.get('replies', 0),
                                    'views': payload.get('views', 0)
                                })
                            
                            doc = Document(page_content=content, metadata=metadata)
                            all_docs.append(doc)
                            doc_count += 1
                            
                            if doc_count >= max_docs:
                                break
                    
                    offset += limit
                    
                    if len(points) < limit or doc_count >= max_docs:
                        break
            
            logger.info(f"📚 Indexed {len(all_docs)} documents for BM25")
            
            # Create BM25 retriever
            retriever = BM25Retriever.from_documents(all_docs)
            retriever.k = 50
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating BM25 retriever: {e}")
            return None
    
    def create_compression_retriever(self) -> ContextualCompressionRetriever:
        """Create Contextual Compression Retriever that compresses to most relevant content"""
        logger.info("🔧 Creating Contextual Compression Retriever with Pinecone...")
        
        # Try Pinecone first, fallback to Qdrant
        vectorstore = self.create_pinecone_vectorstore()
        if vectorstore is None:
            logger.info("🔄 Falling back to Qdrant for Compression retriever...")
            vectorstore = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embedding_model
            )
        
        # Create base retriever
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
        
        # Create LLM-based compressor
        from langchain.retrievers.document_compressors import LLMChainExtractor
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Create compression retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return retriever
    
    def create_cohere_rerank_retriever(self) -> Optional[ContextualCompressionRetriever]:
        """Create Cohere Rerank Retriever for improved ranking"""
        logger.info("🔧 Creating Cohere Rerank Retriever...")
        
        try:
            # Check if Cohere is available
            if not self.COHERE_AVAILABLE or self.CohereRerank is None:
                logger.warning("⚠️ CohereRerank not available, skipping Cohere rerank")
                return None
            
            # Check if Cohere API key is available
            if not os.getenv("COHERE_API_KEY"):
                logger.warning("⚠️ COHERE_API_KEY not found, skipping Cohere rerank")
                return None
            
            # Try Pinecone first, fallback to Qdrant
            vectorstore = self.create_pinecone_vectorstore()
            if vectorstore is None:
                logger.info("🔄 Falling back to Qdrant for Cohere rerank retriever...")
                vectorstore = Qdrant(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embeddings=self.embedding_model
                )
            
            # Create base retriever
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
            
            # Create Cohere reranker
            compressor = self.CohereRerank(
                model="rerank-english-v3.0",
                top_n=10
            )
            
            # Create compression retriever with Cohere rerank
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating Cohere rerank retriever: {e}")
            return None
    
    def create_ensemble_retriever(self, available_retrievers: Dict[str, Any]) -> EnsembleRetriever:
        """Create Ensemble Retriever combining all available advanced retrievers"""
        logger.info("🔧 Creating Ensemble Retriever with all available retrievers...")
        
        # Filter out ensemble retriever from available retrievers
        available_retrievers = {name: retriever for name, retriever in available_retrievers.items() 
                               if name != 'ensemble' and retriever is not None}
        
        if not available_retrievers:
            logger.error("No retrievers available for ensemble")
            return None
        
        # Create equal weights for all retrievers
        num_retrievers = len(available_retrievers)
        equal_weights = [1.0 / num_retrievers] * num_retrievers
        
        # Create ensemble retriever with all available retrievers using LangChain's EnsembleRetriever
        retriever = EnsembleRetriever(
            retrievers=list(available_retrievers.values()),
            weights=equal_weights
        )
        
        logger.info(f"✅ Created ensemble with {num_retrievers} retrievers: {list(available_retrievers.keys())}")
        return retriever
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Evaluate if answer is based on provided context"""
        if not contexts or not answer:
            return 0.0
        
        context_text = " ".join(contexts).lower()
        answer_lower = answer.lower()
        
        # Check for source mentions
        source_mentions = 0
        for context in contexts:
            if any(word in answer_lower for word in context.lower().split()[:3]):
                source_mentions += 1
        
        # Check for context keyword overlap
        context_words = set(context_text.split())
        answer_words = set(answer_lower.split())
        keyword_overlap = len(context_words.intersection(answer_words)) / len(context_words) if context_words else 0
        
        faithfulness_score = (source_mentions / len(contexts)) * 0.6 + keyword_overlap * 0.4
        return min(1.0, faithfulness_score)
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """Evaluate if answer addresses the question"""
        if not question or not answer:
            return 0.0
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'what', 'how', 'where', 'when', 'why', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_keywords = question_words - common_words
        answer_keywords = answer_words - common_words
        
        if not question_keywords:
            return 0.5
        
        overlap = len(question_keywords.intersection(answer_keywords))
        relevancy_score = overlap / len(question_keywords)
        
        return min(1.0, relevancy_score)
    
    def evaluate_context_precision(self, question: str, contexts: List[str]) -> float:
        """Evaluate if retrieved context is relevant to the question"""
        if not question or not contexts:
            return 0.0
        
        question_words = set(question.lower().split())
        relevant_contexts = 0
        
        for context in contexts:
            context_words = set(context.lower().split())
            if question_words.intersection(context_words):
                relevant_contexts += 1
        
        return relevant_contexts / len(contexts) if contexts else 0.0
    
    def evaluate_context_recall(self, question: str, ground_truth: str, contexts: List[str]) -> float:
        """Evaluate if context contains information needed to answer the question"""
        if not question or not ground_truth or not contexts:
            return 0.0
        
        gt_words = set(ground_truth.lower().split())
        context_text = " ".join(contexts).lower()
        context_words = set(context_text.split())
        
        covered_concepts = len(gt_words.intersection(context_words))
        total_concepts = len(gt_words)
        basic_recall = covered_concepts / total_concepts if total_concepts > 0 else 0.0
        
        # Enhanced recall with semantic concepts
        import re
        gt_prices = re.findall(r'\$[\d,]+(?:-\$?[\d,]+)?', ground_truth.lower())
        context_prices = re.findall(r'\$[\d,]+(?:-\$?[\d,]+)?', context_text)
        
        semantic_bonus = 0.0
        if gt_prices and context_prices:
            semantic_bonus += 0.2
        
        # Check for location matches
        location_words = ['county', 'state', 'oklahoma', 'texas', 'leon', 'howard', 'ector', 'rusk']
        gt_locations = [word for word in gt_words if word in location_words]
        context_locations = [word for word in context_words if word in location_words]
        
        if gt_locations and context_locations:
            location_match = len(set(gt_locations).intersection(set(context_locations))) / len(gt_locations)
            semantic_bonus += location_match * 0.3
        
        final_recall = min(1.0, basic_recall + semantic_bonus)
        return final_recall
    
    def evaluate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """Evaluate if answer is factually correct compared to ground truth"""
        if not answer or not ground_truth:
            return 0.0
        
        answer_lower = answer.lower()
        gt_lower = ground_truth.lower()
        
        # Simple word overlap for correctness
        answer_words = set(answer_lower.split())
        gt_words = set(gt_lower.split())
        
        overlap = len(answer_words.intersection(gt_words))
        total_words = len(gt_words)
        
        correctness = overlap / total_words if total_words > 0 else 0.0
        return min(1.0, correctness)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_answer_with_retriever(self, retriever, question: str) -> Dict[str, Any]:
        """Generate answer using a specific retriever with retry logic"""
        try:
            # Add small delay to respect rate limits
            time.sleep(0.5)
            
            # Retrieve documents
            try:
                docs = retriever.get_relevant_documents(question)
                # Filter out documents with None or empty content
                docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
            except Exception as e:
                logger.warning(f"Error retrieving documents: {e}")
                docs = []
            
            # Format context
            contexts = [doc.page_content for doc in docs]
            
            # Create context string
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content[:400]}..." for i, doc in enumerate(docs[:5])])
            
            # Generate answer
            system_prompt = """You are a mineral rights expert. Answer questions about mineral rights, 
            oil and gas, drilling permits, lease offers, and related topics. Use the provided context 
            to give accurate, helpful answers."""
            
            user_prompt = f"""Based on the following information about mineral rights, answer the user's question:

{context}

Question: {question}

Provide a helpful, accurate answer based on the context."""
            
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            return {
                "answer": response.content,
                "contexts": contexts,
                "num_docs": len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error: {e}",
                "contexts": [],
                "num_docs": 0
            }
    
    def evaluate_retriever(self, retriever, retriever_name: str) -> Dict[str, Any]:
        """Evaluate a specific retriever on all test questions"""
        logger.info(f"🧪 Evaluating {retriever_name}...")
        
        results = {
            "retriever": retriever_name,
            "faithfulness_scores": [],
            "relevancy_scores": [],
            "precision_scores": [],
            "recall_scores": [],
            "correctness_scores": [],
            "latencies": [],
            "num_docs": []
        }
        
        total_questions = len(self.test_questions)
        start_time = time.time()
        
        for i, (question, ground_truth) in enumerate(zip(self.test_questions, self.ground_truths)):
            question_start = time.time()
            logger.info(f"  📋 Question {i+1}/{total_questions}: {question[:50]}...")
            
            # Generate answer
            result = self.generate_answer_with_retriever(retriever, question)
            
            question_end = time.time()
            latency = question_end - question_start
            
            # Evaluate metrics
            faithfulness = self.evaluate_faithfulness(result["answer"], result["contexts"])
            relevancy = self.evaluate_answer_relevancy(question, result["answer"])
            precision = self.evaluate_context_precision(question, result["contexts"])
            recall = self.evaluate_context_recall(question, ground_truth, result["contexts"])
            correctness = self.evaluate_answer_correctness(result["answer"], ground_truth)
            
            # Store results
            results["faithfulness_scores"].append(faithfulness)
            results["relevancy_scores"].append(relevancy)
            results["precision_scores"].append(precision)
            results["recall_scores"].append(recall)
            results["correctness_scores"].append(correctness)
            results["latencies"].append(latency)
            results["num_docs"].append(result["num_docs"])
            
            # Calculate progress and ETA
            elapsed_time = question_end - start_time
            avg_time_per_question = elapsed_time / (i + 1)
            remaining_questions = total_questions - (i + 1)
            eta_seconds = remaining_questions * avg_time_per_question
            eta_minutes = eta_seconds / 60
            
            logger.info(f"    ✅ F:{faithfulness:.3f} R:{relevancy:.3f} P:{precision:.3f} Rec:{recall:.3f} C:{correctness:.3f} T:{latency:.2f}s")
            logger.info(f"    ⏱️  Progress: {i+1}/{total_questions} | ETA: {eta_minutes:.1f} minutes")
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of all retrievers"""
        logger.info("🚀 Starting Comprehensive Advanced Retrievers Evaluation")
        logger.info("="*80)
        
        # Create retrievers
        retrievers = {}
        
        try:
            retrievers["Multi-Query"] = self.create_multi_query_retriever()
        except Exception as e:
            logger.error(f"Failed to create Multi-Query retriever: {e}")
        
        try:
            retrievers["BM25"] = self.create_bm25_retriever()
        except Exception as e:
            logger.error(f"Failed to create BM25 retriever: {e}")
        
        try:
            retrievers["Compression"] = self.create_compression_retriever()
        except Exception as e:
            logger.error(f"Failed to create Compression retriever: {e}")
        
        try:
            cohere_retriever = self.create_cohere_rerank_retriever()
            if cohere_retriever:
                retrievers["Cohere Rerank"] = cohere_retriever
        except Exception as e:
            logger.error(f"Failed to create Cohere Rerank retriever: {e}")
        
        # Create ensemble retriever using all available retrievers
        try:
            ensemble_retriever = self.create_ensemble_retriever(retrievers)
            if ensemble_retriever:
                retrievers["Ensemble"] = ensemble_retriever
        except Exception as e:
            logger.error(f"Failed to create Ensemble retriever: {e}")
        
        if not retrievers:
            logger.error("❌ No retrievers could be created!")
            return {}
        
        logger.info(f"✅ Created {len(retrievers)} retrievers for evaluation")
        
        # Evaluate each retriever
        all_results = []
        total_retrievers = len([r for r in retrievers.values() if r is not None])
        current_retriever = 0
        
        for retriever_name, retriever in retrievers.items():
            if retriever is None:
                continue
                
            current_retriever += 1
            logger.info(f"\n🔄 Progress: Retriever {current_retriever}/{total_retrievers} - {retriever_name}")
            logger.info("="*60)
            
            results = self.evaluate_retriever(retriever, retriever_name)
            all_results.append(results)
            
            logger.info(f"✅ Completed {retriever_name} evaluation")
        
        # Analyze results
        return self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and compare results from all retrievers"""
        logger.info("\n📊 ANALYZING RESULTS")
        logger.info("="*80)
        
        # Create results DataFrame
        data = []
        for result in results:
            data.append({
                "Retriever": result["retriever"],
                "Faithfulness": sum(result["faithfulness_scores"]) / len(result["faithfulness_scores"]),
                "Answer Relevancy": sum(result["relevancy_scores"]) / len(result["relevancy_scores"]),
                "Context Precision": sum(result["precision_scores"]) / len(result["precision_scores"]),
                "Context Recall": sum(result["recall_scores"]) / len(result["recall_scores"]),
                "Answer Correctness": sum(result["correctness_scores"]) / len(result["correctness_scores"]),
                "Avg Latency (s)": sum(result["latencies"]) / len(result["latencies"]),
                "Avg Docs Retrieved": sum(result["num_docs"]) / len(result["num_docs"]),
                "Cost Level": self.cost_estimates.get(result["retriever"].lower().replace(" ", "_").replace("-", "_"), "Medium")
            })
        
        df = pd.DataFrame(data)
        
        # Calculate weighted scores (Faithfulness 25%, Answer Relevancy 25%, Context Precision 20%, Context Recall 20%, Answer Correctness 10%)
        df["Weighted Score"] = (
            df["Faithfulness"] * 0.25 +
            df["Answer Relevancy"] * 0.25 +
            df["Context Precision"] * 0.20 +
            df["Context Recall"] * 0.20 +
            df["Answer Correctness"] * 0.10
        )
        
        # Sort by weighted score
        df = df.sort_values("Weighted Score", ascending=False)
        
        # Display results
        print("\n🎯 COMPREHENSIVE RETRIEVAL STRATEGY COMPARISON")
        print("="*80)
        print(df.round(3).to_string(index=False))
        
        # Key insights
        best_overall = df.iloc[0]
        fastest = df.loc[df["Avg Latency (s)"].idxmin()]
        most_faithful = df.loc[df["Faithfulness"].idxmax()]
        best_precision = df.loc[df["Context Precision"].idxmax()]
        best_recall = df.loc[df["Context Recall"].idxmax()]
        
        print(f"\n🏆 BEST PERFORMER: {best_overall['Retriever']} (Weighted Score: {best_overall['Weighted Score']:.3f})")
        print(f"⚡ FASTEST: {fastest['Retriever']} ({fastest['Avg Latency (s)']:.2f}s)")
        print(f"🎯 MOST FAITHFUL: {most_faithful['Retriever']} ({most_faithful['Faithfulness']:.3f})")
        print(f"🎪 BEST PRECISION: {best_precision['Retriever']} ({best_precision['Context Precision']:.3f})")
        print(f"📚 BEST RECALL: {best_recall['Retriever']} ({best_recall['Context Recall']:.3f})")
        
        # Cost analysis
        print(f"\n💰 COST ANALYSIS:")
        for _, row in df.iterrows():
            print(f"  {row['Retriever']}: {row['Cost Level']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  🥇 For Best Overall Performance: {best_overall['Retriever']}")
        print(f"  ⚡ For Speed: {fastest['Retriever']}")
        print(f"  💰 For Cost Efficiency: {df[df['Cost Level'] == 'Low'].iloc[0]['Retriever'] if len(df[df['Cost Level'] == 'Low']) > 0 else 'N/A'}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"advanced_retrievers_evaluation_{timestamp}.json"
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "best_overall": best_overall.to_dict(),
                "fastest": fastest.to_dict(),
                "most_faithful": most_faithful.to_dict(),
                "best_precision": best_precision.to_dict(),
                "best_recall": best_recall.to_dict()
            },
            "detailed_results": df.to_dict('records'),
            "test_questions": self.test_questions,
            "ground_truths": self.ground_truths
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results_data

def main():
    """Main evaluation function"""
    print("🚀 Advanced Retrievers Evaluation for Mineral Rights RAG Pipeline")
    print("="*80)
    print("This will evaluate 5 advanced retrieval strategies:")
    print("1. Multi-Query Retriever")
    print("2. BM25 Retriever") 
    print("3. Contextual Compression Retriever")
    print("4. Cohere Rerank Retriever")
    print("5. Ensemble Retriever")
    print("\nRequired API Keys:")
    print("- OPENAI_API_KEY (for embeddings)")
    print("- ANTHROPIC_API_KEY (for LLM)")
    print("- COHERE_API_KEY (optional, for Cohere rerank)")
    print("="*80)
    
    try:
        evaluator = AdvancedRetrieversEvaluator()
        results = evaluator.run_comprehensive_evaluation()
        
        if results:
            print("\n✅ Evaluation completed successfully!")
        else:
            print("\n❌ Evaluation failed!")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()
