#!/usr/bin/env python3
"""
RAGAS Synthetic Dataset Generator for Mineral Rights RAG Application

This script generates synthetic evaluation datasets using RAGAS for testing
the performance of your LangGraph mineral rights RAG system.

Features:
- Loads multiple data sources (forum, permits, lease offers, mineral offers)
- Generates diverse question types using RAGAS knowledge graph approach
- Creates realistic test scenarios for mineral rights domain
- Exports datasets in multiple formats
"""

import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Core libraries
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS components
from ragas.testset import TestsetGenerator
from ragas.llms.base import llm_factory
from ragas.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MineralRightsDatasetGenerator:
    """
    Generates synthetic evaluation datasets for mineral rights RAG system
    using RAGAS knowledge graph-based approach.
    """
    
    def __init__(self, testset_size: int = 10):
        """
        Initialize the dataset generator.
        
        Args:
            testset_size: Number of synthetic test samples to generate
        """
        self.testset_size = testset_size
        self.documents = []
        self.generator = None
        self.dataset = None
        
        # Validate API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        logger.info(f"Initialized MineralRightsDatasetGenerator with {testset_size} samples")
    
    def load_documents(self) -> List[Document]:
        """
        Load all document sources and convert them to LangChain Document objects.
        
        Returns:
            List of Document objects from all sources
        """
        logger.info("Loading documents from all sources...")
        
        # Updated document paths to match your current file structure
        data_sources = {
            "forum": "forum_enhanced.json",
            "texas_permits": "texas_permits_20251004_cleaned.csv",
            "oklahoma_permits": "oklahoma_permits_streamlined.csv", 
            "mineral_offers": "current_mineral_offers_20251018_095734.csv",
            "lease_offers": "current_lease_offers_20251018_095726.csv"
        }
        
        all_documents = []
        
        # Create progress bar for document loading
        loading_bar = tqdm(
            data_sources.items(),
            desc="Loading documents",
            unit="source",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}'
        )
        
        # Load documents with progress tracking
        for source_name, file_path in loading_bar:
            loading_bar.set_description(f"Loading {source_name}")
            
            if os.path.exists(file_path):
                if source_name == "forum":
                    docs = self._load_forum_data(file_path)
                else:
                    docs = self._load_csv_data(file_path, source_name)
                
                all_documents.extend(docs)
                loading_bar.set_postfix({
                    'Loaded': len(docs),
                    'Total': len(all_documents)
                })
                logger.info(f"Loaded {len(docs)} {source_name} documents")
            else:
                logger.warning(f"File not found: {file_path}")
                loading_bar.set_postfix({'Status': 'File not found'})
        
        loading_bar.close()
        
        # Text splitting for better chunking
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks = fewer total chunks
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split documents into chunks
        split_documents = text_splitter.split_documents(all_documents)
        logger.info(f"Split into {len(split_documents)} document chunks")
        
        # Limit to first 50 documents to reduce processing time and API calls
        if len(split_documents) > 50:
            logger.info(f"Limiting to first 50 documents to reduce API usage")
            split_documents = split_documents[:50]
        
        self.documents = split_documents
        return split_documents
    
    def _load_forum_data(self, file_path: str) -> List[Document]:
        """Load forum data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for doc in data.get("rag_documents", []):
            # Combine all posts in a document
            content = ""
            for post in doc.get("posts", []):
                content += f"Post: {post.get('content', '')}\n"
            
            if content.strip():
                metadata = {
                    "source": "forum",
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "category": doc.get("category", ""),
                    "post_count": len(doc.get("posts", [])),
                    "data_type": "forum"
                }
                documents.append(Document(page_content=content.strip(), metadata=metadata))
        
        return documents
    
    def _load_csv_data(self, file_path: str, source_type: str) -> List[Document]:
        """Load CSV data and convert to documents."""
        try:
            df = pd.read_csv(file_path)
            documents = []
            
            for idx, row in df.iterrows():
                # Create content from relevant columns
                content_parts = []
                for col in df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        content_parts.append(f"{col}: {row[col]}")
                
                content = "\n".join(content_parts)
                
                if content.strip():
                    metadata = {
                        "source": source_type,
                        "row_index": idx,
                        "data_type": source_type,
                        **{col: row[col] for col in df.columns if pd.notna(row[col])}
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def setup_ragas_generator(self):
        """
        Configure RAGAS TestsetGenerator with OpenAI models.
        
        RAGAS uses a knowledge graph-based approach where:
        1. It analyzes your documents to understand relationships
        2. Generates questions that test different aspects of retrieval
        3. Creates realistic scenarios based on your domain
        """
        logger.info("Setting up RAGAS generator...")
        
        # Configure LLM for generation (GPT-4o-mini for cost-effectiveness)
        generator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configure critic LLM (GPT-4 for quality assessment)
        critic_llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using same model for cost efficiency
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configure embeddings
        import openai
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            client=openai_client
        )
        
        # Create TestsetGenerator using from_langchain method
        self.generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )
        
        logger.info("RAGAS generator configured successfully")
    
    def generate_synthetic_dataset(self) -> pd.DataFrame:
        """
        Generate synthetic test dataset using RAGAS.
        
        RAGAS Synthesizer Types Explained:
        - single_hop_specific: Direct questions with specific answers (e.g., "What is the royalty rate in Ector County?")
        - multi_hop_abstract: Questions requiring reasoning across multiple documents (e.g., "How do oil prices affect lease rates?")
        - multi_hop_specific: Complex questions with specific answers from multiple sources (e.g., "Compare lease rates between Texas and Oklahoma counties")
        - single_hop_abstract: General questions about concepts (e.g., "What are mineral rights?")
        
        Returns:
            DataFrame with synthetic test data
        """
        if not self.generator:
            raise ValueError("Generator not set up. Call setup_ragas_generator() first.")
        
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        logger.info(f"Generating {self.testset_size} synthetic test samples...")
        
        try:
            # Create progress bar for generation
            progress_bar = tqdm(
                total=self.testset_size,
                desc="Generating synthetic dataset",
                unit="sample",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            # Custom callback to update progress bar
            def progress_callback(event):
                if hasattr(event, 'sample') and event.sample:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Generated': f"{progress_bar.n}/{self.testset_size}",
                        'Status': 'Processing...'
                    })
            
            # Generate synthetic dataset using the correct method
            # RAGAS will automatically create diverse question types
            self.dataset = self.generator.generate_with_langchain_docs(
                documents=self.documents,
                testset_size=self.testset_size,
                with_debugging_logs=True,
                callbacks=[progress_callback] if hasattr(self.generator, 'callbacks') else None
            )
            
            # Close progress bar
            progress_bar.close()
            
            logger.info(f"Generated {len(self.dataset)} test samples")
            
            # Convert to DataFrame for easier inspection
            df = self.dataset.to_pandas()
            
            # Add metadata
            df['generated_at'] = datetime.now().isoformat()
            df['domain'] = 'mineral_rights'
            df['testset_size'] = self.testset_size
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            raise
    
    def inspect_dataset(self, df: pd.DataFrame, num_samples: int = 5):
        """
        Inspect and display sample questions from the generated dataset.
        
        Args:
            df: Generated dataset DataFrame
            num_samples: Number of samples to display
        """
        logger.info("Dataset Inspection:")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Columns: {list(df.columns)}")
        
        print("\n" + "="*80)
        print("SAMPLE SYNTHETIC TEST QUESTIONS")
        print("="*80)
        
        for i in range(min(num_samples, len(df))):
            row = df.iloc[i]
            print(f"\nSample {i+1}:")
            print(f"Question: {row.get('question', 'N/A')}")
            print(f"Ground Truth: {row.get('ground_truth', 'N/A')[:200]}...")
            print(f"Context Sources: {len(row.get('contexts', []))} documents")
            if row.get('contexts'):
                print(f"First Context: {row['contexts'][0][:100]}...")
            print("-" * 60)
    
    def save_dataset(self, df: pd.DataFrame, base_filename: str = None):
        """
        Save the generated dataset in multiple formats.
        
        Args:
            df: Generated dataset DataFrame
            base_filename: Base filename (without extension)
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"mineral_rights_synthetic_dataset_{timestamp}"
        
        # Save as CSV
        csv_path = f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved as CSV: {csv_path}")
        
        # Save as JSON
        json_path = f"{base_filename}.json"
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Dataset saved as JSON: {json_path}")
        
        # Save as Parquet (more efficient for large datasets)
        try:
            parquet_path = f"{base_filename}.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Dataset saved as Parquet: {parquet_path}")
        except Exception as e:
            logger.warning(f"Could not save as Parquet: {e}")
            parquet_path = None
        
        return {
            'csv': csv_path,
            'json': json_path,
            'parquet': parquet_path
        }
    
    def run_full_pipeline(self):
        """
        Run the complete pipeline: load documents, setup generator, generate dataset, inspect, and save.
        
        Returns:
            Dictionary with paths to saved files and the generated DataFrame
        """
        logger.info("Starting full synthetic dataset generation pipeline...")
        
        try:
            # Step 1: Load documents
            self.load_documents()
            
            # Step 2: Setup RAGAS generator
            self.setup_ragas_generator()
            
            # Step 3: Generate synthetic dataset
            df = self.generate_synthetic_dataset()
            
            # Step 4: Inspect dataset
            self.inspect_dataset(df)
            
            # Step 5: Save dataset
            saved_paths = self.save_dataset(df)
            
            logger.info("Pipeline completed successfully!")
            
            return {
                'dataset': df,
                'saved_paths': saved_paths,
                'num_samples': len(df),
                'num_documents': len(self.documents)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """
    Main function to run the synthetic dataset generation.
    """
    print("Mineral Rights RAGAS Synthetic Dataset Generator")
    print("=" * 60)
    
    try:
        # Initialize generator
        generator = MineralRightsDatasetGenerator(testset_size=40)
        
        # Run full pipeline
        results = generator.run_full_pipeline()
        
        print(f"\n✅ Successfully generated {results['num_samples']} synthetic test samples")
        print(f"📊 Processed {results['num_documents']} document chunks")
        print(f"💾 Files saved:")
        for format_type, path in results['saved_paths'].items():
            if path:  # Only show non-None paths
                print(f"   - {format_type.upper()}: {path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Run the generator
    results = main()
    
    if results:
        print("\n🎉 Dataset generation completed successfully!")
        print("You can now use this synthetic dataset to evaluate your RAG system.")
    else:
        print("\n💥 Dataset generation failed. Check the logs for details.")
