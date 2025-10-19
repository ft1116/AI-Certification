#!/usr/bin/env python3
"""
Comprehensive data ingestion script for Mineral Insights project.
Ingests all data sources into Qdrant vector database for LangGraph usage.

Data Sources:
- mineral_offers.csv: Mineral rights purchase offers
- lease_offers.csv: Oil & gas lease offers  
- forum_enhanced.json: Forum discussions
- texas_permits_20251004_cleaned.csv: Texas drilling permits (with proper county names)
- oklahoma_permits_streamlined.csv: Oklahoma drilling permits (streamlined)
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Vector database and embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MineralDataIngester:
    """Main class for ingesting all mineral insights data into Qdrant."""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "mineral_insights"):
        """Initialize the ingester with Qdrant connection."""
        self.client = QdrantClient(
            url=qdrant_url,
            timeout=300  # 5 minutes timeout for large uploads
        )
        self.collection_name = collection_name
        self.batch_size = 500  # Process in smaller batches to avoid timeouts
        
        # Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.embedding_dim = 1536  # Dimension for text-embedding-3-small
        
        # Data paths
        self.base_path = Path(__file__).parent
        self.data_paths = {
            'mineral_offers': self.base_path / 'current_mineral_offers_20251018_095734.csv',
            'lease_offers': self.base_path / 'current_lease_offers_20251018_095726.csv',
            'forum_enhanced': self.base_path / 'forum_enhanced.json',
            'texas_permits': self.base_path / 'texas_permits_20251004_cleaned.csv',
            'oklahoma_permits': self.base_path / 'oklahoma_permits_streamlined.csv'
        }
        
        # Statistics
        self.stats = {
            'total_documents': 0,
            'mineral_offers': 0,
            'lease_offers': 0,
            'forum_posts': 0,
            'texas_permits': 0,
            'oklahoma_permits': 0,
            'errors': 0,
            'api_calls': 0,
            'estimated_cost': 0.0
        }
    
    def setup_collection(self):
        """Create or recreate the Qdrant collection."""
        logger.info(f"Setting up collection: {self.collection_name}")
        
        try:
            # Delete existing collection if it exists
            collections = self.client.get_collections()
            if any(col.name == self.collection_name for col in collections.collections):
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI API."""
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dim
            
            # Clean and truncate text (OpenAI has 8192 token limit)
            text = text.strip()[:6000]  # Conservative limit
            
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            
            self.stats['api_calls'] += 1
            # Estimate cost: $0.02 per million tokens, ~250 tokens per 1K chars
            estimated_tokens = len(text) * 0.25
            self.stats['estimated_cost'] += (estimated_tokens / 1000000) * 0.02
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error creating OpenAI embedding: {e}")
            return [0.0] * self.embedding_dim
    
    def create_searchable_text(self, data: Dict[str, Any], data_type: str) -> str:
        """Create searchable text from data based on type."""
        if data_type == 'mineral_offer':
            return f"""
            Mineral Rights Purchase Offer
            Location: {data.get('state', '')} {data.get('county', '')} {data.get('section_township_range', '')}
            Price: {data.get('price_text', '')} per acre
            Total Acres: {data.get('total_acres', '')}
            Buyer: {data.get('buyer', '')}
            Details: {data.get('additional_details', '')}
            Content: {data.get('raw_content', '')}
            """.strip()
        
        elif data_type == 'lease_offer':
            return f"""
            Oil & Gas Lease Offer
            Location: {data.get('state', '')} {data.get('county', '')} {data.get('section_township_range', '')}
            Operator: {data.get('operator', '')}
            Bonus: {data.get('bonus_amount_text', '')} per acre
            Royalty: {data.get('royalty_text', '')}
            Term: {data.get('term_text', '')}
            Additional Terms: {data.get('additional_terms', '')}
            Content: {data.get('raw_content', '')}
            """.strip()
        
        elif data_type == 'forum_post':
            return f"""
            Forum Discussion: {data.get('title', '')}
            Category: {data.get('category', '')}
            Author: {data.get('author', '')}
            Content: {data.get('content', '')}
            Replies: {data.get('replies', 0)}
            Views: {data.get('views', 0)}
            """.strip()
        
        elif data_type == 'texas_permit':
            return f"""
            Texas Drilling Permit
            Operator: {data.get('Operator', '')}
            Well: {data.get('Well_Number', '')}
            Location: {data.get('County_Name', data.get('County', ''))} County, Texas, Section {data.get('Section', '')}
            Purpose: {data.get('Filing_Purpose', '')}
            Profile: {data.get('Wellbore_Profile', '')}
            Lease: {data.get('Lease_Name', '')}
            This is a Texas Railroad Commission drilling permit.
            """.strip()
        
        elif data_type == 'oklahoma_permit':
            return f"""
            Oklahoma Drilling Permit
            Operator: {data.get('Entity_Name', '')}
            Well: {data.get('Well_Name', '')} {data.get('Well_Number', '')}
            Location: {data.get('County', '')} County, Section {data.get('Section', '')}
            Type: {data.get('Well_Type', '')}
            Status: {data.get('Well_Status', '')}
            Formation: {data.get('Formation_Name', '')}
            Depth: {data.get('Total_Depth', '')} ft
            """.strip()
        
        return str(data)
    
    def upload_batch_with_retry(self, points: List[PointStruct], data_type: str, max_retries: int = 3):
        """Upload a batch of points with retry logic."""
        for attempt in range(max_retries):
            try:
                if not points:
                    return True
                
                logger.info(f"Uploading batch of {len(points)} {data_type} (attempt {attempt + 1}/{max_retries})")
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Wait for confirmation
                )
                logger.info(f"✅ Successfully uploaded {len(points)} {data_type}")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Upload attempt {attempt + 1} failed for {data_type}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed to upload {data_type} after {max_retries} attempts")
                    return False
                else:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    import time
                    time.sleep(wait_time)
        
        return False
    
    def ingest_mineral_offers(self):
        """Ingest mineral rights purchase offers."""
        logger.info("Ingesting mineral offers...")
        
        try:
            df = pd.read_csv(self.data_paths['mineral_offers'])
            
            # Progress bar for mineral offers
            with tqdm(total=len(df), desc="Mineral Offers", unit="offers") as pbar:
                batch_points = []
                
                for idx, row in df.iterrows():
                    try:
                        data = row.to_dict()
                        searchable_text = self.create_searchable_text(data, 'mineral_offer')
                        embedding = self.create_embedding(searchable_text)
                        
                        point = PointStruct(
                            id=hash(f"mineral_offer_{data.get('id', idx)}") % (2**63 - 1),
                            vector=embedding,
                            payload={
                                'data_type': 'mineral_offer',
                                'source': 'current_mineral_offers_20251018_095734.csv',
                                'data': data,
                                'searchable_text': searchable_text,
                                'ingested_at': datetime.now().isoformat()
                            }
                        )
                        batch_points.append(point)
                        self.stats['mineral_offers'] += 1
                        pbar.update(1)
                        
                        # Upload batch when it reaches batch_size
                        if len(batch_points) >= self.batch_size:
                            if self.upload_batch_with_retry(batch_points, "mineral offers"):
                                batch_points = []  # Clear batch after successful upload
                            else:
                                self.stats['errors'] += len(batch_points)
                                batch_points = []  # Clear batch even on failure
                        
                    except Exception as e:
                        logger.error(f"Error processing mineral offer {idx}: {e}")
                        self.stats['errors'] += 1
                        pbar.update(1)
                
                # Upload remaining points
                if batch_points:
                    if self.upload_batch_with_retry(batch_points, "mineral offers"):
                        pass  # Success
                    else:
                        self.stats['errors'] += len(batch_points)
            
        except Exception as e:
            logger.error(f"Error ingesting mineral offers: {e}")
            self.stats['errors'] += 1
    
    def ingest_lease_offers(self):
        """Ingest oil & gas lease offers."""
        logger.info("Ingesting lease offers...")
        
        try:
            df = pd.read_csv(self.data_paths['lease_offers'])
            
            # Progress bar for lease offers
            with tqdm(total=len(df), desc="Lease Offers", unit="offers") as pbar:
                batch_points = []
                
                for idx, row in df.iterrows():
                    try:
                        data = row.to_dict()
                        searchable_text = self.create_searchable_text(data, 'lease_offer')
                        embedding = self.create_embedding(searchable_text)
                        
                        point = PointStruct(
                            id=hash(f"lease_offer_{data.get('id', idx)}") % (2**63 - 1),
                            vector=embedding,
                            payload={
                                'data_type': 'lease_offer',
                                'source': 'current_lease_offers_20251018_095726.csv',
                                'data': data,
                                'searchable_text': searchable_text,
                                'ingested_at': datetime.now().isoformat()
                            }
                        )
                        batch_points.append(point)
                        self.stats['lease_offers'] += 1
                        pbar.update(1)
                        
                        # Upload batch when it reaches batch_size
                        if len(batch_points) >= self.batch_size:
                            if self.upload_batch_with_retry(batch_points, "lease offers"):
                                batch_points = []  # Clear batch after successful upload
                            else:
                                self.stats['errors'] += len(batch_points)
                                batch_points = []  # Clear batch even on failure
                        
                    except Exception as e:
                        logger.error(f"Error processing lease offer {idx}: {e}")
                        self.stats['errors'] += 1
                        pbar.update(1)
                
                # Upload remaining points
                if batch_points:
                    if self.upload_batch_with_retry(batch_points, "lease offers"):
                        pass  # Success
                    else:
                        self.stats['errors'] += len(batch_points)
            
        except Exception as e:
            logger.error(f"Error ingesting lease offers: {e}")
            self.stats['errors'] += 1
    
    def ingest_forum_data(self):
        """Ingest forum discussions."""
        logger.info("Ingesting forum data...")
        
        try:
            with open(self.data_paths['forum_enhanced'], 'r', encoding='utf-8') as f:
                forum_data = json.load(f)
            
            topics = forum_data.get('rag_documents', [])
            
            # Progress bar for forum data
            with tqdm(total=len(topics), desc="Forum Topics", unit="topics") as pbar:
                batch_points = []
                
                for topic in topics:
                    try:
                        # Create main topic point
                        searchable_text = self.create_searchable_text(topic, 'forum_post')
                        embedding = self.create_embedding(searchable_text)
                        
                        point = PointStruct(
                            id=hash(f"forum_topic_{topic.get('id', '')}") % (2**63 - 1),
                            vector=embedding,
                            payload={
                                'data_type': 'forum_topic',
                                'source': 'forum_enhanced.json',
                                'data': topic,
                                'searchable_text': searchable_text,
                                'ingested_at': datetime.now().isoformat(),
                                # Rich metadata for filtering and search
                                'title': topic.get('title', ''),
                                'url': topic.get('url', ''),
                                'category': topic.get('category', ''),
                                'replies': topic.get('replies', 0),
                                'views': topic.get('views', 0),
                                'post_count': len(topic.get('posts', [])),
                                'states': topic.get('metadata', {}).get('states', []),
                                'counties': topic.get('metadata', {}).get('counties', []),
                                'operators': topic.get('metadata', {}).get('operators', []),
                                'section_township_range': topic.get('metadata', {}).get('section_township_range', ''),
                                'scraped_at': topic.get('metadata', {}).get('scraped_at', ''),
                                'primary_category': topic.get('metadata', {}).get('primary_category', ''),
                                'mentioned_locations': topic.get('metadata', {}).get('mentioned_locations', [])
                            }
                        )
                        batch_points.append(point)
                        self.stats['forum_posts'] += 1
                        
                        # Create individual post points
                        for post in topic.get('posts', []):
                            try:
                                post_text = self.create_searchable_text({
                                    'title': topic.get('title', ''),
                                    'author': post.get('author', ''),
                                    'content': post.get('content', ''),
                                    'category': topic.get('category', ''),
                                    'replies': len(topic.get('posts', [])),
                                    'views': topic.get('views', 0)
                                }, 'forum_post')
                                
                                post_embedding = self.create_embedding(post_text)
                                
                                post_point = PointStruct(
                                    id=hash(f"forum_post_{topic.get('id', '')}_{post.get('post_number', '')}") % (2**63 - 1),
                                    vector=post_embedding,
                                    payload={
                                        'data_type': 'forum_post',
                                        'source': 'forum_enhanced.json',
                                        'topic_id': topic.get('id', ''),
                                        'data': post,
                                        'searchable_text': post_text,
                                        'ingested_at': datetime.now().isoformat(),
                                        # Rich metadata for filtering and search
                                        'author': post.get('author', ''),
                                        'date': post.get('date', ''),
                                        'post_number': post.get('post_number', ''),
                                        'topic_title': topic.get('title', ''),
                                        'topic_url': topic.get('url', ''),
                                        'category': topic.get('category', ''),
                                        'states': topic.get('metadata', {}).get('states', []),
                                        'counties': topic.get('metadata', {}).get('counties', []),
                                        'operators': topic.get('metadata', {}).get('operators', [])
                                    }
                                )
                                batch_points.append(post_point)
                                self.stats['forum_posts'] += 1
                                
                                # Upload batch when it reaches batch_size
                                if len(batch_points) >= self.batch_size:
                                    if self.upload_batch_with_retry(batch_points, "forum documents"):
                                        batch_points = []  # Clear batch after successful upload
                                    else:
                                        self.stats['errors'] += len(batch_points)
                                        batch_points = []  # Clear batch even on failure
                                
                            except Exception as e:
                                logger.error(f"Error processing forum post: {e}")
                                self.stats['errors'] += 1
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing forum topic: {e}")
                        self.stats['errors'] += 1
                        pbar.update(1)
                
                # Upload remaining points
                if batch_points:
                    if self.upload_batch_with_retry(batch_points, "forum documents"):
                        pass  # Success
                    else:
                        self.stats['errors'] += len(batch_points)
            
        except Exception as e:
            logger.error(f"Error ingesting forum data: {e}")
            self.stats['errors'] += 1
    
    def ingest_texas_permits(self):
        """Ingest Texas drilling permits."""
        logger.info("Ingesting Texas permits...")
        
        try:
            df = pd.read_csv(self.data_paths['texas_permits'])
            
            # Progress bar for Texas permits
            with tqdm(total=len(df), desc="Texas Permits", unit="permits") as pbar:
                batch_points = []
                
                for idx, row in df.iterrows():
                    try:
                        data = row.to_dict()
                        searchable_text = self.create_searchable_text(data, 'texas_permit')
                        embedding = self.create_embedding(searchable_text)
                        
                        point = PointStruct(
                            id=hash(f"texas_permit_{data.get('API_Number', idx)}") % (2**63 - 1),
                            vector=embedding,
                            payload={
                                'data_type': 'texas_permit',
                                'source': 'texas_permits_20251004_cleaned.csv',
                                'data': data,
                                'searchable_text': searchable_text,
                                'ingested_at': datetime.now().isoformat()
                            }
                        )
                        batch_points.append(point)
                        self.stats['texas_permits'] += 1
                        pbar.update(1)
                        
                        # Upload batch when it reaches batch_size
                        if len(batch_points) >= self.batch_size:
                            if self.upload_batch_with_retry(batch_points, "Texas permits"):
                                batch_points = []  # Clear batch after successful upload
                            else:
                                self.stats['errors'] += len(batch_points)
                                batch_points = []  # Clear batch even on failure
                        
                    except Exception as e:
                        logger.error(f"Error processing Texas permit {idx}: {e}")
                        self.stats['errors'] += 1
                        pbar.update(1)
                
                # Upload remaining points
                if batch_points:
                    if self.upload_batch_with_retry(batch_points, "Texas permits"):
                        pass  # Success
                    else:
                        self.stats['errors'] += len(batch_points)
            
        except Exception as e:
            logger.error(f"Error ingesting Texas permits: {e}")
            self.stats['errors'] += 1
    
    def ingest_oklahoma_permits(self):
        """Ingest Oklahoma drilling permits."""
        logger.info("Ingesting Oklahoma permits...")
        
        try:
            df = pd.read_csv(self.data_paths['oklahoma_permits'])
            
            # Progress bar for Oklahoma permits
            with tqdm(total=len(df), desc="Oklahoma Permits", unit="permits") as pbar:
                batch_points = []
                
                for idx, row in df.iterrows():
                    try:
                        data = row.to_dict()
                        searchable_text = self.create_searchable_text(data, 'oklahoma_permit')
                        embedding = self.create_embedding(searchable_text)
                        
                        point = PointStruct(
                            id=hash(f"oklahoma_permit_{data.get('API_Number', idx)}") % (2**63 - 1),
                            vector=embedding,
                            payload={
                                'data_type': 'oklahoma_permit',
                                'source': 'oklahoma_permits_streamlined.csv',
                                'data': data,
                                'searchable_text': searchable_text,
                                'ingested_at': datetime.now().isoformat()
                            }
                        )
                        batch_points.append(point)
                        self.stats['oklahoma_permits'] += 1
                        pbar.update(1)
                        
                        # Upload batch when it reaches batch_size
                        if len(batch_points) >= self.batch_size:
                            if self.upload_batch_with_retry(batch_points, "Oklahoma permits"):
                                batch_points = []  # Clear batch after successful upload
                            else:
                                self.stats['errors'] += len(batch_points)
                                batch_points = []  # Clear batch even on failure
                        
                    except Exception as e:
                        logger.error(f"Error processing Oklahoma permit {idx}: {e}")
                        self.stats['errors'] += 1
                        pbar.update(1)
                
                # Upload remaining points
                if batch_points:
                    if self.upload_batch_with_retry(batch_points, "Oklahoma permits"):
                        pass  # Success
                    else:
                        self.stats['errors'] += len(batch_points)
            
        except Exception as e:
            logger.error(f"Error ingesting Oklahoma permits: {e}")
            self.stats['errors'] += 1
    
    def verify_data_files(self):
        """Verify all data files exist."""
        logger.info("Verifying data files...")
        
        missing_files = []
        for name, path in self.data_paths.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
            else:
                logger.info(f"✓ Found {name}: {path}")
        
        if missing_files:
            logger.error("Missing data files:")
            for file in missing_files:
                logger.error(f"  - {file}")
            raise FileNotFoundError(f"Missing {len(missing_files)} data files")
        
        logger.info("All data files verified")
    
    def run_ingestion(self):
        """Run the complete ingestion process."""
        logger.info("Starting comprehensive data ingestion with OpenAI embeddings...")
        start_time = datetime.now()
        
        try:
            # Check OpenAI API key
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            # Verify files exist
            self.verify_data_files()
            
            # Setup collection
            self.setup_collection()
            
            # Ingest all data sources
            self.ingest_mineral_offers()
            self.ingest_lease_offers()
            self.ingest_forum_data()
            self.ingest_texas_permits()
            self.ingest_oklahoma_permits()
            
            # Calculate total
            self.stats['total_documents'] = sum([
                self.stats['mineral_offers'],
                self.stats['lease_offers'],
                self.stats['forum_posts'],
                self.stats['texas_permits'],
                self.stats['oklahoma_permits']
            ])
            
            # Print summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("=" * 60)
            logger.info("OPENAI INGESTION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration}")
            logger.info(f"Total documents: {self.stats['total_documents']}")
            logger.info(f"Mineral offers: {self.stats['mineral_offers']}")
            logger.info(f"Lease offers: {self.stats['lease_offers']}")
            logger.info(f"Forum posts: {self.stats['forum_posts']}")
            logger.info(f"Texas permits: {self.stats['texas_permits']}")
            logger.info(f"Oklahoma permits: {self.stats['oklahoma_permits']}")
            logger.info(f"API calls made: {self.stats['api_calls']}")
            logger.info(f"Estimated cost: ${self.stats['estimated_cost']:.4f}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("=" * 60)
            
            # Save stats
            with open('ingestion_stats.json', 'w') as f:
                json.dump({
                    'timestamp': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'stats': self.stats
                }, f, indent=2)
            
            logger.info("Stats saved to ingestion_stats.json")
            
        except Exception as e:
            logger.error(f"Fatal error during ingestion: {e}")
            raise

def main():
    """Main entry point."""
    try:
        ingester = MineralDataIngester()
        ingester.run_ingestion()
    except Exception as e:
        logger.error(f"Failed to run ingestion: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
