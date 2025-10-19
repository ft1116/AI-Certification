# Mineral Insights Data Ingestion

This directory contains scripts to ingest all mineral insights data into a Qdrant vector database for use with LangGraph.

## Data Sources

The ingestion script processes the following data sources:

1. **mineral_offers.csv** - Mineral rights purchase offers with location, price, and details
2. **lease_offers.csv** - Oil & gas lease offers with terms, operators, and locations  
3. **forum_enhanced.json** - Forum discussions with structured posts and metadata
4. **texas_permits_20251004.csv** - Texas drilling permits with location and operator data
5. **itd_filtered_20251001.csv** - Oklahoma drilling permits with detailed well information

## Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Qdrant** running (see setup below)
3. **OpenAI API key** (see setup below)
4. All data files present in the correct locations

### Setup OpenAI API Key

1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Setup Qdrant

#### Option 1: Docker (Recommended)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Option 2: Local Installation
```bash
pip install qdrant-client[fastembed]
python -m qdrant.http
```

### Run Ingestion

#### Automated (Recommended)
```bash
./run_ingestion.sh
```

#### Manual
```bash
# Install dependencies
pip install -r requirements_ingestion.txt

# Run tests
python test_ingestion.py

# Run full ingestion
python ingest_all_data_to_qdrant.py
```

## Scripts Overview

### `ingest_all_data_to_qdrant.py`
Main ingestion script that:
- Creates embeddings using OpenAI text-embedding-3-small (1536 dimensions)
- Processes all data sources into searchable text
- Uploads to Qdrant with structured metadata
- Provides detailed logging, error handling, and cost tracking

### `test_ingestion.py`
Test script that verifies:
- Qdrant connection and OpenAI API access
- Data file availability
- Small sample ingestion
- OpenAI embedding generation

### `run_ingestion.sh`
Automated script that:
- Sets up virtual environment
- Installs dependencies
- Runs tests
- Executes full ingestion
- Provides user feedback

## Data Structure

Each document in Qdrant contains:

```json
{
  "id": "unique_identifier",
  "vector": [384-dimensional embedding],
  "payload": {
    "data_type": "mineral_offer|lease_offer|forum_post|texas_permit|oklahoma_permit",
    "source": "source_file_name",
    "data": {original_data_object},
    "searchable_text": "processed_searchable_text",
    "ingested_at": "timestamp"
  }
}
```

## Searchable Text Format

The script creates optimized searchable text for each data type:

- **Mineral Offers**: Location, price, buyer, details, content
- **Lease Offers**: Location, operator, bonus, royalty, terms, content
- **Forum Posts**: Title, category, author, content, engagement metrics
- **Texas Permits**: Operator, well, location, purpose, profile
- **Oklahoma Permits**: Operator, well, location, type, status, formation

## Usage with LangGraph

After ingestion, you can query the data in LangGraph:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Connect to Qdrant
client = QdrantClient("http://localhost:6333")

# Search for mineral offers in Texas
results = client.search(
    collection_name="mineral_insights",
    query_vector=embedding_vector,  # Your query embedding
    query_filter=Filter(
        must=[
            FieldCondition(
                key="payload.data_type",
                match=MatchValue(value="mineral_offer")
            ),
            FieldCondition(
                key="payload.data.state",
                match=MatchValue(value="texas")
            )
        ]
    ),
    limit=10
)
```

## Monitoring

The ingestion process provides:
- Real-time logging to console and `ingestion.log`
- Progress statistics
- Error tracking
- Final summary with counts
- Stats saved to `ingestion_stats.json`

## Troubleshooting

### Common Issues

1. **OpenAI API key not set**
   - Set environment variable: `export OPENAI_API_KEY="your-key"`
   - Get API key from: https://platform.openai.com/api-keys

2. **Qdrant not running**
   - Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
   - Check connection: `curl http://localhost:6333/collections`

3. **Missing data files**
   - Verify all files exist in expected locations
   - Check file permissions

4. **API rate limits**
   - OpenAI has rate limits (requests per minute)
   - The script includes progress logging to monitor status
   - Consider running during off-peak hours

5. **Cost monitoring**
   - Monitor costs in the logs and stats file
   - Total cost should be under $1 for your dataset

### Logs

- **Console**: Real-time progress and errors
- **ingestion.log**: Detailed log file
- **ingestion_stats.json**: Final statistics

## Performance & Costs

Expected ingestion times:
- Mineral offers (~14 records): < 1 minute
- Lease offers (~48 records): < 1 minute  
- Forum data (~200k+ posts): 20-60 minutes
- Texas permits (~5k records): 5-15 minutes
- Oklahoma permits (~638 records): 1-3 minutes

**Total estimated time: 30-80 minutes** depending on API rate limits.

**Estimated costs** (using OpenAI text-embedding-3-small at $0.02/million tokens):
- Mineral offers: ~$0.0001
- Lease offers: ~$0.0005
- Forum data: ~$0.32
- Texas permits: ~$0.14
- Oklahoma permits: ~$0.01

**Total estimated cost: ~$0.47** (less than 50 cents!)

## Next Steps

After successful ingestion:

1. **Verify data**: Check Qdrant web UI at http://localhost:6333/dashboard
2. **Test queries**: Use the test script to verify search functionality
3. **Integrate with LangGraph**: Use the collection in your LangGraph workflows
4. **Monitor usage**: Track query performance and optimize as needed

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Run the test script to isolate problems
3. Verify all prerequisites are met
4. Check Qdrant documentation for advanced configuration
