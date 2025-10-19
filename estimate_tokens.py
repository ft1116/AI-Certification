#!/usr/bin/env python3
"""
Estimate token usage for full ingestion.
"""

import pandas as pd
import json
from pathlib import Path

def estimate_tokens_for_text(text):
    """Rough estimate: 1 token ≈ 0.75 words, 1 word ≈ 1.3 tokens"""
    if not text or pd.isna(text):
        return 0
    words = len(str(text).split())
    return int(words * 1.3)

def estimate_file_tokens(file_path, data_type):
    """Estimate tokens for a data file."""
    print(f"\n📊 Analyzing {file_path.name}...")
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        total_tokens = 0
        
        for idx, row in df.iterrows():
            # Create searchable text (similar to ingestion script)
            if data_type == 'mineral_offer':
                text = f"Mineral rights for sale in {row.get('county', '')} County, {row.get('state', '')}. Price: {row.get('price_text', '')}. Details: {row.get('additional_details', '')}"
            elif data_type == 'lease_offer':
                text = f"Lease offer in {row.get('county', '')} County, {row.get('state', '')}. Bonus: {row.get('bonus', '')}. Royalty: {row.get('royalty', '')}. Terms: {row.get('terms', '')}"
            elif data_type == 'texas_permit':
                text = f"Texas drilling permit for {row.get('operator', '')} in {row.get('county', '')} County. Well: {row.get('well_name', '')}"
            elif data_type == 'oklahoma_permit':
                text = f"Oklahoma drilling permit for {row.get('operator', '')} in {row.get('county', '')} County. Well: {row.get('well_name', '')}"
            else:
                text = str(row.to_dict())
            
            total_tokens += estimate_tokens_for_text(text)
        
        print(f"   Records: {len(df)}")
        print(f"   Estimated tokens: {total_tokens:,}")
        return total_tokens, len(df)
    
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        total_tokens = 0
        total_records = 0
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Forum topic
                    title = item.get('title', '')
                    content = item.get('content', '')
                    text = f"Forum topic: {title}. Content: {content}"
                    total_tokens += estimate_tokens_for_text(text)
                    total_records += 1
                    
                    # Forum posts within topic
                    posts = item.get('posts', [])
                    for post in posts:
                        if isinstance(post, dict):
                            author = post.get('author', '')
                            post_content = post.get('content', '')
                            post_text = f"Forum post by {author}: {post_content}"
                            total_tokens += estimate_tokens_for_text(post_text)
                            total_records += 1
        
        print(f"   Records: {total_records}")
        print(f"   Estimated tokens: {total_tokens:,}")
        return total_tokens, total_records
    
    return 0, 0

def main():
    """Calculate total token usage."""
    print("🔍 Estimating Token Usage for Full Ingestion")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    
    files_to_analyze = [
        ('current_mineral_offers_20251018_095734.csv', 'mineral_offer'),
        ('current_lease_offers_20251018_095726.csv', 'lease_offer'),
        ('texas_permits_20251004.csv', 'texas_permit'),
        ('itd_filtered_20251001.csv', 'oklahoma_permit'),
        ('forum_enhanced.json', 'forum')
    ]
    
    total_tokens = 0
    total_records = 0
    
    for filename, data_type in files_to_analyze:
        file_path = base_path / filename
        if file_path.exists():
            tokens, records = estimate_file_tokens(file_path, data_type)
            total_tokens += tokens
            total_records += records
        else:
            print(f"⚠️  File not found: {filename}")
    
    print("\n" + "=" * 60)
    print("📊 TOTAL ESTIMATION")
    print("=" * 60)
    print(f"Total records: {total_records:,}")
    print(f"Total estimated tokens: {total_tokens:,}")
    
    # Cost calculation
    # text-embedding-3-small: $0.00002 per 1K tokens
    cost_per_1k_tokens = 0.00002
    total_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    print(f"Estimated cost: ${total_cost:.4f}")
    
    # Token limits
    print(f"\n💳 OpenAI Limits:")
    print(f"   Free tier: 3M tokens/month")
    print(f"   Pay-as-you-go: No monthly limit")
    print(f"   Your usage: {total_tokens:,} tokens ({total_tokens/1000000:.2f}M)")
    
    if total_tokens < 3000000:
        print(f"\n✅ You're within free tier limits!")
    else:
        print(f"\n💰 You'll need pay-as-you-go billing for this volume")
    
    print(f"\n🎯 Recommendation:")
    if total_cost < 1.0:
        print(f"   Very affordable! Go ahead with full ingestion.")
    elif total_cost < 10.0:
        print(f"   Reasonable cost. Consider proceeding.")
    else:
        print(f"   Higher cost. You might want to process in batches.")

if __name__ == "__main__":
    main()

