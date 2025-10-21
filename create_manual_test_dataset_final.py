#!/usr/bin/env python3
"""
Manual Test Dataset Creator for Mineral Rights RAG System

This creates a simple test dataset without using RAGAS to avoid API rate limits.
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

def create_manual_test_dataset():
    """Create a manual test dataset with realistic mineral rights questions."""
    
    test_questions = [
        {
            "question": "What are typical lease terms for oil and gas drilling in Texas?",
            "ground_truth": "Oil and gas lease terms in Texas typically include a primary term of 3-5 years, royalty rates of 12.5-25%, bonus payments ranging from $100-$5,000 per acre, and various clauses for drilling obligations, shut-in provisions, and surface use agreements.",
            "contexts": ["texas_permits", "lease_offers", "forum_discussions"],
            "difficulty": "medium",
            "question_type": "factual"
        },
        {
            "question": "Show me Pioneer Natural Resources drilling activity in Midland County",
            "ground_truth": "Pioneer Natural Resources has significant drilling activity in Midland County, Texas, with multiple permits for horizontal drilling in the Wolfcamp and Spraberry formations.",
            "contexts": ["texas_permits"],
            "difficulty": "easy",
            "question_type": "specific_search"
        },
        {
            "question": "What's the market price for mineral rights in Oklahoma?",
            "ground_truth": "Mineral rights prices in Oklahoma vary by location and formation, typically ranging from $500-$3,000 per acre for producing areas, with higher prices in the SCOOP/STACK plays.",
            "contexts": ["mineral_offers", "oklahoma_permits"],
            "difficulty": "medium",
            "question_type": "market_information"
        },
        {
            "question": "How do drilling permits work in Texas?",
            "ground_truth": "Texas drilling permits are issued by the Railroad Commission (RRC) and require operators to submit applications with well location, depth, formation targets, and environmental compliance information.",
            "contexts": ["texas_permits", "forum_discussions"],
            "difficulty": "easy",
            "question_type": "procedural"
        },
        {
            "question": "Compare lease offers between Texas and Oklahoma",
            "ground_truth": "Texas lease offers typically have higher bonus payments ($1,000-$5,000/acre) but similar royalty rates (12.5-25%), while Oklahoma offers may have lower bonuses but competitive royalty rates, with both states having similar primary terms of 3-5 years.",
            "contexts": ["lease_offers", "texas_permits", "oklahoma_permits"],
            "difficulty": "hard",
            "question_type": "comparative"
        },
        {
            "question": "What formations are being targeted in Oklahoma drilling permits?",
            "ground_truth": "Oklahoma drilling permits target various formations including the Woodford Shale, Mississippian Lime, Hunton, and Meramec formations, with the Woodford being particularly active in the SCOOP/STACK plays.",
            "contexts": ["oklahoma_permits"],
            "difficulty": "medium",
            "question_type": "geological"
        },
        {
            "question": "What are the current trends in mineral rights transactions?",
            "ground_truth": "Current trends in mineral rights transactions include increased consolidation by major operators, higher valuations in core areas, and growing interest in ESG-compliant operations.",
            "contexts": ["forum_discussions", "mineral_offers"],
            "difficulty": "hard",
            "question_type": "trend_analysis"
        },
        {
            "question": "How do royalty rates vary by formation in Texas?",
            "ground_truth": "Royalty rates in Texas vary by formation, with Eagle Ford typically commanding 20-25%, Permian Basin 15-20%, and Barnett Shale 12.5-18%, depending on location and operator.",
            "contexts": ["texas_permits", "lease_offers"],
            "difficulty": "medium",
            "question_type": "detailed_factual"
        },
        {
            "question": "What are the environmental considerations for drilling permits?",
            "ground_truth": "Environmental considerations for drilling permits include water protection, air quality compliance, wildlife habitat protection, and proper waste disposal, with specific requirements varying by state and formation.",
            "contexts": ["forum_discussions", "texas_permits", "oklahoma_permits"],
            "difficulty": "medium",
            "question_type": "regulatory"
        },
        {
            "question": "Show me recent drilling activity in the Permian Basin",
            "ground_truth": "Recent Permian Basin drilling activity shows continued horizontal drilling in the Wolfcamp and Spraberry formations, with major operators like Pioneer, EOG, and Chevron leading development.",
            "contexts": ["texas_permits", "lease_offers"],
            "difficulty": "easy",
            "question_type": "activity_search"
        }
    ]
    
    # Add metadata
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_questions": len(test_questions),
            "domain": "mineral_rights",
            "difficulty_distribution": {
                "easy": len([q for q in test_questions if q["difficulty"] == "easy"]),
                "medium": len([q for q in test_questions if q["difficulty"] == "medium"]),
                "hard": len([q for q in test_questions if q["difficulty"] == "hard"])
            },
            "question_types": list(set([q["question_type"] for q in test_questions]))
        },
        "questions": test_questions
    }
    
    return dataset

def save_dataset(dataset: Dict[str, Any], base_filename: str = None):
    """Save the dataset in multiple formats."""
    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"mineral_rights_manual_test_dataset_{timestamp}"
    
    # Save as JSON
    json_path = f"{base_filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"✅ Dataset saved as JSON: {json_path}")
    
    # Save as CSV
    csv_path = f"{base_filename}.csv"
    df = pd.DataFrame(dataset["questions"])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ Dataset saved as CSV: {csv_path}")
    
    return {
        'json': json_path,
        'csv': csv_path
    }

def main():
    """Create and save the manual test dataset."""
    print("Creating Manual Test Dataset for Mineral Rights RAG System")
    print("=" * 60)
    
    # Create dataset
    dataset = create_manual_test_dataset()
    
    # Save dataset
    saved_paths = save_dataset(dataset)
    
    print(f"\n📊 Dataset Summary:")
    print(f"Total questions: {dataset['metadata']['total_questions']}")
    print(f"Difficulty distribution: {dataset['metadata']['difficulty_distribution']}")
    print(f"Question types: {dataset['metadata']['question_types']}")
    
    print(f"\n💾 Files created:")
    for format_type, path in saved_paths.items():
        print(f"   - {format_type.upper()}: {path}")
    
    print(f"\n🎉 Manual test dataset created successfully!")
    print("You can now use this dataset to evaluate your RAG system without API rate limits.")

if __name__ == "__main__":
    main()

