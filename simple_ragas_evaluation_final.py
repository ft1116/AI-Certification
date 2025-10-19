"""
Simplified RAGAS Evaluation for LangGraph Pipeline
Works with current dependencies and provides comprehensive evaluation
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Import your LangGraph pipeline
from mineral_insights_langgraph import run_mineral_query

# Load environment
load_dotenv()

# Set up LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mineral-rights-simple-ragas"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

def create_test_dataset():
    """Load your original manual test dataset from CSV file"""
    
    # Load from your existing CSV file
    csv_file = "mineral_rights_manual_test_dataset_20251019_071012.csv"
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Test dataset file not found: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert to the format expected by the evaluation
    test_questions = []
    for _, row in df.iterrows():
        test_questions.append({
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "category": row["question_type"]  # Use question_type as category
        })
    
    print(f"📋 Loaded {len(test_questions)} questions from {csv_file}")
    return test_questions

def simple_similarity_score(text1, text2):
    """Simple similarity scoring based on common words"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def evaluate_faithfulness(answer, context_sources):
    """Evaluate if answer is based on provided context"""
    if not context_sources or not answer:
        return 0.0
    
    # Simple check: if answer mentions sources or contains context keywords
    context_text = " ".join(context_sources).lower()
    answer_lower = answer.lower()
    
    # Check for source mentions
    source_mentions = 0
    for source in context_sources:
        if any(word in answer_lower for word in source.lower().split()[:3]):
            source_mentions += 1
    
    # Check for context keyword overlap
    context_words = set(context_text.split())
    answer_words = set(answer_lower.split())
    keyword_overlap = len(context_words.intersection(answer_words)) / len(context_words) if context_words else 0
    
    # Combine metrics
    faithfulness_score = (source_mentions / len(context_sources)) * 0.6 + keyword_overlap * 0.4
    return min(1.0, faithfulness_score)

def evaluate_answer_relevancy(question, answer):
    """Evaluate if answer addresses the question"""
    if not question or not answer:
        return 0.0
    
    # Extract key terms from question
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    
    # Remove common words
    common_words = {'what', 'how', 'where', 'when', 'why', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    question_keywords = question_words - common_words
    answer_keywords = answer_words - common_words
    
    if not question_keywords:
        return 0.5  # Neutral if no keywords
    
    # Check keyword overlap
    overlap = len(question_keywords.intersection(answer_keywords))
    relevancy_score = overlap / len(question_keywords)
    
    return min(1.0, relevancy_score)

def evaluate_context_precision(question, context_sources):
    """Evaluate if retrieved context is relevant to the question"""
    if not question or not context_sources:
        return 0.0
    
    question_words = set(question.lower().split())
    relevant_sources = 0
    
    for source in context_sources:
        source_words = set(source.lower().split())
        # Check if source contains question keywords
        if question_words.intersection(source_words):
            relevant_sources += 1
    
    return relevant_sources / len(context_sources) if context_sources else 0.0

def evaluate_context_recall(question, ground_truth, context_sources):
    """Evaluate if context contains information needed to answer the question"""
    if not question or not ground_truth or not context_sources:
        return 0.0
    
    # Extract key concepts from ground truth
    gt_words = set(ground_truth.lower().split())
    context_text = " ".join(context_sources).lower()
    context_words = set(context_text.split())
    
    # Basic word overlap
    covered_concepts = len(gt_words.intersection(context_words))
    total_concepts = len(gt_words)
    basic_recall = covered_concepts / total_concepts if total_concepts > 0 else 0.0
    
    # Enhanced recall: Check for semantic concepts
    semantic_bonus = 0.0
    
    # Check for price ranges (e.g., $500-$2000 vs $500, $2000-3000)
    import re
    gt_prices = re.findall(r'\$[\d,]+(?:-\$?[\d,]+)?', ground_truth.lower())
    context_prices = re.findall(r'\$[\d,]+(?:-\$?[\d,]+)?', context_text)
    
    if gt_prices and context_prices:
        # If both have prices, give bonus
        semantic_bonus += 0.2
    
    # Check for percentage ranges (e.g., 20-25% vs 20%, 22.5%)
    gt_percentages = re.findall(r'(\d+(?:\.\d+)?)-?(\d+(?:\.\d+)?)%', ground_truth.lower())
    context_percentages = re.findall(r'(\d+(?:\.\d+)?)-?(\d+(?:\.\d+)?)%', context_text)
    
    if gt_percentages and context_percentages:
        # If both have percentages, give bonus
        semantic_bonus += 0.2
    
    # Check for location matches
    location_words = ['county', 'state', 'oklahoma', 'texas', 'leon', 'howard', 'ector', 'rusk']
    gt_locations = [word for word in gt_words if word in location_words]
    context_locations = [word for word in context_words if word in location_words]
    
    if gt_locations and context_locations:
        location_match = len(set(gt_locations).intersection(set(context_locations))) / len(gt_locations)
        semantic_bonus += location_match * 0.3
    
    # Check for key mineral rights terms
    mineral_terms = ['lease', 'royalty', 'bonus', 'acre', 'drilling', 'operator', 'mineral']
    gt_mineral = [word for word in gt_words if word in mineral_terms]
    context_mineral = [word for word in context_words if word in mineral_terms]
    
    if gt_mineral and context_mineral:
        mineral_match = len(set(gt_mineral).intersection(set(context_mineral))) / len(gt_mineral)
        semantic_bonus += mineral_match * 0.3
    
    # Combine basic recall with semantic bonus
    final_recall = min(1.0, basic_recall + semantic_bonus)
    
    return final_recall

def run_langgraph_evaluation():
    """Run comprehensive evaluation on LangGraph pipeline"""
    
    print("="*80)
    print("🧪 RAGAS-STYLE EVALUATION - LangGraph Pipeline")
    print("="*80)
    
    # Create test dataset
    test_data = create_test_dataset()
    print(f"\n📋 Testing {len(test_data)} questions across multiple categories...")
    print("This will take about 5-10 minutes...\n")
    
    # Prepare data for evaluation
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    categories = []
    confidence_scores = []
    
    # Metrics
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []
    
    # Create progress bar
    progress_bar = tqdm(test_data, desc="Evaluating LangGraph", unit="question")
    
    for i, test_item in enumerate(progress_bar, 1):
        question = test_item["question"]
        ground_truth = test_item["ground_truth"]
        category = test_item["category"]
        
        # Update progress bar description
        progress_bar.set_description(f"Evaluating {category.upper()}")
        progress_bar.set_postfix({
            'Question': f"{question[:30]}...",
            'Completed': f"{i-1}/{len(test_data)}"
        })
        
        try:
            # Run through LangGraph pipeline
            result = run_mineral_query(question, conversation_id=f"ragas_test_{i}")
            
            # Extract results
            answer = result.get("answer", "No answer generated")
            confidence = result.get("confidence", 0.0)
            sources = result.get("sources", [])
            
            # Store for evaluation
            questions.append(question)
            answers.append(answer)
            contexts.append(sources)
            ground_truths.append(ground_truth)
            categories.append(category)
            confidence_scores.append(confidence)
            
            # Calculate metrics
            faithfulness = evaluate_faithfulness(answer, sources)
            relevancy = evaluate_answer_relevancy(question, answer)
            precision = evaluate_context_precision(question, sources)
            recall = evaluate_context_recall(question, ground_truth, sources)
            
            faithfulness_scores.append(faithfulness)
            relevancy_scores.append(relevancy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            # Update progress bar with results
            progress_bar.set_postfix({
                'Confidence': f"{confidence:.3f}",
                'Sources': len(sources),
                'F': f"{faithfulness:.3f}",
                'R': f"{relevancy:.3f}",
                'P': f"{precision:.3f}",
                'Rec': f"{recall:.3f}"
            })
            
        except Exception as e:
            progress_bar.set_postfix({
                'Error': str(e)[:20] + "...",
                'Status': 'Failed'
            })
            # Add empty results for failed queries
            questions.append(question)
            answers.append("Error: Failed to generate answer")
            contexts.append(["Error: No context available"])
            ground_truths.append(ground_truth)
            categories.append(category)
            confidence_scores.append(0.0)
            faithfulness_scores.append(0.0)
            relevancy_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
    
    # Close progress bar
    progress_bar.close()
    
    return {
        'questions': questions,
        'answers': answers,
        'contexts': contexts,
        'ground_truths': ground_truths,
        'categories': categories,
        'confidence_scores': confidence_scores,
        'faithfulness_scores': faithfulness_scores,
        'relevancy_scores': relevancy_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores
    }

def analyze_results(results):
    """Analyze and display evaluation results"""
    
    print("\n" + "="*80)
    print("🎯 RAGAS-STYLE EVALUATION RESULTS")
    print("="*80)
    
    # Calculate overall scores
    faithfulness_score = sum(results['faithfulness_scores']) / len(results['faithfulness_scores'])
    relevancy_score = sum(results['relevancy_scores']) / len(results['relevancy_scores'])
    precision_score = sum(results['precision_scores']) / len(results['precision_scores'])
    recall_score = sum(results['recall_scores']) / len(results['recall_scores'])
    
    # Create results table
    results_table = pd.DataFrame({
        'Metric': ['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall'],
        'Score': [faithfulness_score, relevancy_score, precision_score, recall_score],
        'Grade': [
            'A' if faithfulness_score > 0.8 else 'B' if faithfulness_score > 0.6 else 'C' if faithfulness_score > 0.4 else 'D',
            'A' if relevancy_score > 0.8 else 'B' if relevancy_score > 0.6 else 'C' if relevancy_score > 0.4 else 'D',
            'A' if precision_score > 0.8 else 'B' if precision_score > 0.6 else 'C' if precision_score > 0.4 else 'D',
            'A' if recall_score > 0.8 else 'B' if recall_score > 0.6 else 'C' if recall_score > 0.4 else 'D'
        ],
        'Interpretation': [
            'Excellent - Rarely hallucinates' if faithfulness_score > 0.8 else 'Good - Some hallucination' if faithfulness_score > 0.6 else 'Poor - Frequently makes things up',
            'Excellent - Answers on-topic' if relevancy_score > 0.8 else 'Good - Mostly relevant' if relevancy_score > 0.6 else 'Poor - Often off-topic',
            'Excellent - Retrieved docs very relevant' if precision_score > 0.8 else 'Good - Some irrelevant docs' if precision_score > 0.6 else 'Poor - Many irrelevant docs',
            'Excellent - Found most relevant info' if recall_score > 0.8 else 'Good - Missing some info' if recall_score > 0.6 else 'Poor - Missing lots of info'
        ]
    })
    
    print("\n📊 OVERALL PERFORMANCE TABLE")
    print("="*80)
    print(results_table.to_string(index=False, float_format='%.3f'))
    
    # Category-wise analysis
    print("\n📈 CATEGORY-WISE ANALYSIS")
    print("="*80)
    
    category_df = pd.DataFrame({
        'Category': results['categories'],
        'Faithfulness': results['faithfulness_scores'],
        'Answer Relevancy': results['relevancy_scores'],
        'Context Precision': results['precision_scores'],
        'Context Recall': results['recall_scores'],
        'Confidence': results['confidence_scores']
    })
    
    category_summary = category_df.groupby('Category').agg({
        'Faithfulness': 'mean',
        'Answer Relevancy': 'mean',
        'Context Precision': 'mean',
        'Context Recall': 'mean',
        'Confidence': 'mean'
    }).round(3)
    
    print(category_summary.to_string())
    
    # Detailed results
    print("\n📋 DETAILED RESULTS")
    print("="*80)
    
    detailed_df = pd.DataFrame({
        'Question': [q[:50] + '...' if len(q) > 50 else q for q in results['questions']],
        'Category': results['categories'],
        'Faithfulness': results['faithfulness_scores'],
        'Answer Relevancy': results['relevancy_scores'],
        'Context Precision': results['precision_scores'],
        'Context Recall': results['recall_scores'],
        'Confidence': results['confidence_scores']
    })
    
    print(detailed_df.to_string(index=False, float_format='%.3f'))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_ragas_evaluation_{timestamp}.json"
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "overall_scores": {
            "faithfulness": float(faithfulness_score),
            "answer_relevancy": float(relevancy_score),
            "context_precision": float(precision_score),
            "context_recall": float(recall_score)
        },
        "category_scores": category_summary.to_dict(),
        "detailed_results": detailed_df.to_dict('records'),
        "test_questions": results['questions'],
        "answers": results['answers'],
        "ground_truths": results['ground_truths'],
        "categories": results['categories'],
        "confidence_scores": results['confidence_scores'],
        "num_questions": len(results['questions'])
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("="*80)
    
    if faithfulness_score < 0.7:
        print("🔴 Faithfulness is low - Consider improving context quality and reducing hallucination")
    
    if relevancy_score < 0.7:
        print("🔴 Answer relevancy is low - Review prompt engineering and answer generation")
    
    if precision_score < 0.7:
        print("🔴 Context precision is low - Improve retrieval strategy and document filtering")
    
    if recall_score < 0.7:
        print("🔴 Context recall is low - Increase retrieval count or improve document coverage")
    
    if all(score > 0.7 for score in [faithfulness_score, relevancy_score, precision_score, recall_score]):
        print("🟢 All metrics are good! Your LangGraph pipeline is performing well.")
    
    print("\n✅ Evaluation complete!")
    print("="*80)
    
    return results_data

def main():
    """Main evaluation function"""
    try:
        # Run evaluation
        results = run_langgraph_evaluation()
        
        # Analyze results
        results_data = analyze_results(results)
        
        return results_data
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    main()
