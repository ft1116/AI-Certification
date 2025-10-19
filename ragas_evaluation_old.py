#!/usr/bin/env python3
"""
RAGAS Evaluation Script for Mineral Rights LangGraph Pipeline

This script evaluates your LangGraph system using the manual test dataset
and RAGAS metrics: faithfulness, response relevance, context precision, and context recall.
"""

import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall
)
from datasets import Dataset

# LangGraph imports
from mineral_insights_langgraph import run_mineral_query
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MineralRightsRAGASEvaluator:
    """
    Evaluates the Mineral Rights LangGraph pipeline using RAGAS metrics.
    """
    
    def __init__(self, test_dataset_path: str):
        """
        Initialize the evaluator with the test dataset.
        
        Args:
            test_dataset_path: Path to the CSV test dataset
        """
        self.test_dataset_path = test_dataset_path
        self.test_data = None
        self.evaluation_results = None
        
        # Validate API keys
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        logger.info(f"Initialized RAGAS evaluator with dataset: {test_dataset_path}")
    
    def load_test_dataset(self) -> pd.DataFrame:
        """
        Load the test dataset from CSV.
        
        Returns:
            DataFrame with test questions and ground truth
        """
        logger.info("Loading test dataset...")
        
        try:
            self.test_data = pd.read_csv(self.test_dataset_path)
            logger.info(f"Loaded {len(self.test_data)} test questions")
            
            # Display dataset info
            logger.info(f"Question types: {self.test_data['question_type'].unique()}")
            logger.info(f"Difficulty distribution: {self.test_data['difficulty'].value_counts().to_dict()}")
            
            return self.test_data
            
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            raise
    
    def run_langgraph_queries(self) -> List[Dict[str, Any]]:
        """
        Run all test questions through the LangGraph pipeline.
        
        Returns:
            List of results from LangGraph queries
        """
        logger.info("Running queries through LangGraph pipeline...")
        
        results = []
        
        for idx, row in self.test_data.iterrows():
            question = row['question']
            logger.info(f"Processing question {idx + 1}/{len(self.test_data)}: {question[:50]}...")
            
            try:
                # Run the query through your LangGraph system
                result = run_mineral_query(question, conversation_id=f"eval_{idx}")
                
                # Extract the answer and sources
                answer = result.get('answer', '')
                sources = result.get('sources', [])
                confidence = result.get('confidence', 0.0)
                
                # Create context from sources (simplified for RAGAS)
                context = self._create_context_from_sources(sources, row['contexts'])
                
                results.append({
                    'question': question,
                    'answer': answer,
                    'contexts': context,
                    'ground_truth': row['ground_truth'],
                    'confidence': confidence,
                    'sources_count': len(sources),
                    'question_type': row['question_type'],
                    'difficulty': row['difficulty']
                })
                
                logger.info(f"✅ Question {idx + 1} completed (confidence: {confidence:.2f})")
                
            except Exception as e:
                logger.error(f"❌ Error processing question {idx + 1}: {e}")
                # Add empty result to maintain dataset structure
                results.append({
                    'question': question,
                    'answer': f"Error: {e}",
                    'contexts': [],
                    'ground_truth': row['ground_truth'],
                    'confidence': 0.0,
                    'sources_count': 0,
                    'question_type': row['question_type'],
                    'difficulty': row['difficulty']
                })
        
        logger.info(f"Completed {len(results)} queries through LangGraph")
        return results
    
    def _create_context_from_sources(self, sources: List[str], expected_contexts: str) -> List[str]:
        """
        Create context list from sources for RAGAS evaluation.
        
        Args:
            sources: List of source strings from LangGraph
            expected_contexts: Expected context types from test dataset
            
        Returns:
            List of context strings
        """
        # For RAGAS, we need to provide the actual retrieved documents
        # Since we can't easily access the original documents here,
        # we'll create simplified context based on the sources
        
        context = []
        for source in sources[:5]:  # Limit to top 5 sources
            if source:
                context.append(f"Source: {source}")
        
        # If no sources, add a placeholder
        if not context:
            context = ["No relevant context retrieved"]
        
        return context
    
    def prepare_ragas_dataset(self, query_results: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare the data for RAGAS evaluation.
        
        Args:
            query_results: Results from LangGraph queries
            
        Returns:
            RAGAS Dataset object
        """
        logger.info("Preparing dataset for RAGAS evaluation...")
        
        # Convert to RAGAS format - ensure contexts is a list of lists
        ragas_data = {
            'question': [r['question'] for r in query_results],
            'answer': [r['answer'] for r in query_results],
            'contexts': [r['contexts'] if isinstance(r['contexts'], list) else [r['contexts']] for r in query_results],
            'ground_truth': [r['ground_truth'] for r in query_results]
        }
        
        # Create RAGAS dataset directly from dict
        dataset = Dataset.from_dict(ragas_data)
        logger.info(f"Created RAGAS dataset with {len(dataset)} samples")
        
        return dataset
    
    def run_ragas_evaluation(self, dataset: Dataset) -> Dict[str, float]:
        """
        Run RAGAS evaluation with key metrics.
        
        Args:
            dataset: RAGAS Dataset object
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running RAGAS evaluation...")
        
        try:
            # Define metrics to evaluate
            metrics = [
                Faithfulness(),           # How faithful is the answer to the context?
                ResponseRelevancy(),      # How relevant is the response to the question?
                ContextPrecision(),       # How precise is the retrieved context?
                ContextRecall()           # How well does the context cover the ground truth?
            ]
            
            # Run evaluation with string model names
            logger.info("Starting RAGAS evaluation (this may take a few minutes)...")
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm="claude-sonnet-4-5-20250929",
                embeddings="text-embedding-3-small"
            )
            
            # Extract metrics
            evaluation_results = {
                'faithfulness': result['faithfulness'],
                'response_relevancy': result['response_relevancy'],
                'context_precision': result['context_precision'],
                'context_recall': result['context_recall']
            }
            
            logger.info("✅ RAGAS evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Error during RAGAS evaluation: {e}")
            raise
    
    def generate_evaluation_report(self, query_results: List[Dict[str, Any]], 
                                 ragas_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            query_results: Results from LangGraph queries
            ragas_metrics: RAGAS evaluation metrics
            
        Returns:
            Comprehensive evaluation report
        """
        logger.info("Generating evaluation report...")
        
        # Calculate additional metrics
        avg_confidence = sum(r['confidence'] for r in query_results) / len(query_results)
        avg_sources = sum(r['sources_count'] for r in query_results) / len(query_results)
        
        # Performance by question type
        type_performance = {}
        for q_type in pd.Series([r['question_type'] for r in query_results]).unique():
            type_results = [r for r in query_results if r['question_type'] == q_type]
            type_confidence = sum(r['confidence'] for r in type_results) / len(type_results)
            type_performance[q_type] = {
                'count': len(type_results),
                'avg_confidence': type_confidence
            }
        
        # Performance by difficulty
        difficulty_performance = {}
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in query_results if r['difficulty'] == difficulty]
            if diff_results:
                diff_confidence = sum(r['confidence'] for r in diff_results) / len(diff_results)
                difficulty_performance[difficulty] = {
                    'count': len(diff_results),
                    'avg_confidence': diff_confidence
                }
        
        # Create comprehensive report
        report = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(query_results),
                'dataset_source': self.test_dataset_path
            },
            'ragas_metrics': ragas_metrics,
            'system_performance': {
                'average_confidence': avg_confidence,
                'average_sources_per_query': avg_sources,
                'successful_queries': len([r for r in query_results if r['confidence'] > 0])
            },
            'performance_by_question_type': type_performance,
            'performance_by_difficulty': difficulty_performance,
            'detailed_results': query_results
        }
        
        return report
    
    def save_evaluation_results(self, report: Dict[str, Any], 
                              output_filename: str = None) -> Dict[str, str]:
        """
        Save evaluation results in multiple formats.
        
        Args:
            report: Evaluation report
            output_filename: Base filename for output files
            
        Returns:
            Dictionary with paths to saved files
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ragas_evaluation_results_{timestamp}"
        
        # Save as JSON
        json_path = f"{output_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Evaluation results saved as JSON: {json_path}")
        
        # Save metrics summary as CSV
        csv_path = f"{output_filename}_summary.csv"
        metrics_df = pd.DataFrame([report['ragas_metrics']])
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"✅ Metrics summary saved as CSV: {csv_path}")
        
        # Save detailed results as CSV
        detailed_csv_path = f"{output_filename}_detailed.csv"
        detailed_df = pd.DataFrame(report['detailed_results'])
        detailed_df.to_csv(detailed_csv_path, index=False)
        logger.info(f"✅ Detailed results saved as CSV: {detailed_csv_path}")
        
        return {
            'json': json_path,
            'summary_csv': csv_path,
            'detailed_csv': detailed_csv_path
        }
    
    def print_evaluation_summary(self, report: Dict[str, Any]):
        """
        Print a formatted evaluation summary.
        
        Args:
            report: Evaluation report
        """
        print("\n" + "="*80)
        print("RAGAS EVALUATION RESULTS - MINERAL RIGHTS LANGGRAPH PIPELINE")
        print("="*80)
        
        # RAGAS Metrics Table
        print("\n📊 RAGAS METRICS:")
        print("-" * 50)
        metrics = report['ragas_metrics']
        print(f"Faithfulness:        {metrics['faithfulness']:.3f}")
        print(f"Response Relevancy:  {metrics['response_relevancy']:.3f}")
        print(f"Context Precision:   {metrics['context_precision']:.3f}")
        print(f"Context Recall:      {metrics['context_recall']:.3f}")
        
        # System Performance
        print("\n🚀 SYSTEM PERFORMANCE:")
        print("-" * 50)
        perf = report['system_performance']
        print(f"Average Confidence:  {perf['average_confidence']:.3f}")
        print(f"Avg Sources/Query:   {perf['average_sources_per_query']:.1f}")
        print(f"Successful Queries:  {perf['successful_queries']}/{report['evaluation_metadata']['total_questions']}")
        
        # Performance by Difficulty
        print("\n📈 PERFORMANCE BY DIFFICULTY:")
        print("-" * 50)
        for difficulty, stats in report['performance_by_difficulty'].items():
            print(f"{difficulty.capitalize():8}: {stats['count']} questions, avg confidence: {stats['avg_confidence']:.3f}")
        
        # Performance by Question Type
        print("\n🎯 PERFORMANCE BY QUESTION TYPE:")
        print("-" * 50)
        for q_type, stats in report['performance_by_question_type'].items():
            print(f"{q_type.replace('_', ' ').title():20}: {stats['count']} questions, avg confidence: {stats['avg_confidence']:.3f}")
        
        print("\n" + "="*80)
    
    def run_full_evaluation(self):
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Evaluation report
        """
        logger.info("Starting full RAGAS evaluation pipeline...")
        
        try:
            # Step 1: Load test dataset
            self.load_test_dataset()
            
            # Step 2: Run queries through LangGraph
            query_results = self.run_langgraph_queries()
            
            # Step 3: Prepare RAGAS dataset
            ragas_dataset = self.prepare_ragas_dataset(query_results)
            
            # Step 4: Run RAGAS evaluation
            ragas_metrics = self.run_ragas_evaluation(ragas_dataset)
            
            # Step 5: Generate comprehensive report
            report = self.generate_evaluation_report(query_results, ragas_metrics)
            
            # Step 6: Save results
            saved_paths = self.save_evaluation_results(report)
            
            # Step 7: Print summary
            self.print_evaluation_summary(report)
            
            logger.info("🎉 Full evaluation pipeline completed successfully!")
            
            return {
                'report': report,
                'saved_paths': saved_paths
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation pipeline failed: {e}")
            raise


def main():
    """
    Main function to run the RAGAS evaluation.
    """
    print("RAGAS Evaluation for Mineral Rights LangGraph Pipeline")
    print("=" * 60)
    
    # Path to your test dataset
    test_dataset_path = "mineral_rights_manual_test_dataset_20251019_071012.csv"
    
    try:
        # Initialize evaluator
        evaluator = MineralRightsRAGASEvaluator(test_dataset_path)
        
        # Run full evaluation
        results = evaluator.run_full_evaluation()
        
        print(f"\n💾 Results saved to:")
        for format_type, path in results['saved_paths'].items():
            print(f"   - {format_type.upper()}: {path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Run the evaluation
    results = main()
    
    if results:
        print("\n🎉 RAGAS evaluation completed successfully!")
        print("Check the saved files for detailed results and metrics.")
    else:
        print("\n💥 RAGAS evaluation failed. Check the logs for details.")
