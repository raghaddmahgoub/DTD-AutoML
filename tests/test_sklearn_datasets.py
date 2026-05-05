"""
Quick test script for scikit-learn datasets with AutoML agent.
Generates detailed reports with terminal outputs, model information, and reasoning.
"""
import sys
import os
from pathlib import Path

# --- FIX PATHING ---
# If this file is in GP/tests/test_sklearn_datasets.py, 
# we need to point to GP/ as the project root.
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent # Goes from tests/ up to GP/
sys.path.insert(0, str(PROJECT_ROOT))
# -------------------

import io
import json
from datetime import datetime
import traceback
from typing import Optional

# Now imports should work
try:
    from agents.automl_agent.automl_agent import AutoMLAgent
    # If AgentState is not exportable directly, we define a local type or skip
    from src.utils.logger import Logger
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

logger = Logger()

# Global report storage
test_report = {
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'datasets_tested': [],
    'summary': {
        'total_tests': 0,
        'successful': 0,
        'failed': 0
    }
}

class TestAutoMLAgent(AutoMLAgent):
    """
    Subclass of AutoMLAgent to bypass manual steps if your 
    base agent uses a human-in-the-loop pattern.
    """
    def human_approval_node(self, state):
        print("\n" + "="*60)
        print("AUTO-APPROVING TRAINING PLAN (TEST MODE)")
        print("="*60)
        state['human_approved'] = True
        return state

def format_confusion_matrix_markdown(cm):
    if not cm or not isinstance(cm, list) or len(cm) < 2:
        return "N/A"
    table = [
        "| | Predicted: 0 | Predicted: 1 |",
        "|---|---|---|",
        f"| **Actual: 0** | {cm[0][0]} | {cm[0][1]} |",
        f"| **Actual: 1** | {cm[1][0]} | {cm[1][1]} |"
    ]
    return "\n".join(table)

def test_dataset(data_path, target_column, dataset_name):
    print("\n" + "="*80)
    print(f"TESTING: {dataset_name}")
    print("="*80)
    
    dataset_result = {
        'name': dataset_name,
        'path': data_path,
        'target_column': target_column,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'unknown',
        'error': None,
        'results': {}
    }
    
    # Resolve relative paths to absolute paths based on PROJECT_ROOT
    full_path = PROJECT_ROOT / data_path
    
    if not full_path.exists():
        error_msg = f"Dataset not found at: {full_path}"
        print(error_msg)
        dataset_result['status'] = 'error'
        dataset_result['error'] = error_msg
        test_report['datasets_tested'].append(dataset_result)
        test_report['summary']['failed'] += 1
        return dataset_result
    
    try:
        agent = TestAutoMLAgent()
        # Using .run() as per your previous code structure
        results = agent.run(
            data_path=str(full_path),
            target_column=target_column,
            output_dir="output/test_runs"
        )
        
        if isinstance(results, dict) and results.get('error'):
            raise Exception(results['error'])
        
        dataset_result['status'] = 'success'
        dataset_result['results'] = results
        
        # Performance Printout
        metrics = results.get('model_metrics', {})
        print(f"✅ Success! Best Model: {metrics.get('best_model')}")
        print(f"📈 Score: {metrics.get('best_score', 0):.4f}")
        
        test_report['datasets_tested'].append(dataset_result)
        test_report['summary']['successful'] += 1
        
    except Exception as e:
        print(f"❌ Error testing {dataset_name}: {str(e)}")
        dataset_result['status'] = 'error'
        dataset_result['error'] = str(e)
        dataset_result['traceback'] = traceback.format_exc()
        test_report['datasets_tested'].append(dataset_result)
        test_report['summary']['failed'] += 1
    
    test_report['summary']['total_tests'] += 1

def save_report():
    report_dir = PROJECT_ROOT / "test_reports"
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f"test_report_{timestamp}.md"
    
    # (Generating simple MD content for brevity)
    content = [f"# AutoML Test Report - {timestamp}", ""]
    for d in test_report['datasets_tested']:
        status_icon = "✅" if d['status'] == 'success' else "❌"
        content.append(f"## {status_icon} {d['name']}")
        if d['status'] == 'error':
            content.append(f"**Error:** {d['error']}")
        else:
            m = d['results'].get('model_metrics', {})
            content.append(f"- Best Model: {m.get('best_model')}")
            content.append(f"- Score: {m.get('best_score')}")
    
    with open(report_file, 'w') as f:
        f.write("\n".join(content))
    print(f"\n📁 Detailed Markdown Report saved to: {report_file}")

def main():
    test_datasets = [
        {
            'path': 'assets/data/Classification Datasets/Titanic-Dataset.csv',
            'target': 'Survived',
            'name': 'Titanic (Classification)'
        }
    ]
    
    for ds in test_datasets:
        test_dataset(ds['path'], ds['target'], ds['name'])
    
    save_report()

if __name__ == "__main__":
    main()