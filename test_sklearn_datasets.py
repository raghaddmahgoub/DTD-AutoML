"""
Quick test script for scikit-learn datasets with AutoML agent.
Generates detailed reports with terminal outputs, model information, and reasoning.
"""
import sys
import io
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import traceback

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.automl_agent import AutoMLAgent, AgentState
from src.utils.logger import Logger

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

# Capture terminal output
terminal_output = []

class TeeOutput:
    """Class to capture output while still printing to console."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
        terminal_output.append(obj)
    
    def flush(self):
        for f in self.files:
            f.flush()

def test_dataset(data_path, target_column, dataset_name):
    """Test a single dataset and collect all information."""
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
        'results': {},
        'terminal_output': []
    }
    
    if not Path(data_path).exists():
        error_msg = f"Dataset not found: {data_path}"
        print(error_msg)
        dataset_result['status'] = 'error'
        dataset_result['error'] = error_msg
        test_report['datasets_tested'].append(dataset_result)
        test_report['summary']['failed'] += 1
        return dataset_result
    
    try:
        # Create agent with auto-approval for testing
        class TestAutoMLAgent(AutoMLAgent):
            """AutoML Agent that auto-approves for testing."""
            def human_approval_node(self, state: AgentState) -> AgentState:
                print("\n" + "="*60)
                print("AUTO-APPROVING TRAINING PLAN (TEST MODE)")
                print("="*60)
                state['human_approved'] = True
                state['step'] = 'human_approval_complete'
                return state
        
        agent = TestAutoMLAgent()
        results = agent.run(
            data_path=data_path,
            target_column=target_column
        )
        
        if results.get('error'):
            error_msg = f"Error: {results['error']}"
            print(error_msg)
            dataset_result['status'] = 'error'
            dataset_result['error'] = results['error']
            test_report['datasets_tested'].append(dataset_result)
            test_report['summary']['failed'] += 1
            return dataset_result
        
        # Collect all results
        dataset_result['status'] = 'success'
        dataset_result['results'] = {
            'step': results.get('step', 'unknown'),
            'problem_type': results.get('problem_type', 'N/A'),
            'target_column': results.get('target_column', 'N/A'),
            'use_automl': results.get('use_automl', None),
            'data_summary': results.get('data_summary', {}),
            'selected_models': results.get('selected_models', []),
            'automl_config': results.get('automl_config', {}),
            'model_metrics': results.get('model_metrics', {}),
            'data_analysis_reasoning': results.get('data_analysis_reasoning', ''),
            'model_selection_reasoning': results.get('model_selection_reasoning', ''),
            'reasoning': results.get('reasoning', '')
        }
        
        # Print results
        print(f"\nStatus: {results.get('step', 'unknown')}")
        print(f"Problem Type: {results.get('problem_type', 'N/A')}")
        print(f"Target Column: {results.get('target_column', 'N/A')}")
        print(f"Use AutoML: {results.get('use_automl', 'N/A')}")
        
        if results.get('data_summary'):
            summary = results['data_summary']
            print(f"\nData Summary:")
            print(f"   Rows: {summary.get('data_info', {}).get('rows', 'N/A')}")
            print(f"   Features: {summary.get('feature_info', {}).get('total_features', 'N/A')}")
            print(f"   Numeric: {summary.get('feature_info', {}).get('numeric_features', 'N/A')}")
            print(f"   Categorical: {summary.get('feature_info', {}).get('categorical_features', 'N/A')}")
        
        if results.get('selected_models'):
            print(f"\nSelected Models: {', '.join(results['selected_models'])}")
        
        if results.get('automl_config'):
            config = results['automl_config']
            print(f"\nAutoML Config:")
            print(f"   - Models: {config.get('models', [])}")
            print(f"   - Time Limit: {config.get('time_limit', 'N/A')}s")
            print(f"   - Preset: {config.get('preset', 'N/A')}")
        
        if results.get('model_metrics'):
            metrics = results['model_metrics']
            print(f"\nModel Performance:")
            print(f"   Best Model: {metrics.get('best_model', 'N/A')}")
            print(f"   Best Score: {metrics.get('best_score', 'N/A'):.4f}")
            print(f"   Training Method: {metrics.get('training_method', 'N/A')}")
            print(f"   Models Trained: {metrics.get('models_trained', 'N/A')}")
            if metrics.get('all_models'):
                print(f"   All Models: {', '.join(map(str, metrics.get('all_models', [])))}")
            if metrics.get('all_scores'):
                print(f"   All Scores: {[f'{s:.4f}' for s in metrics.get('all_scores', [])]}")
        
        if results.get('data_analysis_reasoning'):
            print(f"\nData Analysis Reasoning:")
            print(f"   {results['data_analysis_reasoning'][:500]}...")
        
        if results.get('model_selection_reasoning'):
            print(f"\nModel Selection Reasoning:")
            print(f"   {results['model_selection_reasoning'][:500]}...")
        
        print("="*80)
        
        test_report['datasets_tested'].append(dataset_result)
        test_report['summary']['successful'] += 1
        
    except Exception as e:
        error_msg = f"Error testing {dataset_name}: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        dataset_result['status'] = 'error'
        dataset_result['error'] = str(e)
        dataset_result['traceback'] = traceback.format_exc()
        test_report['datasets_tested'].append(dataset_result)
        test_report['summary']['failed'] += 1
    
    test_report['summary']['total_tests'] += 1
    return dataset_result

def generate_report():
    """Generate a comprehensive markdown report."""
    report_content = []
    
    # Header
    report_content.append("# AutoML Agent Test Report")
    report_content.append("")
    report_content.append(f"**Generated:** {test_report['start_time']}")
    report_content.append(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("")
    
    # Summary
    summary = test_report['summary']
    report_content.append("## Test Summary")
    report_content.append("")
    report_content.append(f"- **Total Tests:** {summary['total_tests']}")
    report_content.append(f"- **Successful:** {summary['successful']}")
    report_content.append(f"- **Failed:** {summary['failed']}")
    report_content.append(f"- **Success Rate:** {(summary['successful']/summary['total_tests']*100) if summary['total_tests'] > 0 else 0:.1f}%")
    report_content.append("")
    
    # Detailed Results
    report_content.append("## Detailed Test Results")
    report_content.append("")
    
    for i, dataset in enumerate(test_report['datasets_tested'], 1):
        report_content.append(f"### {i}. {dataset['name']}")
        report_content.append("")
        report_content.append(f"**Dataset Path:** `{dataset['path']}`")
        report_content.append(f"**Target Column:** `{dataset['target_column']}`")
        report_content.append(f"**Test Time:** {dataset['timestamp']}")
        report_content.append(f"**Status:** {'Success' if dataset['status'] == 'success' else 'Failed'}")
        report_content.append("")
        
        if dataset['status'] == 'error':
            report_content.append(f"**Error:** {dataset['error']}")
            if 'traceback' in dataset:
                report_content.append("")
                report_content.append("```")
                report_content.append(dataset['traceback'])
                report_content.append("```")
        else:
            results = dataset['results']
            
            # Basic Info
            report_content.append("#### Basic Information")
            report_content.append(f"- **Problem Type:** {results.get('problem_type', 'N/A')}")
            report_content.append(f"- **Target Column:** {results.get('target_column', 'N/A')}")
            report_content.append(f"- **Workflow Step:** {results.get('step', 'N/A')}")
            report_content.append("")
            
            # Data Summary
            if results.get('data_summary'):
                data_summary = results['data_summary']
                report_content.append("#### Data Summary")
                report_content.append(f"- **Rows:** {data_summary.get('data_info', {}).get('rows', 'N/A')}")
                report_content.append(f"- **Total Features:** {data_summary.get('feature_info', {}).get('total_features', 'N/A')}")
                report_content.append(f"- **Numeric Features:** {data_summary.get('feature_info', {}).get('numeric_features', 'N/A')}")
                report_content.append(f"- **Categorical Features:** {data_summary.get('feature_info', {}).get('categorical_features', 'N/A')}")
                report_content.append(f"- **Missing Values:** {'Yes' if data_summary.get('data_quality', {}).get('has_missing', False) else 'No'}")
                report_content.append("")
            
            # Training Strategy
            report_content.append("#### Training Strategy")
            use_automl = results.get('use_automl')
            if use_automl:
                report_content.append("- **Strategy:** AutoML (AutoGluon)")
                if results.get('automl_config'):
                    config = results['automl_config']
                    report_content.append(f"- **Models:** {', '.join(config.get('models', []))}")
                    report_content.append(f"- **Time Limit:** {config.get('time_limit', 'N/A')}s")
                    report_content.append(f"- **Preset:** {config.get('preset', 'N/A')}")
            else:
                report_content.append("- **Strategy:** Simple Training")
                if results.get('selected_models'):
                    report_content.append(f"- **Selected Models:** {', '.join(results['selected_models'])}")
            report_content.append("")
            
            # Model Performance
            if results.get('model_metrics'):
                metrics = results['model_metrics']
                report_content.append("#### Model Performance")
                report_content.append(f"- **Best Model:** {metrics.get('best_model', 'N/A')}")
                report_content.append(f"- **Best Score:** {metrics.get('best_score', 0):.4f}")
                report_content.append(f"- **Training Method:** {metrics.get('training_method', 'N/A')}")
                report_content.append(f"- **Models Trained:** {metrics.get('models_trained', 'N/A')}")
                if metrics.get('all_models'):
                    report_content.append(f"- **All Models:** {', '.join(map(str, metrics.get('all_models', [])))}")
                if metrics.get('all_scores'):
                    report_content.append(f"- **All Scores:** {[f'{s:.4f}' for s in metrics.get('all_scores', [])]}")
                report_content.append("")
            
            # Reasoning
            if results.get('data_analysis_reasoning'):
                report_content.append("#### Data Analysis Reasoning")
                report_content.append("")
                report_content.append("```")
                report_content.append(results['data_analysis_reasoning'])
                report_content.append("```")
                report_content.append("")
            
            if results.get('model_selection_reasoning'):
                report_content.append("#### Model Selection Reasoning")
                report_content.append("")
                report_content.append("```")
                report_content.append(results['model_selection_reasoning'])
                report_content.append("```")
                report_content.append("")
        
        report_content.append("---")
        report_content.append("")
    
    # Performance Summary Table
    report_content.append("## Performance Summary Table")
    report_content.append("")
    report_content.append("| Dataset | Type | Size | Strategy | Best Model | Score | Status |")
    report_content.append("|---------|------|------|----------|-------------|-------|--------|")
    
    for dataset in test_report['datasets_tested']:
        if dataset['status'] == 'success':
            results = dataset['results']
            data_summary = results.get('data_summary', {})
            rows = data_summary.get('data_info', {}).get('rows', 'N/A')
            problem_type = results.get('problem_type', 'N/A')
            strategy = 'AutoML' if results.get('use_automl') else 'Simple'
            metrics = results.get('model_metrics', {})
            best_model = metrics.get('best_model', 'N/A')
            best_score = metrics.get('best_score', 'N/A')
            if isinstance(best_score, (int, float)):
                best_score = f"{best_score:.4f}"
            status = "Success"
        else:
            rows = 'N/A'
            problem_type = 'N/A'
            strategy = 'N/A'
            best_model = 'N/A'
            best_score = 'N/A'
            status = "Failed"
        
        report_content.append(f"| {dataset['name']} | {problem_type} | {rows} rows | {strategy} | {best_model} | {best_score} | {status} |")
    
    report_content.append("")
    
    # Conclusion
    report_content.append("## Conclusion")
    report_content.append("")
    if summary['successful'] == summary['total_tests']:
        report_content.append("All tests completed successfully! The AutoML agent demonstrated:")
    else:
        report_content.append(f"Tests completed with {summary['failed']} failure(s). The AutoML agent demonstrated:")
    report_content.append("")
    report_content.append("- Dataset loading and analysis")
    report_content.append("- Automatic problem type identification")
    report_content.append("- Intelligent training strategy selection")
    report_content.append("- Model training and evaluation")
    report_content.append("- Detailed reasoning and explanations")
    report_content.append("")
    
    return "\n".join(report_content)

def save_report():
    """Save the report to a file."""
    report_dir = PROJECT_ROOT / "test_reports"
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f"test_report_{timestamp}.md"
    
    report_content = generate_report()
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📄 Report saved to: {report_file}")
    return report_file

def main():
    """Run tests on all downloaded scikit-learn datasets."""
    print("\n" + "="*80)
    print("AutoML Agent - Scikit-learn Dataset Testing")
    print("="*80)
    
    # Test datasets
    test_datasets = [
       
        {
            'path': 'assets/data/Datasets/Regression Datasets/Concrete Compressive Strength/car_prices.csv',
            'target': 'sellingprice',
            'name': 'carprices (Regression)'
        },

    ]
    
    # Run tests
    for dataset in test_datasets:
        test_dataset(dataset['path'], dataset['target'], dataset['name'])
    
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    
    # Generate and save report
    report_file = save_report()
    print(f"\nAll test results have been saved to the report.")
    print(f"Summary: {test_report['summary']['successful']}/{test_report['summary']['total_tests']} tests successful")
    print("")

if __name__ == "__main__":
    main()



