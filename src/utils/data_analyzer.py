"""
Data analysis utilities for tabular data.
Handles data loading, analysis, and target column identification.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path


class DataAnalyzer:
    """Analyzes tabular data to extract features and identify target columns."""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.problem_type: Optional[str] = None  # 'classification' or 'regression'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        Supports: CSV, Excel, JSON
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            self.data = pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            self.data = pd.read_excel(file_path)
        elif extension == '.json':
            self.data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return self.data
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Analyze the loaded data and return statistics.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        analysis = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.astype(str).to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Add statistics for numeric columns
        if analysis['numeric_columns']:
            analysis['numeric_stats'] = self.data[analysis['numeric_columns']].describe().to_dict()
        
        # Add value counts for categorical columns (sample)
        if analysis['categorical_columns']:
            analysis['categorical_info'] = {}
            for col in analysis['categorical_columns'][:10]:  # Limit to first 10
                unique_count = self.data[col].nunique()
                analysis['categorical_info'][col] = {
                    'unique_values': int(unique_count),
                    'top_values': self.data[col].value_counts().head(5).to_dict() if unique_count <= 20 else None
                }
        
        return analysis
    
    def identify_target_column(self, target_column: Optional[str] = None) -> str:
        """
        Identify or validate the target column.
        If not provided, attempts to identify it based on common patterns.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if target_column:
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data.")
            self.target_column = target_column
        else:
            # Try to identify target column based on common names
            common_target_names = [
                'target', 'label', 'y', 'class', 'outcome', 'result',
                'price', 'cost', 'value', 'score', 'rating'
            ]
            
            for name in common_target_names:
                if name.lower() in [col.lower() for col in self.data.columns]:
                    matching_cols = [col for col in self.data.columns if name.lower() in col.lower()]
                    self.target_column = matching_cols[0]
                    break
            
            # If still not found, use the last column
            if self.target_column is None:
                self.target_column = self.data.columns[-1]
        
        return self.target_column
    
    def determine_problem_type(self) -> str:
        """
        Determine if the problem is classification or regression.
        """
        if self.target_column is None:
            raise ValueError("Target column not identified. Call identify_target_column() first.")
        
        target_data = self.data[self.target_column]
        
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(target_data):
            # If numeric with few unique values relative to dataset size, might be classification
            unique_ratio = target_data.nunique() / len(target_data)
            if unique_ratio < 0.05 and target_data.nunique() <= 20:
                self.problem_type = 'classification'
            else:
                self.problem_type = 'regression'
        else:
            self.problem_type = 'classification'
        
        return self.problem_type
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the data for model selection.
        """
        if self.target_column is None:
            raise ValueError("Target column not identified.")
        
        analysis = self.analyze_data()
        problem_type = self.determine_problem_type()
        
        summary = {
            'data_info': {
                'rows': int(analysis['shape'][0]),
                'columns': int(analysis['shape'][1]),
                'memory_mb': round(analysis['memory_usage_mb'], 2),
            },
            'target_info': {
                'column': self.target_column,
                'problem_type': problem_type,
                'unique_values': int(self.data[self.target_column].nunique()),
            },
            'feature_info': {
                'numeric_features': len(analysis['numeric_columns']) - (1 if self.target_column in analysis['numeric_columns'] else 0),
                'categorical_features': len(analysis['categorical_columns']) - (1 if self.target_column in analysis['categorical_columns'] else 0),
                'total_features': analysis['shape'][1] - 1,
            },
            'data_quality': {
                'missing_values_pct': round(sum(analysis['missing_percentage'].values()) / len(analysis['columns']), 2),
                'has_missing': any(v > 0 for v in analysis['missing_values'].values()),
            }
        }
        
        if problem_type == 'classification':
            summary['target_info']['class_distribution'] = self.data[self.target_column].value_counts().to_dict()
        
        return summary


