"""
Dataset loader utility for downloading and accessing datasets from various sources.
Supports Hugging Face, scikit-learn, and direct downloads.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import requests
from typing import Optional, Dict, Any
from src.utils.logger import Logger

logger = Logger()


class DatasetLoader:
    """Load datasets from various sources for testing AutoML agent."""
    
    def __init__(self, download_dir: str = "assets/data/Datasets"):
        """
        Initialize dataset loader.
        
        Args:
            download_dir: Directory to save downloaded datasets
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.download_dir / "Classification Datasets").mkdir(exist_ok=True)
        (self.download_dir / "Regression Datasets").mkdir(exist_ok=True)
    
    def load_from_huggingface(self, dataset_name: str, config_name: Optional[str] = None, 
                               split: str = "train", save: bool = True) -> pd.DataFrame:
        """
        Load dataset from Hugging Face Datasets.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            config_name: Optional configuration name for the dataset
            split: Split to load ('train', 'test', 'all')
            save: Whether to save the dataset locally
        
        Returns:
            DataFrame with the dataset
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading dataset '{dataset_name}' from Hugging Face...")
            
            # Load dataset
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Convert to pandas DataFrame
            if isinstance(dataset, list):
                df = pd.DataFrame(dataset)
            else:
                df = pd.DataFrame(dataset)
            
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Save if requested
            if save:
                save_path = self.download_dir / f"{dataset_name.replace('/', '_')}.csv"
                df.to_csv(save_path, index=False)
                logger.info(f"Dataset saved to: {save_path}")
            
            return df
            
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset from Hugging Face: {str(e)}")
            raise
    
    def load_from_sklearn(self, dataset_name: str, save: bool = True) -> tuple[pd.DataFrame, str]:
        """
        Load built-in dataset from scikit-learn.
        
        Args:
            dataset_name: Name of sklearn dataset ('iris', 'wine', 'breast_cancer', 
                         'diabetes', 'california_housing')
            save: Whether to save the dataset locally
        
        Returns:
            Tuple of (DataFrame, target_column_name)
        """
        from sklearn import datasets
        
        dataset_map = {
            'iris': (datasets.load_iris, 'target'),
            'wine': (datasets.load_wine, 'target'),
            'breast_cancer': (datasets.load_breast_cancer, 'target'),
            'diabetes': (datasets.load_diabetes, 'target'),
            'california_housing': (datasets.fetch_california_housing, 'MedHouseVal'),
        }
        
        if dataset_name not in dataset_map:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
        
        logger.info(f"Loading sklearn dataset: {dataset_name}...")
        
        loader_func, target_col = dataset_map[dataset_name]
        
        # Handle california_housing which needs as_frame
        if dataset_name == 'california_housing':
            data = loader_func(as_frame=True, return_X_y=False)
            df = data.frame
        else:
            data = loader_func(as_frame=True, return_X_y=False)
            df = data.frame
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Save if requested
        if save:
            category = "Classification Datasets" if dataset_name in ['iris', 'wine', 'breast_cancer'] else "Regression Datasets"
            save_path = self.download_dir / category / f"{dataset_name}.csv"
            df.to_csv(save_path, index=False)
            logger.info(f"Dataset saved to: {save_path}")
        
        return df, target_col
    
    def download_from_url(self, url: str, filename: str, save_dir: Optional[str] = None) -> str:
        """
        Download dataset from URL.
        
        Args:
            url: URL to download from
            filename: Name to save the file as
            save_dir: Directory to save (defaults to download_dir)
        
        Returns:
            Path to downloaded file
        """
        if save_dir is None:
            save_dir = self.download_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / filename
        
        if file_path.exists():
            logger.info(f"File already exists: {file_path}")
            return str(file_path)
        
        logger.info(f"Downloading from {url}...")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end='')
            
            print()  # New line after progress
            logger.info(f"Downloaded to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading from URL: {str(e)}")
            raise
    
    def get_popular_datasets_info(self) -> Dict[str, Any]:
        """
        Get information about popular datasets for testing.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'huggingface': {
                'classification': [
                    {'name': 'scikit-learn/adult-census-income', 'desc': 'Adult Census Income (Classification)'},
                    {'name': 'jcmachado/titanic', 'desc': 'Titanic Dataset'},
                    {'name': 'imodels/credit-card', 'desc': 'Credit Card Default'},
                    {'name': 'dair-ai/emotion', 'desc': 'Emotion Classification'},
                    {'name': 'UCSD-AI4H/COVID-Dialogue', 'desc': 'COVID Dialogue Classification'},
                ],
                'regression': [
                    {'name': 'scikit-learn/california-housing', 'desc': 'California Housing Prices'},
                    {'name': 'scikit-learn/diabetes', 'desc': 'Diabetes Regression'},
                    {'name': 'mstz/wine', 'desc': 'Wine Quality'},
                    {'name': 'scikit-learn/boston-house-prices', 'desc': 'Boston House Prices'},
                ]
            },
            'sklearn': {
                'classification': ['iris', 'wine', 'breast_cancer'],
                'regression': ['diabetes', 'california_housing']
            },
            'direct_downloads': {
                'classification': [
                    {
                        'name': 'Adult Income',
                        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                        'desc': 'UCI Adult Income Dataset'
                    },
                    {
                        'name': 'Bank Marketing',
                        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip',
                        'desc': 'UCI Bank Marketing Dataset'
                    },
                ],
                'regression': [
                    {
                        'name': 'Wine Quality Red',
                        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
                        'desc': 'UCI Wine Quality Red Dataset'
                    },
                    {
                        'name': 'Wine Quality White',
                        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                        'desc': 'UCI Wine Quality White Dataset'
                    },
                ]
            }
        }
    
    def list_available_datasets(self):
        """Print available datasets for testing."""
        info = self.get_popular_datasets_info()
        
        print("\n" + "=" * 80)
        print("AVAILABLE DATASETS FOR TESTING")
        print("=" * 80)
        
        print("\n📊 Hugging Face Datasets:")
        print("\n  Classification:")
        for ds in info['huggingface']['classification']:
            print(f"    - {ds['name']}: {ds['desc']}")
        print("\n  Regression:")
        for ds in info['huggingface']['regression']:
            print(f"    - {ds['name']}: {ds['desc']}")
        
        print("\n🔬 Scikit-learn Built-in Datasets:")
        print("\n  Classification:")
        for ds in info['sklearn']['classification']:
            print(f"    - {ds}")
        print("\n  Regression:")
        for ds in info['sklearn']['regression']:
            print(f"    - {ds}")
        
        print("\n📥 Direct Download Datasets:")
        print("\n  Classification:")
        for ds in info['direct_downloads']['classification']:
            print(f"    - {ds['name']}: {ds['desc']}")
        print("\n  Regression:")
        for ds in info['direct_downloads']['regression']:
            print(f"    - {ds['name']}: {ds['desc']}")
        
        print("\n" + "=" * 80)


def download_test_datasets():
    """Download a selection of test datasets."""
    loader = DatasetLoader()
    
    logger.info("Downloading test datasets...")
    
    # Show available datasets
    loader.list_available_datasets()
    
    # Download some sklearn datasets (quick and reliable)
    print("\n📥 Downloading scikit-learn datasets...")
    try:
        df, target = loader.load_from_sklearn('iris', save=True)
        print(f"✅ Downloaded iris dataset: {df.shape}")
        
        df, target = loader.load_from_sklearn('wine', save=True)
        print(f"✅ Downloaded wine dataset: {df.shape}")
        
        df, target = loader.load_from_sklearn('breast_cancer', save=True)
        print(f"✅ Downloaded breast_cancer dataset: {df.shape}")
        
        df, target = loader.load_from_sklearn('diabetes', save=True)
        print(f"✅ Downloaded diabetes dataset: {df.shape}")
        
        df, target = loader.load_from_sklearn('california_housing', save=True)
        print(f"✅ Downloaded california_housing dataset: {df.shape}")
        
    except Exception as e:
        logger.error(f"Error downloading sklearn datasets: {str(e)}")
    
    # Download from Hugging Face if available
    print("\n📥 Attempting to download Hugging Face datasets...")
    hf_datasets = [
        ('scikit-learn/adult-census-income', 'classification', 'adult-census-income'),
    ]
    
    for dataset_name, category, save_name in hf_datasets:
        try:
            df = loader.load_from_huggingface(dataset_name, split='train', save=True)
            # Move to appropriate directory
            category_dir = loader.download_dir / f"{category.capitalize()} Datasets"
            old_path = loader.download_dir / f"{save_name}.csv"
            new_path = category_dir / f"{save_name}.csv"
            if old_path.exists():
                old_path.rename(new_path)
                print(f"✅ Downloaded {dataset_name}: {df.shape}")
        except Exception as e:
            logger.warn(f"Could not download {dataset_name}: {str(e)}")
            print(f"⚠️  Skipped {dataset_name} (may need datasets library)")
    
    print("\n✅ Dataset download complete!")
    print(f"Datasets saved to: {loader.download_dir}")


if __name__ == "__main__":
    download_test_datasets()

