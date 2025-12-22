from .dataset_summary import inspect_dataset

def run_eda(df, target_col=None):
    """
    Public entry point for EDA.
    """
    return {
        "dataset_summary": inspect_dataset(df, target_col)
    }
