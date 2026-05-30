# data_loader.py
import os
import pickle
import pandas as pd
import itertools
from functools import reduce
from operator import mul
from typing import List, Dict, Any

from console import printer

class DataLoader:
    """
    Handle loading and preprocessing of supervised and transfer learning data.

    This class provides methods to load experimental results from pickle files
    and CSV files, handling both supervised learning and transfer learning data.
    """

    def __init__(self, supervised_base_dir: str = None, tl_csv_path: str = None) -> None:
        """
        Initialize the DataLoader with paths to supervised and transfer learning data.

        Parameters
        ----------
        supervised_base_dir : str, optional
            Base directory for supervised learning results. Default is './supervised_learning'.
        tl_csv_path : str, optional
            Full path to the transfer learning CSV file. Default is './transfer_learning_results.csv'.

        Returns
        -------
        None
        """
        self.supervised_base_dir = supervised_base_dir or "./supervised_learning"
        self.tl_csv_path = tl_csv_path or "./transfer_learning_results.csv"
        
        if not os.path.isdir(self.supervised_base_dir):
            printer.warning(f"Supervised data directory does not exist: {self.supervised_base_dir}")
        if not os.path.isfile(self.tl_csv_path):
            printer.warning(f"Transfer learning data file does not exist: {self.tl_csv_path}")

    def load_transfer_learning_results(self, csv_path: str, affinity_type: str) -> pd.DataFrame:
        """
        Load transfer learning results from a CSV file and preprocess them.

        This method loads transfer learning results, filters by affinity type,
        parses configuration columns, and performs data cleaning.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing transfer learning results.
        affinity_type : str
            The affinity type to filter for (e.g., 'pic50' or 'pk').

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with transfer learning results, or empty DataFrame if loading fails.
        """
        if not os.path.exists(csv_path):
            printer.error(f"Transfer learning data file not found at {csv_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_path)

            # The first column with the method is often unnamed in CSVs from pandas
            if 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'method'}, inplace=True)

            # Filter out excluded methods and non-fold runs
            if 'method' in df.columns:
                df = df[df['method'] != 'no_pretrain_ccsa_no_followup']
            if 'fold' in df.columns:
                df = df[df['fold'] != 'single_run']

            # Filter by affinity type early to reduce data size
            if 'affinity_type' in df.columns:
                df = df[df['affinity_type'] == affinity_type].copy()
                df.reset_index(drop=True, inplace=True)
            
            if df.empty:
                printer.warning(f"No data found for affinity_type='{affinity_type}' in {csv_path}")
                return pd.DataFrame()

            # --- PARSE CONFIGURATION COLUMNS ---
            # Correctly parse boolean flags from the 'scaling_config' string.
            # This will create or overwrite columns.
            if 'scaling_config' in df.columns:
                df['cross_domain_scaling'] = df['scaling_config'].apply(lambda x: 'cross_domain' in str(x))
                df['source_smote'] = df['scaling_config'].apply(lambda x: 'source_smote' in str(x) and 'no_source_smote' not in str(x))
                df['target_smote'] = df['scaling_config'].apply(lambda x: 'target_smote' in str(x) and 'no_target_smote' not in str(x))
                
            else:
                printer.warning("'scaling_config' column not found. Cannot derive smote/scaling flags.")

            # add pretrain status column
            df["pretrain_status"] = df["method"].map(
                lambda x: "True" if "pretrain" in x and "no_pretrain" not in x else "False"
            )
            
            # Add ccsa_status column
            df["ccsa_status"] = df["method"].map(
                lambda x: "True" if "ccsa" in x else "False"
            )

            # Correctly define the base method for proper pairing
            def get_base_method(method: str) -> str:
                pair_map = {
                    'pretrain_ccsa': 'pretrain',
                    'pretrain_ccsa_then_task': 'pretrain_task',
                    'pretrain_ccsa_then_encoder_task': 'pretrain_encoder_task',
                    'no_pretrain_ccsa': 'no_pretrain',
                    'no_pretrain_ccsa_then_task': 'no_pretrain_task',
                    'no_pretrain_ccsa_then_encoder_task': 'no_pretrain_encoder_task'
                }
                return pair_map.get(method, method)

            df["base_method"] = df["method"].apply(get_base_method)
            
            # --- FINAL DATA CLEANING ---
            if 'val/mcc' in df.columns:
                df['val/mcc'] = pd.to_numeric(df['val/mcc'], errors='coerce')
                df.dropna(subset=['val/mcc'], inplace=True)

            # change the anme of the bioloigdf
            df['modality'] = df['biological_target']
            df.drop(columns=['biological_target'], inplace=True)
            return df
            
        except Exception as e:
            printer.error(f"Failed to load or process transfer learning CSV at {csv_path}. Error: {e}")
            return pd.DataFrame()

    def load_supervised_learning_results(self, affinity_type: str, include_pca: bool = True) -> pd.DataFrame:
        """
        Load all supervised learning results for a specific affinity type.

        This method searches through the supervised learning directory structure
        and loads all pickle files containing classification results.

        Parameters
        ----------
        affinity_type : str
            The affinity type to load (e.g., 'pic50' or 'pk').
        include_pca : bool, optional
            Whether to include PCA results. Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all supervised learning results.
        """
        results_list = []
        base_path_template = os.path.join(
            self.supervised_base_dir,
            "{inhibitor}",
            "{modality}",
            "{scaler}",
            affinity_type
        )
        

        for inhibitor in ["ACSF", "MACE", "SOAP"]:
            for modality in ["inhibitor", "protein", "pocket", "inhibitor_protein_pocket", "inhibitor_protein", "inhibitor_pocket"]:
                for scaler in ["none_scaler", "standard_scaler", "minmax_scaler"]:
                    current_base_path = base_path_template.format(
                        inhibitor=inhibitor,
                        modality=modality,
                        scaler=scaler
                    )
                    
                    # Load data without PCA - both SMOTE and non-SMOTE variants
                    for smote_variant in ["", "_with_smote"]:
                        no_pca_filename = f"baseline_results_class_no_pca{smote_variant}.pkl"
                        no_pca_path = os.path.join(current_base_path, no_pca_filename)
                        
                        if os.path.exists(no_pca_path):
                            target_smote = True if smote_variant == "_with_smote" else False
                            results_list.extend(self._extract_results_from_pickle(
                                path=no_pca_path, inhibitor=inhibitor,
                                modality=modality, scaler=scaler, pca_status="no_pca",
                                target_smote=target_smote
                            ))
                    
                    # Load data with PCA if requested - both SMOTE and non-SMOTE variants
                    if include_pca:
                        for smote_variant in ["", "_with_smote"]:
                            pca_filename = f"baseline_results_class_with_pca_50c{smote_variant}.pkl"
                            pca_path = os.path.join(current_base_path, pca_filename)
                            if os.path.exists(pca_path):
                                target_smote = True if smote_variant == "_with_smote" else False
                                results_list.extend(self._extract_results_from_pickle(
                                    path=pca_path, inhibitor=inhibitor,
                                    modality=modality, scaler=scaler, pca_status="with_pca_50components",
                                    target_smote=target_smote
                                ))
        
        df_results = pd.DataFrame(results_list)
        return df_results

    def _extract_results_from_pickle(self, path: str, **kwargs) -> List[dict]:
        """
        Extract and format classification results from a pickle file.

        Parameters
        ----------
        path : str
            Path to the pickle file containing results.
        **kwargs
            Additional key-value pairs to include in each result record
            (e.g., inhibitor, modality, scaler, pca_status).

        Returns
        -------
        list[dict]
            List of dictionaries, each containing results for one fold/model combination.
        """
        extracted_data = []
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            if "classification" in data:
                for model_name, df_model in data["classification"].items():
                    if isinstance(df_model, pd.DataFrame):
                        for _, row in df_model.iterrows():
                            record = {
                                "method": model_name,
                                "fold": row.get("fold"),
                                "val/mcc": row.get("mcc"),
                                "val/loss": row.get("loss"), # Assuming 'loss' is the key
                                "val/accuracy": row.get("accuracy"), # Assuming 'accuracy' is the key
                                "val/f1": row.get("f1_score"), # Assuming 'f1_score' is the key
                                **kwargs
                            }
                            extracted_data.append(record)
        except (IOError, pickle.UnpicklingError) as e:
            printer.warning(f"Could not read or parse pickle file at {path}. Error: {e}")
        return extracted_data
    