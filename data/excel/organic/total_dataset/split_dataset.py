import argparse
import multiprocessing
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

error_with_protein: List[str] = [
    '1ffy_MRC_A_1993', '1qu2_MRC_A_1993', '1qu3_MRC_A_993', '4q5v_2ZE_A_1301',
    '6kda_SAH_A_901', '6kdb_SAH_A_901', '6zy5_EVP_D_101', '6zy6_EVP_D_101',
    '6zy7_EVP_D_101', '6zy8_EVP_D_101', '3c2l_F2A_A_338', '3c2m_F2A_A_338',
    '4f5p_F2A_A_404', '5zc9_RCG_B_600',
]


def process_dataset(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load metadata CSV and split into IC50 and Ki/Kd subsets.

    Parameters
    ----------
    file_path : str
        Path to the HiQBind metadata CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        IC50 DataFrame and Ki/Kd DataFrame, each deduplicated on (PDBID, Ligand Name).
    """
    df = pd.read_csv(file_path)
    unique_pdb_df = df.drop_duplicates(subset=['PDBID', 'Ligand Name'])
    ic50_df = unique_pdb_df[unique_pdb_df['Binding Affinity Measurement'] == 'ic50'].reset_index(drop=True)
    ki_kd_df = unique_pdb_df[unique_pdb_df['Binding Affinity Measurement'].isin(['ki', 'kd'])].reset_index(drop=True)
    return ic50_df, ki_kd_df


def verify_successful_optimization(row: Dict, path_to_data: str) -> bool:
    """
    Check whether the GFN2-xTB geometry optimisation for a given entry completed successfully.

    Parameters
    ----------
    row : Dict
        Row from the metadata DataFrame (with PDBID, Ligand Name, Ligand Chain, Ligand Residue Number).
    path_to_data : str
        Base directory containing the optimisation output files.

    Returns
    -------
    bool
        True if both ORCA and the geometry optimisation terminated normally.
    """
    pdbid = row['PDBID']
    ligand_name = row['Ligand Name']
    ligand_chain = row['Ligand Chain']
    ligand_residue_number = row['Ligand Residue Number']
    found_orca = False
    found_opt = False

    if f'{pdbid}_{ligand_name}_{ligand_chain}_{ligand_residue_number}' in error_with_protein:
        return False

    try:
        out_path = (
            f'{path_to_data}/{pdbid}'
            f'/{pdbid}_{ligand_name}_{ligand_chain}_{ligand_residue_number}'
            f'/GFN2-xTB/{pdbid}_{ligand_name}_{ligand_chain}_{ligand_residue_number}.out'
        )
        with open(out_path, 'r') as file:
            for line in file:
                if "ORCA TERMINATED NORMALLY" in line:
                    found_orca = True
                if "OPTIMIZATION RUN DONE" in line:
                    found_opt = True
    except FileNotFoundError:
        return False

    return found_orca and found_opt


def convert_to_log_value(row: pd.Series) -> Optional[float]:
    """
    Convert a binding affinity value to its negative log (pIC50/pKi scale).

    Parameters
    ----------
    row : pd.Series
        Row containing 'Binding Affinity Value' and 'Binding Affinity Unit'.

    Returns
    -------
    Optional[float]
        Negative log of the molar concentration, or None if the unit is unrecognised.
    """
    value = row['Binding Affinity Value']
    unit = row['Binding Affinity Unit']

    if unit == 'uM':
        value_in_nM = value * 1000
    elif unit == 'nM':
        value_in_nM = value
    else:
        return None

    molar_value = value_in_nM * 1e-9
    return -1 * np.log10(molar_value)


def filter_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate PDBID entries, keeping only successfully optimised structures.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a 'successful_optimization' boolean column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with at most one entry per PDBID.
    """
    filtered_df = pd.DataFrame(columns=df.columns)
    original_count = len(df)

    for pdbid in df['PDBID'].unique():
        pdbid_data = df[df['PDBID'] == pdbid].copy()

        if len(pdbid_data) == 1:
            if pdbid_data['successful_optimization'].iloc[0]:
                filtered_df = pd.concat([filtered_df, pdbid_data])
            continue

        successful_data = pdbid_data[pdbid_data['successful_optimization'] == True]

        if len(successful_data) == 0:
            print(f"No successful optimizations found for PDBID {pdbid}")
            continue

        if len(successful_data) == 1:
            filtered_df = pd.concat([filtered_df, successful_data])
            continue

        sorted_data = successful_data.sort_values('Ligand Chain')
        filtered_df = pd.concat([filtered_df, sorted_data.iloc[[0]]])

    filtered_df = filtered_df.reset_index(drop=True)
    deleted_count = original_count - len(filtered_df)
    print(f"Filtering removed {deleted_count} samples out of {original_count} (kept {len(filtered_df)})")
    return filtered_df


def apply_verification_parallel(
    df: pd.DataFrame, path_to_data: str, num_processes: Optional[int] = None
) -> List[bool]:
    """
    Apply verify_successful_optimization in parallel across DataFrame rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of compounds to verify.
    path_to_data : str
        Base directory containing optimisation output files.
    num_processes : Optional[int], default=None
        Number of worker processes (defaults to CPU count minus one).

    Returns
    -------
    List[bool]
        Per-row optimisation success flags.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count() - 1

    verify_func = partial(verify_successful_optimization, path_to_data=path_to_data)
    rows = df.to_dict('records')

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(verify_func, rows)

    return results


def main() -> None:
    """Main entry point: verify optimisations and split dataset into pIC50 and pK CSVs."""
    parser = argparse.ArgumentParser(description="Split HiQBind dataset into pIC50 and pK CSVs.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the directory containing raw HiQBind optimised compound files.",
    )
    parser.add_argument(
        "--metadata",
        default="hiqbind_sm_metadata.csv",
        help="Path to the HiQBind metadata CSV (default: hiqbind_sm_metadata.csv).",
    )
    args = parser.parse_args()

    ic50_data, ki_kd_data = process_dataset(args.metadata)
    path_to_optimized_data = args.data_dir

    print("Starting parallel verification for IC50 data...")
    ic50_data['successful_optimization'] = apply_verification_parallel(ic50_data, path_to_optimized_data)
    ic50_data['pIC50'] = ic50_data.apply(convert_to_log_value, axis=1)
    filtered_ic50_data = filter_duplicates(ic50_data)
    filtered_ic50_data.to_csv('../pic50/organic_pic50.csv', index=False)

    print("Starting parallel verification for Ki/Kd data...")
    ki_kd_data['successful_optimization'] = apply_verification_parallel(ki_kd_data, path_to_optimized_data)
    ki_kd_data['pIC50'] = ki_kd_data.apply(convert_to_log_value, axis=1)
    filtered_ki_kd_data = filter_duplicates(ki_kd_data)
    filtered_ki_kd_data.to_csv('../pk/organic_pk.csv', index=False)


if __name__ == "__main__":
    main()
