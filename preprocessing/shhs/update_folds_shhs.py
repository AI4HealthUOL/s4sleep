import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import random

def check_data(data):
    # Check if labels are within [0, 4] (Sleep stages W, N1, N2, N3, REM)
    return set([0, 1, 2, 3, 4]).issubset(set(data))

def update_shhs_folds(data_path):
    path = Path(data_path)
    df_file = path / "df_memmap.pkl"
    
    if not df_file.exists():
        print(f"Error: {df_file} not found.")
        return

    print(f"Loading {df_file}...")
    try:
        df_memmap = pd.read_pickle(df_file)
    except Exception as e:
        # Fallback for old pickles
        import pickle
        with open(df_file, "rb") as f:
            df_memmap = pickle.load(f)

    # Filter complete stages
    if 'comp_stages' not in df_memmap.columns:
        print("Calculating 'comp_stages' column...")
        # Assuming 'current_label' exists and contains numpy arrays of labels
        df_memmap['comp_stages'] = df_memmap['current_label'].apply(
            lambda x: check_data(np.array(x)) if isinstance(x, np.ndarray) else False
        )
    
    filtered_df = df_memmap[df_memmap['comp_stages']]
    num_comp = len(filtered_df)
    print(f"Total samples with complete stages: {num_comp}")
    
    if num_comp == 0:
        print("No samples found with complete stages (0-4 present).")
        return

    # Split sizes
    n_test = int(num_comp * 0.3)
    n_val = 100
    n_train = num_comp - n_test - n_val
    
    print(f"Split sizes: Train={n_train}, Val={n_val}, Test={n_test}")

    # Random split
    indices = list(filtered_df.index)
    random.seed(42) # Ensure reproducibility unlike original script
    test_indices = random.sample(indices, n_test)
    remaining_indices = list(set(indices) - set(test_indices))
    val_indices = random.sample(remaining_indices, n_val)
    train_indices = list(set(remaining_indices) - set(val_indices))
    
    # Update strat_fold
    # 0: Train, 1: Validation, 2: Test
    df_memmap['strat_fold'] = -1 

    df_memmap.loc[train_indices, 'strat_fold'] = 0
    df_memmap.loc[val_indices, 'strat_fold'] = 1
    df_memmap.loc[test_indices, 'strat_fold'] = 2
    
    print("Fold update complete.")
    print(df_memmap['strat_fold'].value_counts())
    
    # Save
    out_file = path / "df_memmap_updated.pkl"
    df_memmap.to_pickle(out_file)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update SHHS dataset folds")
    parser.add_argument("--data_path", required=True, help="Path to folder containing df_memmap.pkl")
    args = parser.parse_args()
    
    update_shhs_folds(args.data_path)
