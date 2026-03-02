import argparse
import sys
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne  # Using MNE as per original repo preferences for SEDF
import resampy
from scipy import signal

# Add project root to path to import clinical_ts
# Script is at s4sleep/preprocessing/sedf/preprocess_sedf.py
# Root is at ../../
sys.path.append(str(Path(__file__).parents[2]))

try:
    from clinical_ts.data.time_series_dataset_utils import reformat_as_memmap
except ImportError:
    print("Warning: clinical_ts module not found. Memmap generation might fail. Ensure you are running from the project root or have set PYTHONPATH.")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# --- Configuration & Constants ---
FS = 100  # Target sampling frequency
WIN_SIZE = 2
OVERLAP = 1
# NFFT calculated as 2^(ceil(log2(win_size*fs)))
NFFT = int(math.pow(2, math.ceil(math.log2(WIN_SIZE * FS))))

# Annotation Mapping (AASM Standard used in SEDF)
STAGES = {
    'Sleep stage W': 0, 
    'Sleep stage 1': 1, 
    'Sleep stage 2': 2, 
    'Sleep stage 3': 3, 
    'Sleep stage 4': 3, 
    'Sleep stage R': 4, 
    'Movement time': 5, 
    'Sleep stage ?': 5
}

# Channels for SEDF (Expanded set based on original script)
CHANNELS_REQUIRED = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']

def process_ts(raw_path, output_path, recreate=True, limit=None):
    """
    Process Time Series data from Sleep-EDF (expanded) dataset.
    Extracts signals, resamples to 100Hz, aligns with hypnograms, and saves as .npy.
    """
    output_path = Path(output_path)
    npy_path = output_path / "npy"
    if recreate:
        output_path.mkdir(parents=True, exist_ok=True)
        npy_path.mkdir(parents=True, exist_ok=True)
    
    raw_path = Path(raw_path)
    print(f"Scanning {raw_path}...")
    
    # SEDF file naming convention: SC4ssNEO-PSG.edf / SC4ssNEC-Hypnogram.edf
    psg_files = sorted(list(raw_path.rglob("*PSG.edf")))
    hyp_files = sorted(list(raw_path.rglob("*Hypnogram.edf")))
    
    if not psg_files:
        print(f"No *PSG.edf files found in {raw_path}")
        return

    # Match files by subject/recording ID (first 7 chars usually: SC4001E)
    pairs = []
    hyp_dict = {h.name[:7]: h for h in hyp_files}
    
    for psg in psg_files:
        stem = psg.name[:7]
        if stem in hyp_dict:
            pairs.append((psg, hyp_dict[stem]))
        else:
            # print(f"Warning: No hypnogram found for {psg.name}")
            pass

    print(f"Found {len(pairs)} matched pairs.")
    
    if limit:
        pairs = pairs[:limit]
        print(f"Limiting to {limit} pairs.")

    processed_records = []
    
    for psg_file, hyp_file in tqdm(pairs, desc="Processing TS"):
        stem = psg_file.stem 
        out_data_path = npy_path / (stem + ".npy")
        out_label_path = npy_path / (stem + "_label.npy")
        
        if not recreate and out_data_path.exists():
            # Skip existing
             processed_records.append({
                "filename": stem,
                "data": str(out_data_path.name),
                "label": str(out_label_path.name),
                "data_length": np.load(out_data_path, mmap_mode='r').shape[0],
                "data_original": str(psg_file),
                "patient_id": int(stem[3:5]) if stem[3:5].isdigit() else 0
            })
             continue

        try:
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            
            # Select channels
            available_chs = raw.ch_names
            
            picks = []
            for req_ch in CHANNELS_REQUIRED:
                # Fuzzy match?
                match = [ch for ch in available_chs if req_ch in ch]
                if match:
                    picks.append(match[0])
            
            if len(picks) != len(CHANNELS_REQUIRED):
                # print(f"Skipping {stem}: Missing channels. Required {CHANNELS_REQUIRED}, Found {available_chs}")
                continue
                
            raw.pick_channels(picks)
            
            # Resample if needed
            if int(raw.info['sfreq']) != FS:
                raw.resample(FS)
                
            # Get data [Channels, Time] -> Transpose to [Time, Channels]
            data = raw.get_data().T * 1e6
            data = data * 1e6
            
            # 2. Read Annotations
            annots = mne.read_annotations(hyp_file)
            
            labels = np.full(len(data), 5, dtype=np.int8) # Default to 5 (Movement/Unknown)
            
            for annot in annots:
                desc = annot['description']
                # Map description
                label_val = STAGES.get(desc, 5)
                
                onset = annot['onset']
                duration = annot['duration']
                
                start_idx = int(onset * FS)
                end_idx = int((onset + duration) * FS)
                
                # Clip to data bounds
                start_idx = max(0, start_idx)
                end_idx = min(len(labels), end_idx)
                
                if start_idx < end_idx:
                    labels[start_idx:end_idx] = label_val
            
            # Save
            np.save(out_data_path, data.astype(np.float32))
            np.save(out_label_path, labels)
            
            processed_records.append({
                "filename": stem,
                "data": str(out_data_path.name),
                "label": str(out_label_path.name),
                "data_length": len(data),
                "strat_fold": -1, # Placeholder
                "patient_id": int(stem[3:5]) if stem[3:5].isdigit() else 0,
                "data_original": str(psg_file)
            })
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue

    if processed_records:
        df = pd.DataFrame(processed_records)
        df_file = output_path / "df.pkl"
        
        pids = sorted(df['patient_id'].unique())
        # Random shuffle with seed
        rng = np.random.RandomState(42)
        rng.shuffle(pids)
        
        # 10 Folds
        fold_map = {pid: i % 10 for i, pid in enumerate(pids)}
        df['strat_fold'] = df['patient_id'].map(fold_map)
        
        # Save DF
        df.to_pickle(df_file)
        
        # Create memmap
        print("Creating Memmap for Time Series...")
        try:
             reformat_as_memmap(
                 df,
                 output_path / "memmap.npy",
                 fs=FS,
                 fs_annotation=FS,
                 data_folder=npy_path,
                 delete_npys=False, 
                 annotation=True,
                 df_as_parquet=False
             )
             # df.to_pickle(output_path / "df_memmap.pkl") # Created by reformat_as_memmap
        except Exception as e:
            print(f"Memmap generation failed: {e}")

def process_spec(ts_path, output_path, recreate=True, limit=None):
    """
    Generate Spectrograms from preprocessed Time Series (.npy files).
    """
    ts_path = Path(ts_path)
    output_path = Path(output_path)
    
    in_npy_path = ts_path / "npy"
    # Fallback if standard structure isn't followed
    if not in_npy_path.exists() and (ts_path / "df_memmap.pkl").exists():
         in_npy_path = ts_path / "npy" 
    
    if not in_npy_path.exists():
         print(f"Error: Input npy folder {in_npy_path} not found.")
         return

    npy_out_path = output_path / "npy"
    if recreate:
        output_path.mkdir(parents=True, exist_ok=True)
        npy_out_path.mkdir(parents=True, exist_ok=True)

    df_path_pkl = ts_path / "df_memmap.pkl"
    if df_path_pkl.exists():
        df = pd.read_pickle(df_path_pkl)

        if 'data_original' in df.columns:
             file_list = df['data_original'].tolist()
        else:
             file_list = df['data'].tolist()
             
        # Resolve paths
        files = []
        for f in file_list:
             f_path = Path(f)
             if f_path.is_absolute():
                  files.append(f_path)
             else:
                  files.append(in_npy_path / f_path)
    else:
        # Glob NPYs (excluding labels)
        files = list(in_npy_path.glob("*.npy"))
        files = [f for f in files if "_label" not in f.name]

    if limit:
        files = files[:limit]
        print(f"Limiting to {limit} files.")

    print(f"Processing {len(files)} files for Spectrograms...")
    
    spec_records = []
    
    for file_path in tqdm(files, desc="Processing Spec"):
        stem = file_path.stem
        out_file = npy_out_path / (stem + ".npy")
        
        if not recreate and out_file.exists():
             continue
             
        try:
            # Load TS [Time, Channels]
            data = np.load(file_path) 
            
            # Spectrogram Parameters matching SHHS/Original
            pad_width = ((int(0.5*FS), int(0.5*FS)), (0, 0)) 
            signals = np.pad(data, pad_width, mode='edge')
            
            f_spec, t_spec, Zxx = signal.spectrogram(
                signals.T, 
                fs=FS, 
                window=signal.windows.hamming(WIN_SIZE * FS), 
                noverlap=int(FS*OVERLAP), 
                nfft=NFFT
            )
            # Zxx shape: [Channels, Freqs, Time_bins]
            
            eps = np.finfo(float).eps
            Zxx_db = 20 * np.log10(np.abs(Zxx) + eps)
            
            # Save
            np.save(out_file, Zxx_db.astype(np.float32))
            
            spec_records.append({
                "filename": stem,
                "data": out_file.name,
                # We will copy/link labels later
                "label": stem + "_label.npy",
                "data_length": Zxx_db.shape[-1]
            })
            
        except Exception as e:
            print(f"Error {stem}: {e}")

    # Copy Labels & Create Memmap
    if spec_records:
         print("Linking label files...")
         for f in files:
             stem = f.stem
             src_lbl = f.parent / (stem + "_label.npy")
             dst_lbl = npy_out_path / (stem + "_label.npy")
             if src_lbl.exists() and not dst_lbl.exists():
                 try:
                     # Copy is safer across filesystems than symlink sometimes
                     import shutil
                     shutil.copy(src_lbl, dst_lbl)
                 except:
                     pass
                     
         df_spec = pd.DataFrame(spec_records)
         
         # Inherit folds if available
         if 'df' in locals():
              
              # Map stem to fold
              fold_map = pd.Series(df.strat_fold.values, index=df.filename).to_dict()
              pid_map = pd.Series(df.patient_id.values, index=df.filename).to_dict()
              
              df_spec['strat_fold'] = df_spec['filename'].map(fold_map)
              df_spec['patient_id'] = df_spec['filename'].map(pid_map)
         
         try:
            reformat_as_memmap(
                 df_spec,
                 output_path / "memmap.npy",
                 fs=1.0,
                 fs_annotation=1.0,
                 data_folder=npy_out_path,
                 delete_npys=False, 
                 annotation=True,
                 df_as_parquet=False
             )
            # df_spec.to_pickle(output_path / "df_memmap.pkl")
         except Exception as e:
             print(f"Spec memmap failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEDF Preprocessing (TS & Spec)")
    parser.add_argument("--mode", choices=["ts", "spec", "all"], default="all")
    parser.add_argument("--raw_path", help="Path to raw SEDF data (containing *PSG.edf)")
    parser.add_argument("--ts_path", help="Path to preprocessed Time Series (input for spec, output for ts)")
    parser.add_argument("--spec_path", help="Path to output Spectrograms")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N subjects")
    args = parser.parse_args()
    
    if args.mode in ["ts", "all"]:
        if args.raw_path and args.ts_path:
            process_ts(args.raw_path, args.ts_path, limit=args.limit)
        elif args.mode == "ts":
            print("Error: --raw_path and --ts_path required for TS mode")

    if args.mode in ["spec", "all"]:
        if args.ts_path and args.spec_path:
            process_spec(args.ts_path, args.spec_path, limit=args.limit)
        elif args.mode == "spec":
             print("Error: --ts_path and --spec_path required for Spec mode")
