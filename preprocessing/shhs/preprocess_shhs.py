import argparse
import sys
import os
import math
import warnings
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import pyedflib # Requires: pip install pyedflib
import resampy  # Requires: pip install resampy
from scipy import signal

# Add project root to path to import clinical_ts if needed
sys.path.append(str(Path(__file__).parents[2]))
try:
    from clinical_ts.data.time_series_dataset_utils import reformat_as_memmap
except ImportError:
    print("Warning: clinical_ts module not found. Memmap generation might fail.")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# --- Configuration & Constants ---
FS = 100  # Target sampling frequency
WIN_SIZE = 2
OVERLAP = 1
# NFFT calculated as 2^(ceil(log2(win_size*fs)))
NFFT = int(math.pow(2, math.ceil(math.log2(WIN_SIZE * FS))))

# SHHS Channel Configurations
CHANNELS_TO_USE = ['EEG', 'EEG(sec)', 'EOG(L)', 'EOG(R)', 'EMG']
TARGET_FS_LIST = [FS] * len(CHANNELS_TO_USE)

# Annotation Mapping
ANN_STOI = {
    'Wake': 0, 'Stage 1 sleep': 1, 'Stage 2 sleep': 2, 
    'Stage 3 sleep': 3, 'Stage 4 sleep': 3, 'REM sleep': 4, 
    'Movement': 5, "Unknown": 6
}

def parse_nsrr_xml(xml_path):
    """Parses SHHS *-nsrr.xml annotation files."""
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
            
        # Check start time
        patterns_start = re.findall(r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', content)
        if len(patterns_start) != 1:
            # Some files might not start at 0 or have different structure
            # print(f"Warning: {xml_path.name} distinctive start time pattern not found.")
            pass

        # Find stages
        patterns_stages = re.findall(
            r'<EventType>Stages.Stages</EventType>\n'
            r'<EventConcept>.+</EventConcept>\n'
            r'<Start>[0-9\.]+</Start>\n'
            r'<Duration>[0-9\.]+</Duration>',
            content
        )
        
        stages = []
        starts_30 = []
        
        for pattern in patterns_stages:
            lines = pattern.splitlines()

            concept_line = lines[1] 

            concept = re.search(r'<EventConcept>(.+)</EventConcept>', concept_line).group(1)
            
            stage_val = 6 # Unknown
            for key, val in ANN_STOI.items():
                if key in concept:
                    stage_val = val
                    break
            
            start_line = lines[2]
            start = float(re.search(r'<Start>([0-9\.]+)</Start>', start_line).group(1))
            
            duration_line = lines[3]
            duration = float(re.search(r'<Duration>([0-9\.]+)</Duration>', duration_line).group(1))
            
            # SHHS is usually 30s epochs
            epoch_duration = int(duration) // 30
            if epoch_duration > 0:
                stages.extend([stage_val] * epoch_duration)
                starts_30.extend(list(range(int(start), int(start+duration), 30)))
                
        return stages, starts_30
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return [], []

def process_ts(raw_path, output_path, recreate=True, limit=None):
    output_path = Path(output_path)
    npy_path = output_path / "npy"
    if recreate:
        output_path.mkdir(parents=True, exist_ok=True)
        npy_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Scan Dataset
    records = []
    raw_path = Path(raw_path)
    print(f"Scanning {raw_path}...")
    

    xml_files = sorted(list(raw_path.rglob("*-nsrr.xml")))
    
    for xml_file in xml_files:
        stem = xml_file.name.replace("-nsrr.xml", "")

        edf_file = xml_file.parent / (stem + ".edf")
        if not edf_file.exists():
            # Try global search (slower)
             candidates = list(raw_path.rglob(stem + ".edf"))
             if candidates:
                 edf_file = candidates[0]
             else:
                 continue
        
        records.append({
            "stem": stem,
            "xml_file": xml_file,
            "edf_file": edf_file
        })
    
    df_meta = pd.DataFrame(records)
    
    if df_meta.empty:
        print("No paired EDF/XML files found.")
        return

    if limit:
        df_meta = df_meta.head(limit)
        print(f"Limiting to {limit} records.")

    processed_records = []
    
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Processing TS"):
        edf_path = row['edf_file']
        xml_path = row['xml_file']
        stem = row['stem']
        
        out_data_path = npy_path / (stem + ".npy")
        out_label_path = npy_path / (stem + "_label.npy")
        
        if not recreate and out_data_path.exists() and out_label_path.exists():
             processed_records.append({
                "filename": stem,
                "data": str(out_data_path.name),
                "label": str(out_label_path.name),
                "data_length": np.load(out_data_path, mmap_mode='r').shape[0], # Lazy load
                "data_original": str(edf_path),
                # Extract patient ID if needed for folds
                "patient_id": int(stem.split('-')[1]) if '-' in stem else 0
            })
             continue

        # 1. Read Annotations
        stages, _ = parse_nsrr_xml(xml_path)
        if not stages:
            continue
            
        # 2. Read Signals
        try:
            f = pyedflib.EdfReader(str(edf_path))
            signal_labels = f.getSignalLabels()
            
            sigbufs = []
            found_channels = []
            
            for ch_name in CHANNELS_TO_USE:
                idx = -1
                # Exact match first
                if ch_name in signal_labels:
                    idx = signal_labels.index(ch_name)
                    found_channels.append(ch_name)
                # Fallback logic from original code
                elif ch_name == "EEG(sec)" and "EEG sec" in signal_labels:
                     idx = signal_labels.index("EEG sec")
                     found_channels.append(ch_name)
                # If still not found, try case insensitive or look for alternatives
                if idx != -1:
                    sig = f.readSignal(idx)
                    orig_fs = f.getSampleFrequency(idx)
                    
                    # Resample if needed
                    if orig_fs != FS:
                        sig = resampy.resample(sig, orig_fs, FS).astype(np.float32)
                    else:
                        sig = sig.astype(np.float32)
                    sigbufs.append(sig)
                else:
                    # Missing channel
                    pass 
            
            f.close()
            
            if len(sigbufs) != len(CHANNELS_TO_USE):
                # print(f"Skipping {stem}: Missing channels. Found {found_channels}, required {CHANNELS_TO_USE}")
                continue
                
            # Stack: (Time, Channels)
            # Find min length
            min_len = min([len(s) for s in sigbufs])
            data = np.stack([s[:min_len] for s in sigbufs], axis=1) # Shape: [T, C]
            
            # Align labels
            # stages is list of 30s epochs labels
            # data is 100Hz signal
            # Create sample-wise labels
            
            n_epochs = len(stages)
            data_len_expected = n_epochs * 30 * FS
            
            # Truncate data to match annotations or vice versa
            limit = min(data.shape[0], data_len_expected)
            data = data[:limit]
            
            # Create label array
            labels = np.zeros(limit, dtype=np.int8)
            for i, stage in enumerate(stages):
                start = i * 30 * FS
                end = (i + 1) * 30 * FS
                if end > limit: break
                labels[start:end] = stage
                
            # Save
            np.save(out_data_path, data)
            np.save(out_label_path, labels)
            
            processed_records.append({
                "filename": stem,
                "data": str(out_data_path.name),
                "label": str(out_label_path.name),
                "data_length": len(data),
                "data_original": str(edf_path),
                "patient_id": int(stem.split('-')[1]) if '-' in stem else 0
            })
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue

    if processed_records:
        df = pd.DataFrame(processed_records)
        df_file = output_path / "df.pkl"
        df.to_pickle(df_file)
        
        # Create memmap structure
        print("Creating Memmap for Time Series...")
        try:
             # Basic fold assignment placeholder
             pids = sorted(df['patient_id'].unique())
             rng = np.random.RandomState(42)
             rng.shuffle(pids)
             fold_map = {pid: i % 10 for i, pid in enumerate(pids)}
             df['strat_fold'] = df['patient_id'].map(fold_map)

             df = reformat_as_memmap(
                 df,
                 output_path / "memmap.npy",
                 fs=FS,
                 fs_annotation=FS,
                 data_folder=npy_path,
                 delete_npys=False, 
                 annotation=True,
                 df_as_parquet=False
             )
             # df.to_pickle(output_path / "df_memmap.pkl")
        except Exception as e:
            print(f"Memmap generation failed: {e}")

def process_spec(ts_path, output_path, limit=None, recreate=True):
    ts_path = Path(ts_path)
    output_path = Path(output_path)
    
    # Input is the 'npy' folder inside ts_path usually, or the root if directly provided?
    # Based on previous structure: ts_path/npy/*.npy
    
    in_npy_path = ts_path / "npy"
    if not in_npy_path.exists():
         # Maybe the user pointed directly to npy folder?
         if (ts_path / "df_memmap.pkl").exists():
              in_npy_path = ts_path / "npy"
         else:
             print(f"Check path {ts_path}. Expected to find 'npy' folder or df_memmap.pkl")
             
    if not in_npy_path.exists():
         print(f"Input npy folder {in_npy_path} not found.")
         return

    npy_out_path = output_path / "npy"
    if recreate:
        output_path.mkdir(parents=True, exist_ok=True)
        npy_out_path.mkdir(parents=True, exist_ok=True)

    # Use the DF to iterate if available
    df_path = ts_path / "df_memmap.pkl"
    if not df_path.exists():
        files = list(in_npy_path.glob("*.npy"))
        # Exclude labels
        files = [f for f in files if "_label" not in f.name]
    else:
        df = pd.read_pickle(df_path)
        # Check if 'data' is int (memmapped)
        if pd.api.types.is_integer_dtype(df['data']):
             files = [in_npy_path / f for f in df['data_original'].tolist()]
        else:
             files = [in_npy_path / f for f in df['data'].tolist()]

    print(f"Processing {len(files)} files for Spectrograms...")
    
    if limit:
        files = files[:limit]
        print(f"Limiting to {limit} files.")

    spec_records = []
    
    for file_path in tqdm(files, desc="Spec Gen"):
        file_path = Path(file_path)
        stem = file_path.stem
        out_file = npy_out_path / (stem + ".npy")
        
        # Check if label exists, if not skip (shouldn't happen if sourced from valid TS)
        src_lbl = in_npy_path / (stem + "_label.npy")
        if not src_lbl.exists():
            print(f"Missing label for {stem}, skipping.")
            continue

        if not recreate and out_file.exists():
             spec_records.append({
                "filename": stem,
                "data": out_file.name,
                "label": stem + "_label.npy",
                "data_length": np.load(out_file, mmap_mode='r').shape[-1],
                "data_original": str(file_path),
                "patient_id": int(stem.split('-')[1]) if '-' in stem else 0
             })
             continue
             
        try:
            data = np.load(file_path) # [T, C]
            # Spectrogram Logic from uploaded file
            # Padding
            pad_width = ((int(0.5*FS), int(0.5*FS)), (0, 0)) # Pad time dimension
            signals = np.pad(data, pad_width, mode='edge') 
            f_spec, t_spec, Zxx = signal.spectrogram(
                signals.T, 
                fs=FS, 
                window=signal.windows.hamming(WIN_SIZE * FS), 
                noverlap=int(FS*OVERLAP), 
                nfft=NFFT
            )
            # Zxx shape: [C, Freq, Time]
            
            eps = np.finfo(float).eps
            Zxx_db = 20 * np.log10(np.abs(Zxx) + eps)
            Zxx_db = np.moveaxis(Zxx_db, -1, 0)  # (C, F, T) -> (T, C, F)
            
            np.save(out_file, Zxx_db.astype(np.float32))
        
            
            spec_records.append({
                "filename": stem,
                "data": out_file.name,
                "label": stem + "_label.npy", 
                "data_length": Zxx_db.shape[0], # Time dimension is now 0 after transpose
                "data_original": str(file_path),
                "patient_id": int(stem.split('-')[1]) if '-' in stem else 0
            })
            
            # Symlink label
            dst_lbl = npy_out_path / (stem + "_label.npy")
            if not dst_lbl.exists():
                 try:
                     os.symlink(src_lbl, dst_lbl)
                 except OSError:
                     import shutil
                     shutil.copy(src_lbl, dst_lbl)
            
        except Exception as e:
            print(f"Error {stem}: {e}")

    # Finalize Spec DF
    if spec_records:
         
         print("Copying/Linking label files...")
         # Already done in loop but ensuring consistency
         pass

         df_spec = pd.DataFrame(spec_records)
         # Fold logic
         pids = sorted(df_spec['patient_id'].unique())
         rng = np.random.RandomState(42)
         rng.shuffle(pids)
         fold_map = {pid: i % 10 for i, pid in enumerate(pids)}
         df_spec['strat_fold'] = df_spec['patient_id'].map(fold_map)

         print("Creating Memmap for Spectrograms...")
         try:
             reformat_as_memmap(
                 df_spec,
                 output_path / "memmap.npy",
                 fs=FS,
                 fs_annotation=FS,
                 data_folder=npy_out_path,
                 delete_npys=False, 
                 annotation=True
             )
             df_spec.to_pickle(output_path / "df_memmap.pkl")
         except Exception as e:
             print(f"Spec Memmap generation failed: {e}")
             src_lbl = f.parent / (stem + "_label.npy")
             dst_lbl = npy_out_path / (stem + "_label.npy")
             if src_lbl.exists() and not dst_lbl.exists():
                 # Symlink relative
                 try:
                     os.symlink(src_lbl, dst_lbl)
                 except:
                     import shutil
                     shutil.copy(src_lbl, dst_lbl)
                     
         df_spec = pd.DataFrame(spec_records)
         # Merge with original meta if possible to keep folds
         if 'df' in locals(): # df from TS load
             common = df[['filename', 'strat_fold', 'patient_id']].copy() if 'patient_id' in df.columns else df[['filename']].copy()
             # Adjust merge if filename in df has .npy extension vs stem
             # ...
             pass
         
         try:
            # Simple fallback for memmap
            if 'strat_fold' not in df_spec.columns:
                 df_spec['strat_fold'] = np.random.randint(0, 8, size=len(df_spec))

            df_spec = reformat_as_memmap(
                 df_spec,
                 output_path / "memmap.npy",
                 fs=FS,
                 fs_annotation=FS,
                 data_folder=npy_out_path,
                 delete_npys=False, 
                 annotation=True,
                 df_as_parquet=False
             )
            # df_spec.to_pickle(output_path / "df_memmap.pkl") # reformat_as_memmap already saves it
         except Exception as e:
             print(f"Spec memmap failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHHS Preprocessing (TS & Spec)")
    parser.add_argument("--mode", choices=["ts", "spec", "all"], default="all")
    parser.add_argument("--raw_path", help="Path to raw SHHS EDFs")
    parser.add_argument("--ts_path", help="Path to preprocessed Time Series (input for spec, output for ts)")
    parser.add_argument("--spec_path", help="Path to output Spectrograms")
    parser.add_argument("--limit", type=int, help="Limit number of files for testing", default=None)
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
