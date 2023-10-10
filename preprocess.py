__all__ = ['get_data_annot_stats', 'resample_data', 'prepare_dataset_with_annotations', 'prepare_SEDF']

# for preprocessing Sleep-edf recordings.
# generate npy and memmap for both time-series and spectrogram

import pyedflib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import resampy
import warnings
from clinical_ts.stratify import stratify, stratify_batched
from clinical_ts.timeseries_utils import *
from scipy import signal
import math
import mne
warnings.simplefilter(action='ignore', category=FutureWarning)

data_path = '../../datasets/sedf' # path of raw dataset


target_folder = "../../dataprepro/sedf/ts" # path of preprocessed data
# target_folder_spectra = "../../dataprepro/sedf/spec"




ann_stoi_SEDF = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4, 'Movement time': 5, 'Sleep stage ?': 6}
channel_to_use_SEDF = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
channel_to_resample_SEDF = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
target_fs = [100, 100, 100]

target_root = Path(target_folder)
#target_root_spectra = Path(target_folder_spectra)



def get_data_annot_stats(data_path, annotation=True, rhythm=True):

    annFile_list = list(Path(data_path).glob('**/*-Hypnogram.edf'))
    result = []

    for filename in tqdm(list(Path(data_path).glob('**/*-PSG.edf'))):

        f = pyedflib.EdfReader(str(filename))
        try:
            header = f.getSignalHeaders()
        except:
            print("Invalid file:", filename)
            continue
        f.close()

        meta = {"filename": filename}

        for key in header[0]:
            item = [header[i][key] for i in range(len(header))]
            meta.update({key: item})

        meta.update({"sample_frequency": [i/30 for i in meta["sample_frequency"]]})
        meta.update({"sample_rate": [i/30 for i in meta["sample_rate"]]})
        if annotation:
            if not Path(str(filename)).exists():
                meta["symbol"] = []
                if(rhythm):
                    meta["rhythm"] = []
                else:
                    meta["aux_note"] = []
            else:

                for i, pt in enumerate(annFile_list):
                    if (str(filename)[:-9]) in str(pt):
                        f = mne.read_annotations(pt)
                        meta['SlstageID'], meta['Slstage'] = f.onset, f.description
                        meta["symbol"] = np.unique(meta['Slstage'])

                        continue

            meta['channel'] = meta.pop('label')
        result.append(meta)

    df_stats = pd.DataFrame(result)
    df_stats["eeg_id"] = df_stats.filename.apply(lambda x: x.stem[:-4])

    if(annotation):
        unique_symbols, unique_symbols_counts = np.unique([item for sublist in list(df_stats.symbol) for item in sublist], return_counts=True)
        print("Sleep stage annotations:")
        for us, usc in zip(unique_symbols, unique_symbols_counts):
            print(us, usc)

    return df_stats


# resampling selected channels,
def resample_data(sigbufs, channel_list, channel_to_resample, fs, target_fs):

    for i, cl in enumerate(channel_to_resample):
        if cl not in channel_list:  # if there is a typo for inputting resampling channel names
            print(f"No channel is with the name of '{cl}'")
            quit()
        else:

            sigbufs[cl] = resampy.resample(sigbufs[cl], fs[i], target_fs[i]).astype(np.float32)

    return sigbufs



def prepare_dataset_with_annotations(df_stats, ann_stoi, dataset_name="SEFD", discard_labels=[""], strat_folds=10, rhythm=True, create_segments=True, min_len_segments=100, drop_unk=False, target_fs=target_fs, channels=12, channel_to_resample = channel_to_resample_SEDF, target_folder=target_folder, recreate_data=True):
    result = []
    target_root = Path(target_folder) if target_folder is None else Path(target_folder)
    target_root.mkdir(parents=True, exist_ok=True)
    if(recreate_data is True):

        metadata = []
        metadata_single = []
        for sample_id, row in tqdm(df_stats.iterrows(), total=len(df_stats)):
            filename = row["filename"]
            try:
                f = pyedflib.EdfReader(str(filename))
                channel_list = df_stats.loc[sample_id].loc['channel']
                fs_all = df_stats.loc[sample_id].loc['sample_frequency']
                sigbufs = {}

                for item in channel_to_use_SEDF:
                    sigbufs[item] = f.readSignal(channel_list.index(item))


                f.close()
            except:
                print("Invalid file:", filename)
                continue
            

            fs = [100, 100, 100]

            data_dict = resample_data(sigbufs=sigbufs, channel_list=channel_list, channel_to_resample=channel_to_resample_SEDF, fs=fs, target_fs=target_fs)
            data = np.zeros((len(data_dict[channel_to_resample[0]]), len(data_dict)), dtype=np.float32)
            for i in range(len(data_dict)):
                data[:, i] = data_dict[channel_to_resample[i]]
            for e, item in enumerate(channel_list):
                if item in channel_to_resample:
                    fs_all[e] = target_fs[channel_to_resample.index(item)]
            df_stats['sample_frequency'].replace(df_stats['sample_frequency'][sample_id], fs_all)
            df_stats['sample_rate'].replace(df_stats['sample_rate'][sample_id], fs_all)

            meta = df_stats.iloc[sample_id]

            ann_sample = np.array(df_stats.iloc[sample_id]['SlstageID'])  # count from the second label/first label
            ann_annotation = np.array(df_stats.iloc[sample_id]['Slstage'])
            segments = []
            segments_label = []
            ID_move = []
            count_move = 0
            for i, (sym, sta) in enumerate(zip(ann_annotation, ann_sample)):

                if i == 0 and ann_sample[1]-ann_sample[0] > 1800:
                    sta_temp = ann_sample[1]-1800  # time (second)
                    i_count = 0
                    while sta_temp + 30*i_count < ann_sample[i+1]:
                        staID = sta_temp*fs[0] + 30*fs[0]*i_count

                        segments.append(staID)
                        segments_label.append(ann_stoi[sym])
                        i_count += 1
                if i == 0 and ann_sample[1]-ann_sample[0] <= 1800:
                    sta_temp = sta  # time (second)
                    i_count = 0
                    while sta_temp + 30*i_count < ann_sample[i+1]:
                        staID = sta_temp*fs[0] + 30*fs[0]*i_count

                        segments.append(staID)
                        segments_label.append(ann_stoi[sym])
                        i_count += 1

                if i>=1 and i < len(ann_sample)-2:  # until the second last
                    sta_temp = ann_sample[i]
                    i_count = 0
                    while sta_temp + 30*i_count < ann_sample[i+1]:
                        staID = sta_temp*fs[0] + 30*fs[0]*i_count
                        segments.append(staID)
                        segments_label.append(ann_stoi[sym])
                        i_count += 1

                if i == len(ann_sample)-2 :  # the second last
                    if ann_annotation[i+1] == 'Sleep stage ?':
                        ann_sample_tempEnd = min(ann_sample[i+1], ann_sample[i]+1800)
                        sta_temp = ann_sample[i]
                        i_count = 0
                        while sta_temp + 30*i_count < ann_sample_tempEnd:

                            staID = sta_temp*fs[0] + 30*fs[0]*i_count

                            segments.append(staID)
                            segments_label.append(ann_stoi[sym])
                            i_count += 1
                        break

                    if ann_annotation[i+1] != 'Sleep stage ?':
                        sta_temp = ann_sample[i]
                        i_count = 0
                        while sta_temp + 30*i_count < ann_sample[i+1]:
                            staID = sta_temp*fs[0] + 30*fs[0]*i_count
                            segments.append(staID)
                            segments_label.append(ann_stoi[sym])
                            i_count += 1


                if i == len(ann_sample)-1:  # the last
                    sta_temp = ann_sample[i]
                    staID = sta_temp*fs[0]
                    segments.append(staID)
                    segments_label.append(ann_stoi[sym])




            meta_temp = {"data": Path(filename.stem[:-4]+".npy"),  "label": Path(filename.stem[:-4]+"_ann.npy"), "current_label": segments_label, "move_count":count_move, "ann_stoi": ann_stoi_SEDF, "ori_start_index": segments[0], "ori_end_index": segments[-1]+3000, "patient_id": row["patient_id"] if "patient_id" in df_stats.columns else sample_id}
            meta_1 = meta_temp | dict(meta)
            ID_whole = list(range(int(segments[0]), int((segments[-1]+3000))))
            ID_chose = [ID for ID in ID_whole if ID not in ID_move]

            np.save(target_root/(filename.stem[:-4]+".npy"), data[ID_chose])
            np.save(target_root/(filename.stem[:-4]+"_ann.npy"), segments_label)
            metadata.append(meta_1)

            win_size = 2
            overlap = 1
            nfft = int(math.pow(2, math.ceil(math.log2(win_size * target_fs[0]))))

            signals = np.pad(data[ID_chose], ((int(0.5*fs[0]), int(0.5*fs[0])), (0, 0)))
            _, _, Zxx = signal.spectrogram(signals.T, fs=fs[0], window=signal.windows.hamming(win_size * fs[0]), noverlap=int(fs[0]*overlap), nfft=nfft)

            eps = np.finfo(float).eps  # get the smallest representable float value
            Zxx_db = 20 * np.log10(np.abs(Zxx) + eps)
            Zxx_db=np.moveaxis(Zxx_db,-1,0)


            #np.save(target_root_spectra/(filename.stem[:-4]+".npy"), Zxx_db)
            #np.save(target_root_spectra/(filename.stem[:-4]+"_ann.npy"), segments_label)

            meta_whole = {"data": Path(filename.stem[:-4]+".npy"), "label": Path(filename.stem[:-4]+"_ann.npy"), "label_unique": np.unique(segments_label), "patient_id": row["patient_id"] if "patient_id" in df_stats.columns else sample_id}
            metadata_single.append(meta_whole)


        df = pd.DataFrame(metadata)
        df_single = pd.DataFrame(metadata_single)


        lbl_unique_single = np.unique([item for sublist in list(df_single["label_unique"]) for item in sublist])

        df["dataset"] = dataset_name
        df_patients = (df_single).groupby("patient_id")["label_unique"].apply(lambda x: list(x))
        patients_ids = list(df_patients.index)
        patients_labels = list(df_patients.apply(lambda x: [item for sublist in x for item in sublist]))
        patients_num_ecgs = list(df_patients.apply(len))

        stratified_ids = stratify(patients_labels, lbl_unique_single, [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs)
        stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

        df["strat_fold"]=-1
        for i, split in enumerate(stratified_patient_ids):
            df.loc[df.patient_id.isin(split), "strat_fold"] = i

        lbl_itos = [""]*int(1+max(np.unique(list(ann_stoi.values()))))
        for k in ann_stoi.keys():
            lbl_itos[ann_stoi[k]]= k if lbl_itos[ann_stoi[k]]=="" else lbl_itos[ann_stoi[k]]+'|'+k

        dataset_add_mean_col(df, data_folder=target_root)
        dataset_add_std_col(df, data_folder=target_root)
        dataset_add_length_col(df, data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        save_dataset(df, lbl_itos, mean, std, target_root)
        #save_dataset(df, lbl_itos, mean, std, target_root_spectra)

        df.to_pickle(target_root/("df"+".pkl"), protocol=4)
        #df.to_pickle(target_root_spectra/("df"+".pkl"), protocol=4)
    return df, lbl_itos, mean, std


def prepare_SEDF(data_path, ann_stoi=ann_stoi_SEDF, create_segments=True, drop_unk=False, target_fs=target_fs, strat_folds=10, channels=2, channel_to_resample = channel_to_resample_SEDF, target_folder=target_folder, recreate_data=True):
    print("Preparing dataset sedf.\nLoading dataset stats...")
    df_stats = get_data_annot_stats(data_path)
    df_stats["eeg_id"]=df_stats.filename.apply(lambda x: x.stem[:-4])
    print("\n\nProcessing records...")
    return prepare_dataset_with_annotations(df_stats, ann_stoi, dataset_name="SEDF", discard_labels=[""], rhythm=True, create_segments=create_segments, drop_unk=drop_unk, target_fs=target_fs, channel_to_resample=channel_to_resample_SEDF, target_folder=target_folder, recreate_data=recreate_data)


#################################################
prepare_SEDF(data_path, ann_stoi=ann_stoi_SEDF,  create_segments=True, drop_unk=False, target_fs=target_fs, channels=2, channel_to_resample = channel_to_resample_SEDF, target_folder=target_folder, recreate_data=True)
df = pd.read_pickle(open(target_root/("df"+".pkl"), "rb"))
reformat_as_memmap(df, target_root/("memmap.npy"), data_folder=target_root, annotation=True, max_len=0, delete_npys=True, col_data="data", col_lbl="label", batch_length=0)
#df = pd.read_pickle(open(target_root_spectra/("df"+".pkl"), "rb"))
#reformat_as_memmap(df, target_root_spectra/("memmap.npy"), data_folder=target_root_spectra, annotation=True, max_len=0, delete_npys=True, col_data="data", col_lbl="label", batch_length=0)
