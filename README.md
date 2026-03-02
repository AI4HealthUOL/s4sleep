# S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models

Welcome to the official GitHub repository for the paper "[S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models](https://arxiv.org/abs/2310.06715)". If you consider this repository useful for you research, we would appreciate a citation of our preprint.

```bibtex
@article{Wang2025,
title={S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models},
author={Tiezhi Wang and Nils Strodthoff},
journal={Computers in Biology and Medicine},
volume={187},
pages={109735},
year={2025},
issn={0010-4825},
doi={10.1016/j.compbiomed.2025.109735}
}
```

## Setup & Environment

1. **Environment**: Create the conda environment from `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate s4sleep
   ```

2. **Requirements**: Python >= 3.10, PyTorch >= 2.0, Lightning >= 2.0

3. **Additional Package**: Install the Cauchy kernel extension for the S4 model:
   ```bash
   cd extensions/cauchy && python setup.py install
   ```

## Datasets

We utilize two primary datasets for our research:

- `Sleep-EDF`: https://www.physionet.org/content/sleep-edfx/1.0.0/
- `SHHS visit1`: https://sleepdata.org/datasets/shhs

Both datasets should be appropriately placed in the main directory or as instructed by specific scripts.

## Preprocessing

To preprocess the raw data, run the respective preprocessing scripts:

**Sleep-EDF:**
```bash
python preprocessing/sedf/preprocess_sedf.py --raw_path <path_to_raw_edf> --ts_path <output_path_for_timeseries> --spec_path <output_path_for_spectrograms>
```

**SHHS:**
```bash
python preprocessing/shhs/preprocess_shhs.py --raw_path <path_to_raw_shhs> --ts_path <output_path_for_timeseries> --spec_path <output_path_for_spectrograms>
```

## Classification

To run the classification:

1. **Configuration**: The config files use a modular Hydra structure. Main configs are in `./conf/` with sub-configs in `./conf/ts/` for model components.

2. **Training a sleep staging model**:
   ```bash
   # Time-series model on Sleep-EDF
   python main.py --config-name=sedf_ts
   
   # Spectrogram model on Sleep-EDF
   python main.py --config-name=sedf_spec
   
   # Override specific parameters
   python main.py --config-name=sedf_ts trainer.max_epochs=100 ts/enc=ees4
   ```

3. **Outputs**: Training logs and checkpoints are saved to `./outputs/` directory.

## Acknowledgments
This work partly builds on the S4 layer kindly provided by https://github.com/HazyResearch/state-spaces/
