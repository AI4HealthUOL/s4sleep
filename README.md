# S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models

Welcome to the official GitHub repository for the paper "[S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models](https://arxiv.org/abs/2310.06715)". If you consider this repository useful for you research, we would appreciate a citation of our preprint.

@article{wang2023s4sleep,
      title={S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models}, 
      author={Tiezhi Wang and Nils Strodthoff},
      year={2023},
      eprint={2310.06715},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      journal={arXiv preprint 2310.06715}
} 

## Table of Contents
- [Setup & Environment](#setup--environment)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Classification](#classification)
- [Contribution](#contribution)
- [Acknowledgments](#acknowledgments)

## Setup & Environment

1. **Environment**: The environment details and requirements for Linux systems are provided in the `environment.yml` folder. Please ensure to set up the provided environment to avoid compatibility issues.

2. **Additional Package**: You'll need to install the `cauchy` or other related packages for the S4 model. See the official repository for more details:

	https://github.com/HazyResearch/state-spaces/


## Datasets

We utilize two primary datasets for our research:

- `Sleep-EDF`: https://www.physionet.org/content/sleep-edfx/1.0.0/
- `SHHS visit1`: https://sleepdata.org/datasets/shhs

Both datasets should be appropriately placed in the main directory or as instructed by specific scripts.

## Preprocessing

To preprocess the raw data:

Run the `preprocess.py` script:
```
python preprocess.py
```

This will process the raw data from the aforementioned datasets and prepare them for classification.

## Classification

To run the classification:

1. **Configuration**: Modify the config files located at `./conf/data` to specify the directory containing the preprocessed data.

2. **Training a sleep staging model**:
Set the desired config file using the `--config-name` flag. Example for a time-series-based model on SEDF:
	```
        python main.py --config-name=sedf_ts.yaml
        ```

You can use other config settings by specifying different `.yaml` files located in the `./classification/code/conf` directory. We provide config files for the best-performing model architectures for time series and spectrograms as input representations.

## Contribution

Feel free to submit pull requests, raise issues, or suggest enhancements to improve the project. We appreciate community contributions!

## Acknowledgments
This work partly builds on the S4 layer kindly provided by https://github.com/HazyResearch/state-spaces/

---

For any further queries or concerns, please refer to the official publication or contact the project maintainers.
