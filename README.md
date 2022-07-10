# GoEmotions

Implementation of [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), using 
BERT-base.

## How to run

### Install dependencies

```bash
!pip install -r requirements.txt
```

### Create input data

```bash
!python data_transformation.py --path_to_input_file "./data/train.tsv" --dataset_type "train"
!python data_transformation.py --path_to_input_file "./data/dev.tsv" --dataset_type "dev"
!python data_transformation.py --path_to_input_file "./data/test.tsv" --dataset_type "test"
```

### Run model

```bash
!python run_model.py
```

