# Diverse Keyphrase Generation with Neural Unlikelihood Training 

![](https://img.shields.io/badge/python-3.7-brightgreen.svg) ![](https://img.shields.io/badge/torch-1.3.1-orange.svg)

This is the official codebase for the following paper, implemented in PyTorch:

Hareesh Bahuleyan and Layla El Asri. **Diverse Keyphrase Generation with Neural Unlikelihood Training.** COLING  2020. ArXiv Link

## Setup Instructions

1. Create and activate Python 3.7.5 virtual environment using `conda`:
    ```
    conda create --name keygen python=3.7.5
    source activate keygen
    ```

2. Install necessary packages using pip:
    ```
    pip install -r requirements.txt

    # Download spacy model
    python -m spacy download en_core_web_sm
    ```

3. Sent2Vec Installation
Sent2Vec is used in the evaluation script. 
Please install sent2vec from https://github.com/epfml/sent2vec, using the steps below:

    - Clone/Download the directory: `git clone https://github.com/epfml/sent2vec`
    - Go to sent2vec directory: `cd sent2vec/`
    - `git checkout f827d014a473aa22b2fef28d9e29211d50808d48`
    - Run `make`
    - Run `pip install cython`
    - Inside the src folder: `cd src/`
        - `python setup.py build_ext`
        - `pip install .`
    - Download a [pre-trained sent2vec model](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models). For example, we used `sent2vec_wiki_unigrams`. Finally, copy it to `data/sent2vec/wiki_unigrams.bin`

4. Data Download 
Download the pre-processed data files in JSON format by visiting [this link](https://drive.google.com/drive/folders/1OZrLwW0_M5J-zUSYFZz2qxXNnayonox8?usp=sharing):
Unzip the file and copy it to `data/`

    The data folder should now have the following structure:
    ```
    data/
    ├── kp20k_sorted/
    ├── KPTimes/
    │   └── kptimes_sorted/
    ├── sample_testset/
    ├── sent2vec/
    │   └── wiki_unigrams.bin
    └── stackexchange/
        └── se_sorted/
    ```

## Training Instructions

To train a DivKGen model using one of the configurations provided under `configurations/`: 

```
# Specify the dataset
export DATASET=kp20k

# Specify the configuration name
export EXP=copy_seq2seq_attn_mle_greedy.tgt_15.0.copy_18.0

# Run training script
allennlp train configurations/$DATASET/$EXP.jsonnet -s output/$DATASET/$EXP/ -f --include-package keyphrase_generation -o '{ "trainer": {"cuda_device": 0} }'

```
The outputs (training logs, model checkpoints, tensorboard logs) will be stored under: `output/$DATASET/$EXP`

__Notes__:
1. If your loss collapses NaN during training, this could be due to numerical underflow. The way to fix this is to edit `path/to/conda/envs/keygen/lib/python3.7/site-packages/allennlp/nn/utils.py` function `masked_log_softmax()` and change the line `vector = vector + (mask + 1e-45).log()` to `vector = vector + (mask + 1e-35).log()`.
2. Similary, find and replace all instances of `1e-45` in `path/to/conda/envs/keygen/lib/python3.7/site-packages/allennlp/models/encoder_decoders/copynet_seq2seq.py` to `1e-35`
3. During validation after every epoch, if it throws a Type Mismatch Error (`RuntimeError: "argmax_cuda" not implemented for 'Bool'`), this can be fixed by explicit type casting by changing the line `matches = (expanded_source_token_ids == expanded_target_token_ids)` to `matches = (expanded_source_token_ids == expanded_target_token_ids).int()` in `path/to/conda/envs/keygen/lib/python3.7/site-packages/allennlp/models/encoder_decoders/copynet_seq2seq.py`

## Evaluation Instructions
Finally, the evalution script can be run as follows:
1. Go to `run_eval.sh`, set the `HOME_PATH` variable. This corresponds to the `absolute/path/to/keyphrase-generation/folder`
2. Set the datasets. For instance, if we set both `EVALSET` and `DATASET` to `kp20k`, then we use the best model trained on `kp20k` to evaluate on `kp20k`. This is useful when you would like to evaluate a model trained on Dataset A on Dataset B. 
2. Next, `bash run_eval.sh` will print the quality and diversity results and also save them to `output/$DATASET/$EXP`

_Note_: In the paper, we present EditDist as a diversity evaluation metric, for which we initially used a different fuzzy string matcher. However, this codebase uses an alternative library [rapidfuzz](https://github.com/maxbachmann/rapidfuzz), which offers a similar funcitonality.

## Citation
If you found this code useful in your research, please cite:
```
@inproceedings{divKeyGen2020,
  title={Diverse Keyphrase Generation with Neural Unlikelihood Training},
  author={Bahuleyan, Hareesh and El Asri, Layla},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics (COLING)},
  year={2020}
}
```
