# KV-PLM

Source code for *A Deep-learning System Bridging Molecule Structure and Biomedical Text with Comprehension Comparable to Human Professionals*. Our operating system is Ubuntu 16.04. For training process, the 2080 Ti GPU is used.

## Requirements

- torch==1.6.0
- transformers>=3.3.1
- numpy>=1.19.3
- sklearn
- tqdm
- seqeval
- chainer\_chemistry
- subworm-nmt

We strongly suggest you to create a conda environment for this project. Installation is going to be finished in a several minutes.

```
conda create -n KV python=3.6
conda activate KV
sh scripts/conda_environment.sh
```

## Download

Before running the code, please [download](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX?usp=sharing) the pre-trained models and put them under save\_model/ .

If you are going to run the code without the pre-training models above, please choose 'Sci-BERT' mode for $MODEL.

## File Usage

The users may be going to use the files below:


- run\_chem.py: Fine-tuning code for ChemProt dataset
- run\_molecule.py: Fine-tuning code for MoleculeNet dataset
- run\_ner.py: Fine-tuning code for BC5CDR NER task
- run\_USPTO.py: Fine-tuning code for USPTO-1k few-shot dataset
- chemprot/
  - preprocess.py: Data pre-processing code for ChemProt
  - train/dev/test.txt: Raw data for ChemProt
- MoleculeNet/
  - \*.txt / \*.npy: Pre-processed data for MoleculeNet task
- NER/
  - preprocess.py: Data pre-processing code for BC5CDR
  - BC5CDR/: Raw data for BC5CDR
- Ret/
  - align\_des\_filt.txt: Molecule description text
  - align\_smiles.txt: Molecule SMILES text
  - calcu\_sent.py: PCdes\_choice test code
  - calcu\_test.py: Retrieval training evaluation code
  - preprocess.py: Data pre-processing code for versatile reading
- USPTO/
  - \*.txt / \*.npy: Pre-processed data for USPTO task
- scripts/
  - data\_preprocess.sh: Data pre-processing bash file
  - finetune.sh: Molecule Structure tasks and Natural Language tasks fine-tuning bash file
  - versatile\_reading.sh: Versatile Reading tasks fine-tuning bash file
  - smiles\_bpe.sh: Util file to generate bpe subwords


## Data Preprocessing

Switch to scripts/ directory and run the following command to pre-process the raw data:

`sh data_preprocessing.sh`

Edit the `smiles_bpe.sh` file and run it to use BPE tokenizer and get subword results.

## Downstream Tasks

We strongly recommend you to test our code with a GPU. Usually each downstream fine-tuning process takes no more than an hour.

Currently we support downstream fine-tuning and validation on rxnfp, Sci-BERT, KV-PLM and KV-PLM\*.

For **Molecule Structure Tasks** and **Natural Language Tasks**, go to scripts/ directory and modify the `finetune.sh` script according to:

```
DATASET='moleculenet'
# 'bc5cdr' for NER, 'chemprot' for RE, 'uspto' for chemical reaction classification and 'moleculenet' for molecule property classification.
TASK='BBBP'
# for MoleculeNet, we support 4 sub-tasks: 'BBBP', 'sider', 'HIV' and 'tox21'.
MODEL='Sci-BERT'
# could be 'Sci-BERT', 'KV-PLM', 'KV-PLM*', 'BERT' or 'SMI-BERT'.
...
```

Then run the script.

For **Versatile Reading Tasks**, `versatile_reading.sh` provides the training process. Modify the script according to:

```
MODEL='Sci-BERT'
# same to above
```

Run the script and you will get the fine-tuned model and encoding result for the test sets in ../Ret/output\_sent/ directory. Go to ../Ret/ and run `calcu_test.py` and `calcu_sent.py` for evaluation.

In Ret/data/ directory, you can see `PCdes_for_human.txt` which is PCdes test example that we provide to human professional annotators.

## Results

Change the random seed in the .py files and calculate the average score for the tasks above. We provide experiment results from our paper for reference:

![main result]('main_result.png')
