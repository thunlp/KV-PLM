# KV-PLM

Source code for *A Deep-learning System Bridging Molecule Structure and Biomedical Text with Comprehension Comparable to Human Professionals*. Our operating system is Ubuntu 16.04. For training process, the 2080 Ti GPU is used.

## Simplified Instruction

If you want to quickly explore our job or do not have much deep-learning experience, you can simply follow the instructions in this section.

- Step 1: Download the zip or clone the repository to your workspace.
- Step 2: Download the `ckpt_ret01.pt` from [googledrive](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX?usp=sharing). Create a new directory by `mkdir save_model` and then put the downloaded model under `save_model/` directory.
- Step 3: Install Anaconda (py3) and then create a conda environment by the following command (remember to input 'y' when asked `Proceed ([y]/n)? `):
```
conda create -n KV python=3.6
conda activate KV
sh scripts/conda_environment.sh
```
  Note that there may be error when installing transformers if you're using MacOS. See [here](https://github.com/huggingface/transformers/issues/2831) for help.
- Step 4: Check the `demo_matching.py` file and set `if_cuda=False` in line 7 if there is no GPU available. Run the command:
```
python demo_matching.py
```
  And then explore the versatile reading task following the instructions of the program:
~~~bash
# input the SMILES string and the textual description you want to match and then type enter
SMILES_string: >> CC(CN)O
description: >> It is an amino alcohol and a secondary alcohol.
# the program will return a score between 0 and 1 (higher is more similar)
Matching_score =  0.8025086522102356

# automatically loop until you type control+C
SMILES_string: >>
~~~

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

KV-PLM and other pre-trained models can be downloaded from [googledrive](https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX?usp=sharing). We recommend you to download the models and put them under save\_model/ before running the code.

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

## Demo

We provide a simple demo for versatile reading exploring. Download the `ckpt_ret01.pt` and put it under `save_model/` directory. Run `python demo_matching.py` and input your SMILES string and description sentence following the instruction. Set `if_cuda=False` if there is no GPU available, and the model loading will take around 30 s.

There are some examples:
```
INP
- SMILES: CC(CN)O
- description: It is an amino alcohol and a secondary alcohol.
OUT
- matching score: 0.8025(True)

INP
- SMILES: CC(CN)O
- description: A hydroxy acid with anti-inflammatory effect.
OUT
- matching score: 0.4086(False)

INP
- SMILES: C(CCl)Cl
- description: flammable liquid with a pleasant smell.
OUT
- matching score: 0.5849(True)

INP
- SMILES: OC(=O)C1=CC=CC=C1O
- description: a clear colorless liquid with a pungent odor.
OUT
- matching score: 0.2279(False)

INP
- SMILES: OC(=O)C1=CC=CC=C1O
- description: A hydroxy acid with anti-inflammatory effect. It has a role as metabolite.
OUT
- matching score: 0.4795(True)

INP
- SMILES: OC(=O)C1=CC=CC=C1O
- description: appears as pale yellow needles, almond odor.
OUT
- matching score: 0.4499(True)
```

You can also test the matching score between two SMILES strings:
```
INP
- SMILES: OC(=O)C1=CC=CC=C1O
- description: C1=CC=C(C(=C1)C(=O)O)O
OUT
- matching score: 0.7464(True)

INP
- SMILES: C1=CC=C(C(=C1)C(=O)O)O
- description: C1(C(C(=O)OC1C(C(=O)O)O)O)O
OUT
- matching score: 0.1287(False)
```
