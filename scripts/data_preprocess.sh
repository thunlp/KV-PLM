
# chemprot RE data preprocessing

cd ../chemprot/
mkdir predata

python preprocess.py train
python preprocess.py dev
python preprocess.py test

# BC5CDR NER data preprocessing
cd ../NER/

python preprocess.py train
python preprocess.py dev
python preprocess.py test

# Retrieval data preprocessing

cd ../Ret/

python preprocess.py train
python preprocess.py dev
python preprocess.py test

python sci_preprocess.py train
python sci_preprocess.py dev
python sci_preprocess.py test
# data for USPTO and MoleculeNet has been already provided

