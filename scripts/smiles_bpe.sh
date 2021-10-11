dataset='Mole'
task='test'
mole='BBBP'

US='USPTO'


cd ..

if [ $dataset = $US ]
then
	python preprocess_smiles.py USPTO/smiles_$task.txt tmp.txt 1
	subword-nmt apply-bpe -c bpe_coding.txt --vocabulary bpe_vocab.txt --vocabulary-threshold 80 < tmp.txt > tmpsub.txt --separator ~~
	python bpe_convert.py tmpsub.txt USPTO/sub_$task.npy 1
else
	python preprocess_smiles.py MoleculeNet/text_$mole.txt tmp.txt 0
        subword-nmt apply-bpe -c bpe_coding.txt --vocabulary bpe_vocab.txt --vocabulary-threshold 80 < tmp.txt > tmpsub.txt --separator ~~
        python bpe_convert.py tmpsub.txt MoleculeNet/sub_$mole.npy 0
fi

