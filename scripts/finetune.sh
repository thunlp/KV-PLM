dataset='chemprot'
task='HIV'
model='KV-PLM*'

sci='Sci-BERT'
kv='KV-PLM'
kv1='KV-PLM*'
smi='SMI-BERT'

cmd='--init_checkpoint'
if [ $model = $kv1 ]
then
	cmd=$cmd' save_model/ckpt_KV_1.pt'
elif [ $model = $kv ]
then
	cmd=$cmd' save_model/ckpt_KV.pt'
elif [ $model = $sci ]
then
	cmd=''
elif [ $model = $smi ]
then
	cmd=$cmd' save_model/ckpt_smibert.pt'
else
	cmd=$cmd' '$model
fi

cd ..

mol='moleculenet'
us='USPTO'
cp='chemprot'
cdr='bc5cdr'

if [ $dataset = $mol ]
then	
	python run_molecule.py $cmd --task $task
elif [ $dataset = $us ]
then
	python run_USPTO.py $cmd
elif [ $dataset = $cp ]
then
	python run_chem.py $cmd
else
	python run_ner.py $cmd
fi

