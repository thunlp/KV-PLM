MODEL='KV-PLM*'
IFT='0'

SCI='Sci-BERT'
KV='KV-PLM'
KV1='KV-PLM*'

cmd='--init_checkpoint'
if [ $MODEL = $KV1 ]
then
	cmd=$cmd' save_model/ckpt_KV_1.pt'
elif [$MODEL = $KV ]
then
	cmd=$cmd' save_model/ckpt_KV.pt'
fi

cd ..

python run_retriev.py $cmd --iftest $IFT

