from transformers import BertTokenizer, BertForPreTraining
import sys
import torch
import torch.nn as nn
import numpy as np

if_cuda = True

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class BigModel(nn.Module):
    def __init__(self, main_model):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(0.1)

    def forward(self, tok, att, cud=True):
        typ = torch.zeros(tok.shape).long()
        if cud:
            typ = typ.cuda()
        pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
        logits = self.dropout(pooled_output)
        return logits

bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
model = BigModel(bert_model0.bert)
model.load_state_dict(torch.load('save_model/ckpt_ret01.pt'))
if if_cuda:
    model = model.cuda()
model.eval()
while True:
    SM = input("SMILES string: ")
    txt = input("description: ")
    inp_SM = tokenizer.encode(SM)#[i+30700 for i in tokenizer.encode(SM)]
    inp_SM = inp_SM[:min(128, len(inp_SM))]
    inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
    att_SM = torch.ones(inp_SM.shape).long()

    inp_txt = tokenizer.encode(txt)
    inp_txt = inp_txt[:min(128, len(inp_txt))]
    inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0)
    att_txt = torch.ones(inp_txt.shape).long()

    if if_cuda:
        inp_SM = inp_SM.cuda()
        att_SM = att_SM.cuda()
        inp_txt = inp_txt.cuda()
        att_txt = att_txt.cuda()

    with torch.no_grad():
        logits_des = model(inp_txt, att_txt, if_cuda)
        logits_smi = model(inp_SM, att_SM, if_cuda)
        score = torch.cosine_similarity(logits_des, logits_smi, dim=-1)
        print('Matching score = ', score[0].item())
        print('\n')

