import argparse
import pickle
import os
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler 
from tqdm import tqdm, trange
import modeling
from smtokenization import SmilesTokenizer
from transformers import AutoModel, BertModel, BertForPreTraining, AutoTokenizer
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from sklearn.metrics import f1_score,roc_auc_score
import pdb

class OldModel(nn.Module):
    def __init__(self, pt_model):
        super(OldModel, self).__init__()
        self.ptmodel = pt_model
        self.emb = nn.Embedding(390, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        embs = self.ptmodel.bert.embeddings.word_embeddings(input_ids)
        msk = torch.where(input_ids>=30700)
        for k in range(msk[0].shape[0]):
            i = msk[0][k].item()
            j = msk[1][k].item()
            embs[i,j] = self.emb(input_ids[i,j]-30700)
        '''
        msk = (input_ids>=30700)
        embs = self.emb((input_ids-30700)*msk)
        return self.ptmodel.bert(inputs_embeds=embs, attention_mask=attention_mask, token_type_ids=token_type_ids)

class BigModel(nn.Module):
    def __init__(self, bert_model, config, num_labels):
        super(BigModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, tokens, token_type_ids, attention_mask):
        pooled = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)['pooler_output']
        encoded = self.dropout(pooled)
        return self.classifier(encoded)

class UP_dataset(Dataset):
    def __init__(self, pth_data, pth_lab, pth_text, tok, rx):
        self.tok = tok
        if tok:
            f = open(pth_text)
            self.data = f.readlines()
            f.close()
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        else:
            self.data = np.load(pth_data, allow_pickle=True)
        if rx:
            self.tokenizer = SmilesTokenizer.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
            
        self.lab = np.load(pth_lab)
       
    def __len__(self):
        return len(self.lab)

    def __getitem__(self, index):
        if self.tok:
            lab = self.lab[index]
            line = self.data[index].strip('\n')
            token = self.tokenizer.encode(line)
            dat = np.zeros(128)
            dat[:min(128, len(token))] = token[:min(128, len(token))]
            att = np.zeros(128)
            att[:min(128, len(token))] = np.ones(min(128, len(token)))
            return torch.tensor(dat.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long()

        sq = self.data[index]
        dat = np.zeros(128)#64
        sub = [102] +[ i+30700 for i in sq ] + [103]
        ans = [-100]*len(sub)
        dat[:min(128, len(sub))] = sub[:min(128, len(sub))]
        lab = self.lab[index]
        att = np.zeros(128)
        att[:min(128, len(sub))] = np.ones(min(128, len(sub)))

        return torch.tensor(dat.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long()

def prepare_model_and_optimizer(args, device):
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    #config.vocab_size = 48
   
    bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
    bert_model = OldModel(bert_model0)
    if args.init_checkpoint is None:
        args.tok = 1
        model = BigModel(bert_model0.bert, config, args.num_labels)
    elif args.init_checkpoint=='rxnfp':
        config.hidden_size = 256
        bert_model =  BertModel.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
        model = BigModel(bert_model, config, args.num_labels)
        args.rx = 1
        args.tok = 1
    else:
        pt = torch.load(args.init_checkpoint)
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
            pretrained_dict = {k[7:]: v for k, v in pt.items()}
            args.tok = 0
            bert_model.load_state_dict(pretrained_dict, strict=True)
            model = BigModel(bert_model, config, args.num_labels)
        elif 'bert.embeddings.word_embeddings.weight' in pt:
            args.tok = 1
            bert_model0.load_state_dict(pt, strict=True)
            model = BigModel(bert_model0.bert, config, args.num_labels)
    
    #model = torch.nn.DataParallel(model)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer = BertAdam(
            optimizer_grouped_parameters,
            weight_decay=args.weight_decay,
            lr=args.lr,
            warmup=args.warmup,
            t_total=args.total_steps,
            )
    return model,optimizer

def Eval(model, dataloader):
    model.eval()
    with torch.no_grad():
        acc = 0
        allcnt = 0
        y_true = None
        y_score = None
        for batch in tqdm(dataloader):
            (tok, lab, att) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            
            if y_true is None:
                y_true = lab
                y_score = torch.argmax(logits, dim=-1)
            else:
                y_true = torch.cat( (y_true, lab), axis=0 )
                y_score = torch.cat((y_score, torch.argmax(logits, dim=-1)), axis=0)

            output = torch.argmax(logits, axis=1)
            acc+=sum((lab.cuda()==output).int()).item()
            allcnt+=logits.shape[0]
    print(f1_score(y_true.cpu(), y_score.cpu(), average='micro'), f1_score(y_true.cpu(), y_score.cpu(), average='macro'))
    return acc/allcnt
   
def main(args):
    device = torch.device('cuda')
    model, optimizer = prepare_model_and_optimizer(args, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    TrainSet = UP_dataset(args.pth_data+'_train.npy', args.pth_lab+'_train.npy', args.pth_text+'_train.txt', args.tok, args.rx)
    DevSet = UP_dataset(args.pth_data+'_val.npy', args.pth_lab+'_val.npy', args.pth_text+'_val.txt', args.tok, args.rx)
    TestSet = UP_dataset(args.pth_data+'_test.npy', args.pth_lab+'_test.npy', args.pth_text+'_test.txt', args.tok, args.rx)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=True, drop_last=False)
    #loss_func = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    global_step = 0
    tag = True
    best_acc = 0
    #model.load_state_dict(torch.load(args.output))
    for epoch in range(args.epoch):
        if tag==False:
            break
        
        acc = Eval(model, dev_dataloader)
        print('Epoch:', epoch, ', DevAcc:', acc)
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.output)
            print('Save checkpoint ', global_step)
        
        acc = 0
        allcnt = 0
        sumloss = 0
        y_true = None
        y_score = None
        model.train()
        for idx,batch in enumerate(tqdm(train_dataloader)):
            (tok, lab, att) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            
            loss = loss_fn(logits.view(-1, args.num_labels),
                    lab.cuda().view(-1),
                    )
            output = torch.argmax(logits, axis=1)
            acc+=sum((lab.cuda()==output).int()).item()
            allcnt += tok.shape[0]
            sumloss += loss.item()
            loss.backward()
            if idx%4==1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if global_step>args.total_steps:
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch:', epoch, ', Acc:', acc/allcnt, ' Loss:', sumloss/allcnt)
    acc = Eval(model, dev_dataloader)
    print('Epoch:', args.epoch, ', DevAcc:', acc)
    if acc>best_acc:
        best_acc = acc
        torch.save(model.state_dict(), args.output)
        print('Save checkpoint ', global_step)
    model.load_state_dict(torch.load(args.output))
    acc = Eval(model, test_dataloader)
    print('Test Acc:', acc)

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--config_file", default='bert_base_config.json', type=str,)
    parser.add_argument("--num_labels", default=1000, type=int,)
    parser.add_argument("--init_checkpoint", default=None, type=str,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=5e-5, type=float,)
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=4000, type=int,)
    parser.add_argument("--pth_data", default='USPTO/sub', type=str,)
    parser.add_argument("--pth_lab", default='USPTO/fewlab', type=str,)
    parser.add_argument("--pth_text", default='USPTO/smiles', type=str,)
    parser.add_argument("--tok", default=0, type=int,)
    parser.add_argument("--rx", default=0, type=int,)
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--epoch", default=30, type=int,)
    parser.add_argument("--seed", default=1011, type=int,)
    parser.add_argument("--output", default='finetune_save/ckpt_UPfew1.pt', type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
