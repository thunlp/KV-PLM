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
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from sklearn.metrics import roc_auc_score
from seqeval.metrics import f1_score
#from sklearn.metrics import f1_score
from transformers import AutoModel, BertModel, BertConfig
import pdb

class BigModel(nn.Module):
    def __init__(self, bert_model, config):
        super(BigModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden, 3)
        
    def forward(self, tokens, token_type_ids, attention_mask, pos=None):
        encoded = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)['last_hidden_state']
        output = self.dropout(encoded)
        return self.classifier(output)
        
class NER_dataset(Dataset):
    def __init__(self, pth_tok, pth_lab, pth_att, pth_msk, pth_pos, iftr=False):
        self.tok = np.load(pth_tok)
        self.lab = np.load(pth_lab)
        self.att = np.load(pth_att)
        self.msk = np.load(pth_msk)

    def __len__(self):
        return self.tok.shape[0]

    def __getitem__(self, index):
        tok = self.tok[index]
        att = self.att[index]
        msk = self.msk[index]
        lab = self.lab[index]
        return torch.tensor(tok.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long(), torch.tensor(msk.copy()).long()

def prepare_model_and_optimizer(args, device):
    config = modeling.BertConfig.from_json_file(args.config_file)
    config.hidden = 768
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    
    if args.init_checkpoint is not None:
        if args.init_checkpoint=='BERT':
            con = BertConfig(vocab_size=31090,)
            bert_model = BertModel(con)
        elif args.init_checkpoint=='rxnfp':
            bert_model =  BertModel.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
            args.pth_data += 'rxnfp/'
            config.hidden = 256
        else:
            pt = torch.load(args.init_checkpoint)
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
                pretrained_dict = {k[20:]: v for k, v in pt.items()}
            elif 'bert.embeddings.word_embeddings.weight' in pt:
                pretrained_dict = {k[5:]: v for k, v in pt.items()}
            else:
                pretrained_dict = {k[12:]: v for k, v in pt.items()}
            bert_model.load_state_dict(pretrained_dict, strict=False)
    
    model = BigModel(bert_model, config)
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
    mark = {0:'O', 1:'B-MISC', 2:'I-MISC'}
    with torch.no_grad():
        acc = 0
        allcnt = 0
        bag_true = []
        bag_pred = []
        for batch in tqdm(dataloader):
            (tok, lab, att, msk) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            pos_x, pos_y = np.where(msk==1)
            bag_true += [lab[pos_x[i]][pos_y[i]].item() for i in range(len(pos_x)) ]
            bag_pred += [torch.argmax(logits[pos_x[i]][pos_y[i]]).item() for i in range(len(pos_x)) ]
    return f1_score([[mark[i] for i in bag_true]], [[mark[i] for i in bag_pred]])
    #return f1_score([mark[i] for i in bag_true], [mark[i] for i in bag_pred], average='macro')

def main(args):
    mark = {0:'O', 1:'B-MISC', 2:'I-MISC'}
    device = torch.device('cuda')
    model, optimizer = prepare_model_and_optimizer(args, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    TrainSet = NER_dataset(args.pth_data+'train_tok.npy', args.pth_data+'train_lab.npy',args.pth_data+'train_att.npy', args.pth_data+'train_msk.npy', args.pth_data+'train_pos.npy', iftr=False)
    DevSet = NER_dataset(args.pth_data+'dev_tok.npy', args.pth_data+'dev_lab.npy',args.pth_data+'dev_att.npy', args.pth_data+'dev_msk.npy', args.pth_data+'dev_pos.npy')
    TestSet =NER_dataset(args.pth_data+'test_tok.npy', args.pth_data+'test_lab.npy',args.pth_data+'test_att.npy', args.pth_data+'test_msk.npy', args.pth_data+'test_pos.npy')
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
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    global_step = 0
    tag = True
    best_acc = 0
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
        bag_true = []
        bag_pred = []
        model.train()
        
        for idx,batch in enumerate(tqdm(train_dataloader)):
            (tok, lab, att, msk) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            
            loss = loss_func(logits.view(-1, 3),
                    lab.cuda().view(-1),
                    )
            pos_x, pos_y = np.where(msk==1)
            bag_true += [lab[pos_x[i]][pos_y[i]].item() for i in range(len(pos_x)) ]
            bag_pred += [torch.argmax(logits[pos_x[i]][pos_y[i]]).item() for i in range(len(pos_x)) ]

            allcnt += tok.shape[0]
            sumloss += loss.item()
            loss.backward()
            #if idx%2==1:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step>args.total_steps:
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
       
        print('Epoch:', epoch, ', Acc:', f1_score([[mark[i] for i in bag_true]], [[mark[i] for i in bag_pred]]), ', Loss:', sumloss/allcnt)
        #print('Epoch:', epoch, ', Acc:', f1_score([mark[i] for i in bag_true], [mark[i] for i in bag_pred], average='macro'), ', Loss:', sumloss/allcnt)
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
    parser.add_argument("--init_checkpoint", default=None, type=str,)
    parser.add_argument("--pth_data", default='NER/', type=str,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=3e-5, type=float,)
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=1000, type=int,)
    parser.add_argument("--batch_size", default=16, type=int,)
    parser.add_argument("--epoch", default=4, type=int,)
    parser.add_argument("--seed", default=1234, type=int,)
    parser.add_argument("--output", default='finetune_save/ckpt_NER_1.pt', type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
