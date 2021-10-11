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
from sklearn.metrics import f1_score,roc_auc_score
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter
from transformers import AutoTokenizer,AutoModel,BertModel,BertForPreTraining,BertConfig
from smtokenization import SmilesTokenizer

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
    def __init__(self, bert_model, config, multi):
        super(BigModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if multi==0:
            self.classifier = nn.Linear(config.hidden_size, 2)
        else:
            self.classifier = []
            for i in range(multi):
                self.classifier.append(nn.Linear(config.hidden_size, 2))
            self.classifier = nn.ModuleList(self.classifier)
        self.multi = multi

    def forward(self, tokens, token_type_ids, attention_mask):
        pooled = self.bert(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)['pooler_output']
        encoded = self.dropout(pooled)
        if self.multi==0:
            return self.classifier(encoded)
        return [ self.classifier[i](encoded) for i in range(self.multi) ]


class Mole_dataset(Dataset):
    def __init__(self, pth_data, pth_lab, pth_text, seq, tok, rx):
        self.data = np.load(pth_data, allow_pickle=True)
        self.lab = np.load(pth_lab)
        self.seq = seq
        self.tok = tok
        if tok:
            f = open(pth_text, 'r')
            self.data = f.readlines()
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            if rx:
                self.tokenizer = SmilesTokenizer.from_pretrained('rxnfp/rxnfp/models/transformers/bert_mlm_1k_tpl')

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        if self.tok:
            index = self.seq[index]
            lab = self.lab[index]
            token = self.tokenizer.encode(self.data[index].strip('\n'))
            tok = np.zeros(64)
            att = np.zeros(64)
            tok[:min(64, len(token))] = token[:min(64, len(token))]
            att[:min(64, len(token))] = 1
            return torch.tensor(tok.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long()
        prop = random.randint(0,9)
        sq = self.seq[index]
        dat = np.zeros(32)
        sub = [102] +[ i+30700 for i in self.data[sq] ] + [103]
        dat[:min(32, len(sub))] = sub[:min(32, len(sub))]
        lab = self.lab[sq]
        att = np.zeros(32)
        att[:min(32, len(sub))] = np.ones(min(32, len(sub)))
        
        return torch.tensor(dat.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long()
        
def prepare_model_and_optimizer(args, device):
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
        
    bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
    bert_model = OldModel(bert_model0)
    
    if args.init_checkpoint=='BERT':
        con = BertConfig(vocab_size=31090,)
        bert_model = BertModel(con)
        args.tok = 1
        model = BigModel(bert_model, config, args.multi)
    elif args.init_checkpoint=='rxnfp':
        bert_model =  BertModel.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
        args.pth_data += 'rxnfp/'
        config.hidden = 256
        args.tok = 1
        model = BigModel(bert_model, config, args.multi)
        args.rx = 1
    elif args.init_checkpoint is None:
        args.tok = 1
        model = BigModel(bert_model0.bert, config, args.multi)
    else:
        pt = torch.load(args.init_checkpoint)
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
            pretrained_dict = {k[7:]: v for k, v in pt.items()}
            args.tok = 0
            bert_model.load_state_dict(pretrained_dict, strict=False)
            model = BigModel(bert_model, config, args.multi)
        elif 'bert.embeddings.word_embeddings.weight' in pt:
            #pretrained_dict = {k[5:]: v for k, v in pt.items()}
            args.tok = 1
            bert_model0.load_state_dict(pt, strict=True)
            model = BigModel(bert_model0.bert, config, args.multi)
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

def Eval(model, dataloader, multi):
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
            
            if multi>0:
                if y_true is None:
                    y_true = torch.transpose(lab,1,0)
                    y_score = torch.nn.Sigmoid()(logits[0][:,1]-logits[0][:,0]).unsqueeze(0)
                    for i in range(1, len(logits)):
                        y_score = torch.cat( (y_score, torch.nn.Sigmoid()(logits[i][:,1]-logits[i][:,0]).unsqueeze(0)), axis=0 )
                else:
                    y_true = torch.cat((y_true, torch.transpose(lab,1,0) ), axis=1)
                    y_score1 = torch.nn.Sigmoid()(logits[0][:,1]-logits[0][:,0]).unsqueeze(0)
                    for i in range(1, len(logits)):
                        y_score1 = torch.cat( (y_score1, torch.nn.Sigmoid()(logits[i][:,1]-logits[i][:,0]).unsqueeze(0)), axis=0 )
                    y_score = torch.cat( (y_score, y_score1.clone()), axis=1 )
            else:
                if y_true is None:
                    y_true = lab
                    y_score =  torch.nn.Sigmoid()(logits[:,1]-logits[:,0])
                else:
                    y_true = torch.cat( (y_true, lab), axis=0 )
                    y_score = torch.cat( (y_score, torch.nn.Sigmoid()(logits[:,1]-logits[:,0])), axis=0 )
    
    if multi==0:
        return roc_auc_score(y_true.squeeze().cpu(), y_score.squeeze().cpu())
    else:
        alltrue = y_true.view(-1,1).squeeze()
        allscore = y_score.view(-1,1).squeeze()
        finaltrue = None
        finalscore = None
        
        for i in range(len(alltrue)):
            if alltrue[i].item()==-1:
                continue
            if finaltrue is None:
                finaltrue = alltrue[i].unsqueeze(0)
                finalscore = allscore[i].unsqueeze(0)
            else:
                finaltrue = torch.cat((finaltrue, alltrue[i].unsqueeze(0)), axis=0)
                finalscore = torch.cat((finalscore, allscore[i].unsqueeze(0)), axis=0)
        return roc_auc_score(finaltrue.squeeze().cpu(), finalscore.squeeze().cpu())
    
def main(args):
    device = torch.device('cuda')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.task=='tox21':
        args.sm_pth = args.sm_pth + 'tox21.npy'
        args.pth_lab = args.pth_lab + 'tox21.npy'
        args.pth_data = args.pth_data + 'tox21.npy'
        args.pth_text = args.pth_text + 'tox21.txt'
        args.total_steps = 1200
        args.multi = 12
        sm_list = np.load(args.sm_pth)
        seq = np.arange(len(sm_list))
        np.random.shuffle(seq)
        scaf = []
        k = int(len(seq)/10)
        scaf.append(seq[:8*k])
        scaf.append(seq[8*k:9*k])
        scaf.append(seq[9*k:])
    elif args.task=='HIV':
        args.sm_pth = args.sm_pth + 'HIV.npy'
        args.pth_lab = args.pth_lab + 'HIV.npy'
        args.pth_data = args.pth_data + 'HIV.npy'
        args.pth_text = args.pth_text + 'HIV.txt'
        args.total_steps = 6000
        args.multi = 0
        sm_list = np.load(args.sm_pth)
        seq = np.arange(len(sm_list))
        sp = ScaffoldSplitter()
        scaf = sp.train_valid_test_split(dataset=seq, smiles_list=sm_list, frac_train=0.8,
                               frac_valid=0.1, frac_test=0.1,include_chirality=False)
    elif args.task=='BBBP':
        args.sm_pth = args.sm_pth + 'BBBP.npy'
        args.pth_lab = args.pth_lab + 'BBBP.npy'
        args.pth_data = args.pth_data + 'BBBP.npy'
        args.pth_text = args.pth_text + 'BBBP.txt'
        args.total_steps = 300
        args.multi = 0
        sm_list = np.load(args.sm_pth)
        seq = np.arange(len(sm_list))
        sp = ScaffoldSplitter()
        scaf = sp.train_valid_test_split(dataset=seq, smiles_list=sm_list, frac_train=0.8,
                               frac_valid=0.1, frac_test=0.1,include_chirality=False)
    elif args.task=='sider':
        args.sm_pth = args.sm_pth + 'sider.npy'
        args.pth_lab = args.pth_lab + 'sider.npy'
        args.pth_data = args.pth_data + 'sider.npy'
        args.pth_text = args.pth_text + 'sider.txt'
        args.total_steps = 200
        args.multi = 27
        sm_list = np.load(args.sm_pth)
        seq = np.arange(len(sm_list))
        np.random.shuffle(seq)
        scaf = []
        k = int(len(seq)/10)
        scaf.append(seq[:8*k])
        scaf.append(seq[8*k:9*k])
        scaf.append(seq[9*k:])
    
    model, optimizer = prepare_model_and_optimizer(args, device)
    
    TrainSet = Mole_dataset(args.pth_data, args.pth_lab, args.pth_text, scaf[0], args.tok, args.rx)
    DevSet = Mole_dataset(args.pth_data, args.pth_lab, args.pth_text, scaf[1], args.tok, args.rx)
    TestSet = Mole_dataset(args.pth_data, args.pth_lab, args.pth_text, scaf[2], args.tok, args.rx)
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
    loss_func = torch.nn.CrossEntropyLoss()
    global_step = 0
    tag = True
    best_acc = 0
    for epoch in range(args.epoch):
        if tag==False:
            break
        
        acc = Eval(model, dev_dataloader, args.multi)
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
            if args.multi==0:
                loss = loss_func(logits.view(-1, 2),
                        lab.cuda().view(-1),
                        )
            else:
                loss = torch.tensor(0.0).cuda()
                for i in range(len(logits)):
                    for j in range(lab.shape[0]):
                        if lab[j,i].item()==-1:
                            continue
                        loss += loss_func(
                            (logits[i].view(-1, args.num_labels)[j]).unsqueeze(0),
                            lab[j,i].cuda().view(-1),
                            )
            
            allcnt += tok.shape[0]
            sumloss += loss.item()
            loss.backward()
            if idx%2==1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if global_step>args.total_steps:
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch:', epoch, ', Loss:', sumloss/allcnt)

    acc = Eval(model, dev_dataloader, args.multi)
    print('Epoch:', args.epoch, ', DevAcc:', acc)
    if acc>best_acc:
        best_acc = acc
        torch.save(model.state_dict(), args.output)
        print('Save checkpoint ', global_step)
    model.load_state_dict(torch.load(args.output))
    acc = Eval(model, test_dataloader, args.multi)
    print('Test Acc:', acc)

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--config_file", default='bert_base_config.json', type=str,)
    parser.add_argument("--num_labels", default=2, type=int,)
    parser.add_argument("--init_checkpoint", default=None, type=str,)
    parser.add_argument("--task", default='tox21', type=str,)
    parser.add_argument("--multi", default=1, type=int,)
    parser.add_argument("--tok", default=0, type=int,)
    parser.add_argument("--rx", default=0, type=int,)
    parser.add_argument("--sm_pth", default='MoleculeNet/sm_', type=str,)
    parser.add_argument("--resume", default=-1, type=int,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=5e-6, type=float,)
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=1200, type=int,)
    parser.add_argument("--pth_data", default='MoleculeNet/sub_', type=str,)
    parser.add_argument("--pth_lab", default='MoleculeNet/lab_', type=str,)
    parser.add_argument("--pth_text", default='MoleculeNet/text_', type=str,)
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--epoch", default=20, type=int,)
    parser.add_argument("--seed", default=1011, type=int,)
    parser.add_argument("--output", default='finetune_save/ckpt_test1.pt', type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
