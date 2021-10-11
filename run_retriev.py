import argparse
import pickle
import os
import random
import json
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler 
from tqdm import tqdm, trange
import modeling 
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from sklearn.metrics import f1_score
from transformers import AutoTokenizer,AutoModel,BertForPreTraining, BertConfig, BertModel

class OldModel(nn.Module):
    def __init__(self, pt_model):
        super(OldModel, self).__init__()
        self.ptmodel = pt_model
        self.emb = nn.Embedding(390, 768)

    def forward(self, input_ids, attention_mask, token_type_ids):
        embs = self.ptmodel.bert.embeddings.word_embeddings(input_ids)
        msk = torch.where(input_ids>=30700)
        for k in range(msk[0].shape[0]):
            i = msk[0][k].item()
            j = msk[1][k].item()
            embs[i,j] = self.emb(input_ids[i,j]-30700)

        return self.ptmodel.bert(inputs_embeds=embs, attention_mask=attention_mask, token_type_ids=token_type_ids)



class BigModel(nn.Module):
    def __init__(self, main_model, config):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, tok, att):
        typ = torch.zeros(tok.shape).long().cuda()
        pooled_output = self.main_model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())['pooler_output']
        logits = self.dropout(pooled_output)
        return logits#_smi.mm(logits_des.t())

class Align_dataset(Dataset):
    def __init__(self, pth, iftest=0):
        self.tokdes = np.load(pth+'_tokdes.npy')
        self.attdes = np.load(pth+'_attdes.npy')
        self.iftest = iftest
        if iftest>=2:
            self.toksmi = self.tokdes
            self.attsmi = self.attdes
            self.cor = [0]
        else:
            self.toksmi = np.load(pth+'_toksmi.npy')
            self.attsmi = np.load(pth+'_attsmi.npy')
            self.cor = np.load(pth+'_cor.npy')

    def __len__(self):
        return self.toksmi.shape[0]

    def __getitem__(self, index0):
        if self.iftest>=2:
            index = index0
            index0 = 0
        else:
            pos1 = self.cor[index0]
            pos2 = self.cor[index0+1]
            index = random.randint(pos1,pos2-1)
        tokdes = self.tokdes[index]
        attdes = self.attdes[index]
        #toksmi = np.array([ti-390 if ti>=31090 else ti for ti in self.toksmi[index0]])#self.toksmi[index0]
        toksmi = self.toksmi[index0]
        attsmi = self.attsmi[index0]
        return torch.tensor(tokdes.copy()).long(), torch.tensor(attdes.copy()).long(), torch.tensor(toksmi.copy()).long(), torch.tensor(attsmi.copy()).long()


def prepare_model_and_optimizer(args, device):
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    

    bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
    bert_model = OldModel(bert_model0)
    if args.init_checkpoint is not None:
        if args.init_checkpoint=='BERT':
            con = BertConfig(vocab_size=31090,)
            bert_model = BertModel(con)
            model = BigModel(bert_model, config)
        else:
            pt = torch.load(args.init_checkpoint)
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
                pretrained_dict = {k[7:]: v for k, v in pt.items()}
                args.tok = 0
                bert_model.load_state_dict(pretrained_dict, strict=True)
                model = BigModel(bert_model, config)
            elif 'bert.embeddings.word_embeddings.weight' in pt:
                args.tok = 1
                bert_model0.load_state_dict(pt, strict=True)
                model = BigModel(bert_model0.bert, config)
    else:
        args.tok = 1
        model = BigModel(bert_model0.bert, config)
    
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

def Contra_Loss(logits_des, logits_smi, margin):
    #scores = logits_smi.mm(logits_des.t())
    scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
    diagonal = scores.diag().view(logits_smi.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_des = (margin + scores - d1).clamp(min=0)
    cost_smi = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_des = cost_des.masked_fill_(I, 0)
    cost_smi = cost_smi.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    #if self.max_violation:
    cost_des = cost_des.max(1)[0]
    cost_smi = cost_smi.max(0)[0]

    return cost_des.sum() + cost_smi.sum()



def Eval(model, dataloader, iftest=0):
    model.eval()
    with torch.no_grad():
        acc = 0
        allcnt = 0
        allout = None
        alllab = None
        des = None
        smi = None
        for batch in tqdm(dataloader):
            (tokdes, attdes, toksmi, attsmi) = batch
            logits_des = model(tokdes.cuda().squeeze(), attdes.cuda().squeeze())
            logits_smi = model(toksmi.cuda(), attsmi.cuda())
            #scores = logits_smi.mm(logits_des.t())
            scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
            argm = torch.argmax(scores, axis=1)#1
            acc += sum((argm==torch.arange(argm.shape[0]).cuda()).int()).item()
            allcnt += argm.shape[0]
            if iftest>0:
                if des is None:
                    des = logits_des
                    smi = logits_smi
                else:
                    des = torch.cat((des, logits_des), axis=0)
                    smi = torch.cat((smi, logits_smi), axis=0)
                
    if iftest==1:
        np.save('Ret/output_sent/test_smi.npy', smi.cpu())
    elif iftest==2:
        np.save('Ret/output_sent/test_des.npy', des.cpu())
    elif iftest==3:
        np.save('Ret/output_sent/filt_des.npy', des.cpu())
    print(acc/allcnt)       
    return acc/allcnt

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    model, optimizer = prepare_model_and_optimizer(args, device)
    if args.iftest==3:
        args.pth_test = args.pth_test + 'sent/filt'
    else:
        if args.tok:
            args.pth_test = args.pth_test + 'sci/test'
        else:
            args.pth_test = args.pth_test + 'sent/test'
    if args.tok:
        args.pth_train = args.pth_train + 'sci/train'
        args.pth_dev = args.pth_dev + 'sci/dev'
    else:
        args.pth_train = args.pth_train + 'sent/train'
        args.pth_dev = args.pth_dev + 'sent/dev'
    TrainSet = Align_dataset(args.pth_train)
    DevSet = Align_dataset(args.pth_dev)
    TestSet = Align_dataset(args.pth_test, iftest=args.iftest)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True)
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=False)#True
    #loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)#CrossEntropyLoss()
    loss_func = torch.nn.CrossEntropyLoss()
    global_step = 0
    tag = True
    best_acc = 0
    if args.iftest>0:
        model.load_state_dict(torch.load(args.output))
        acc = Eval(model,test_dataloader, iftest=args.iftest)
        return
    for epoch in range(args.epoch):
        if tag==False:
            break
        acc = Eval(model,dev_dataloader)
        print('Epoch:', epoch, ', DevAcc:', acc)
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.output)
            print('Save checkpoint ', global_step)
        acc = 0
        allcnt = 0
        sumloss = 0
        model.train()
        for idx,batch in enumerate(tqdm(train_dataloader)):
            (tokdes, attdes, toksmi, attsmi) = batch
            logits_des = model(tokdes.cuda(), attdes.cuda())
            logits_smi = model(toksmi.cuda(), attsmi.cuda())

            loss = Contra_Loss(logits_des, logits_smi, args.margin)
            scores = logits_smi.mm(logits_des.t())
            argm = torch.argmax(scores, axis=1)
            acc += sum((argm==torch.arange(argm.shape[0]).cuda()).int()).item()
            allcnt += argm.shape[0]
            sumloss += loss.item()
            loss.backward()
            #if idx%4==1:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step>args.total_steps:
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch:', epoch, ', Acc:', acc/allcnt, ', Loss:', sumloss/allcnt)
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
    parser.add_argument("--tok", default=0, type=int,)
    parser.add_argument("--iftest", default=0, type=int,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=5e-5, type=float,)#4
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=5000, type=int,)#3000
    parser.add_argument("--pth_train", default='Ret/', type=str,)
    parser.add_argument("--pth_dev", default='Ret/', type=str,)
    parser.add_argument("--pth_test", default='Ret/', type=str,)
    parser.add_argument("--batch_size", default=64, type=int,)
    parser.add_argument("--epoch", default=30, type=int,)
    parser.add_argument("--seed", default=99, type=int,)#73 99 108
    parser.add_argument("--margin", default=0.2, type=int,)
    parser.add_argument("--output", default='finetune_save/ckpt_retriev03.pt', type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
