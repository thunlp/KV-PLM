import argparse
import pickle
import os
import random
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler 
from tqdm import tqdm, trange
import modeling 
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, BertConfig

class chemprot_dataset(Dataset):
    def __init__(self, pth):
        self.tok = np.load(pth+'_tok.npy')
        self.att = np.load(pth+'_att.npy')
        self.lab = np.load(pth+'_lab.npy')

    def __len__(self):
        return self.tok.shape[0]

    def __getitem__(self, index):
        tok = self.tok[index]
        lab = self.lab[index]
        att = self.att[index]
        return torch.tensor(tok.copy()).long(), torch.tensor(lab.copy()).long(),torch.tensor(att.copy()).long()

def prepare_model_and_optimizer(args, device):
    config = modeling.BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased')
    model.classifier = nn.Linear(768,13)
    
    if args.init_checkpoint is not None:
        if args.init_checkpoint=='BERT':
            con = BertConfig(vocab_size=31090,)
            model = BertForSequenceClassification(con)
            model.classifier = nn.Linear(768,13)
        elif args.init_checkpoint=='rxnfp':
            model =  BertForSequenceClassification.from_pretrained('rxnfp/transformers/bert_mlm_1k_tpl')
            args.pth_train = 'chemprot/rxpredata/train'
            args.pth_dev = 'chemprot/rxpredata/dev'
            args.pth_test = 'chemprot/rxpredata/test'
            model.classifier = nn.Linear(256,13)
        else:
            pt = torch.load(args.init_checkpoint)
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in pt:
                pretrained_dict = {k[20:]: v for k, v in pt.items()}
            elif 'bert.embeddings.word_embeddings.weight' in pt:
                pretrained_dict = {k[5:]: v for k, v in pt.items()}
            else:
                pretrained_dict = {k[12:]: v for k, v in pt.items()}
            model.bert.load_state_dict(pretrained_dict, strict=False)
    
    model = torch.nn.DataParallel(model)
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
        output = None
        label = None
        for idx,batch in enumerate(tqdm(dataloader)):
            (tok, lab, att) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            logits = logits[0]#
            if output is None:
                output = torch.argmax(logits, axis=1).cpu()
                label = lab
            else:
                output = torch.cat((output, torch.argmax(logits, axis=1).cpu()), axis=-1)
                label = torch.cat((label, lab), axis=-1)
            
        acc=f1_score(output, label.squeeze(), average="micro")
    return acc


    return acc/allcnt

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    model, optimizer = prepare_model_and_optimizer(args, device)
    TrainSet = chemprot_dataset(args.pth_train)
    DevSet = chemprot_dataset(args.pth_dev)
    TestSet = chemprot_dataset(args.pth_test)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size, drop_last=True,
                                  num_workers=4, pin_memory=True)
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True)
    loss_func = torch.nn.CrossEntropyLoss()
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
        model.train()
        
        for batch in tqdm(train_dataloader):
            (tok, lab, att) = batch
            typ = torch.zeros(tok.shape).long().cuda()
            logits = model(tok.cuda(), token_type_ids=typ, attention_mask=att.cuda())
            logits = logits[0]#
            loss = loss_func(
                logits.view(-1, args.num_labels),
                lab.cuda().view(-1),
                )
            acc+=tok.shape[0]*f1_score(torch.argmax(logits, axis=1).cpu(), lab.squeeze(), average="micro")
            allcnt += tok.shape[0]
            sumloss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step>args.total_steps:
                tag = False
                break
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
    parser.add_argument("--num_labels", default=13, type=int,)
    parser.add_argument("--init_checkpoint", default=None, type=str,)
    parser.add_argument("--save_pth", default='save_model/', type=str,)
    parser.add_argument("--resume", default=-1, type=int,)
    parser.add_argument("--weight_decay", default=0, type=float,)
    parser.add_argument("--lr", default=4e-5, type=float,)
    parser.add_argument("--warmup", default=0.2, type=float,)
    parser.add_argument("--total_steps", default=2000, type=int,)
    parser.add_argument("--pth_train", default='chemprot/predata/train', type=str,)
    parser.add_argument("--pth_dev", default='chemprot/predata/dev', type=str,)
    parser.add_argument("--pth_test", default='chemprot/predata/test', type=str,)
    parser.add_argument("--batch_size", default=8, type=int,)
    parser.add_argument("--epoch", default=5, type=int,)
    parser.add_argument("--seed", default=1011, type=int,)
    parser.add_argument("--output", default='finetune_save/ckpt_cp1.pt', type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
