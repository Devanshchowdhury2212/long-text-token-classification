import argparse
from re import S
import numpy as np
import os
from sklearn import metrics

from sklearn.metrics import f1_score
from utils1 import data_to_token
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import pandas as pd
from utils1 import target_id_map,EarlyStopping
import torch 
import tez
import torch.nn as nn
from sklearn.model_selection import KFold

def arg_passer():# Token,, Directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='allenai/longformer-large-4096',type = str,required=False)
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--output", type=str,default='model', required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument('--file_dir',default='../train',type = str,required=False)
    parser.add_argument('--csv',default='../train.csv',type = str,required=False)
    parser.add_argument('--max_length',default=1024,type = int,required=False)
    parser.add_argument('--stride',default=128,type = int,required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=2, required=False) # 8/batch_size
    parser.add_argument("--epochs", type=int, default=1, required=False)
    return parser.parse_args()



class FeedbackDataset:
    def __init__(self,samples,tokenizer,max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        input_ids = self.samples.idx['input_ids']
        input_labels = self.samples.idx['input_labels']
        input_labels = [target_id_map[z] for z in input_labels]
        other_label_id = target_id_map["O"]
        padding_label_id = target_id_map["PAD"]
        # Add starting Token (0) and Starting Label
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels

        if len(input_labels) > self.max_length-1:
            input_ids = input_ids[:self.max_length-1]
            input_labels = input_labels[:self.max_length-1]        

        input_ids =  input_ids + [self.tokenizer.eos_token_id] 
        input_labels = input_labels + [other_label_id]

        attention_mask = [1] * len(input_ids)
        pad_req = self.max_length - len(input_ids)
        if pad_req >0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_req 
            input_labels = input_labels + [padding_label_id] * pad_req 
            attention_mask = attention_mask + [0] * pad_req
        return {
            "ids": torch.tensor(input_ids, dtype=torch.int),
            "mask": torch.tensor(attention_mask, dtype=torch.int),
            "targets": torch.tensor(input_labels, dtype=torch.int),
        }

class FeedbackModel(tez.Model):
    def __init__(self, model_name,epochs,lr,batch_size,num_labels,steps_pr_epoch,num_warmup_steps):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.model_name = model_name
        self.lr = lr
        self.steps_pr_epoch = steps_pr_epoch
        self.num_warmup_steps = num_warmup_steps

        hidden_dropout_prob = 0.1
        layer_norm_eps  = 1e-6

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": hidden_dropout_prob, #.1 o
            "layer_norm_eps": layer_norm_eps,           #e-5 o
            "add_pooling_layer": False,
            "num_labels": self.num_labels,
        })

        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.linear = nn.Linear(self.config.hidden_size,self.num_labels)

    def monitor_metrics(self, outputs, targets,attention_mask):
        if targets is None:
            return {}
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_pred = outputs.view(-1,self.num_labels)
        gt = targets.view(-1).cpu().numpy()
        outputs = active_pred.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(gt[idxs],outputs[idxs],average='macro')
        return {'f1_score':f1_score}

    def loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)
        return loss

    def fetch_scheduler(self):
        # create your own scheduler
        sch = get_cosine_schedule_with_warmup(
            self.optimizer,num_warmup = int(.1*self.num_warmup_steps), num_training_steps = self.num_warmup_steps,num_cycles = 1,
            last_epoch  = -1)
        return sch

    def fetch_optimizer(self):
        # create your own optimizer
        params_optimizers = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {'params':[n for n, p in params_optimizers if not any(nd in n for nd in no_decay)],'weight_decay':.001},
            {'params':[n for n, p in params_optimizers if any(nd in n for nd in no_decay)],'weight_decay':0}
        ]
        opt = AdamW(optimizer_parameters,lr=self.learning_rate)
        return opt

    def forward(self, ids, masks, token_type_ids = None, targets=None):
        if token_type_ids == None:
            transformer_out = self.transformer(ids,masks)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        l1 = self.linear(self.dropout1(sequence_output))
        l2 = self.linear(self.dropout2(sequence_output))
        l3 = self.linear(self.dropout3(sequence_output))
        l4 = self.linear(self.dropout4(sequence_output))
        l5 = self.linear(self.dropout5(sequence_output))
        l_avg = (l1+l2+l3+l4+l5)/5
        logits = torch.softmax(l_avg,dim=-1)
        loss = 0
        if targets is not None:
            loss1  = self.loss(l1,targets,masks)
            loss2  = self.loss(l2,targets,masks)
            loss3  = self.loss(l3,targets,masks)
            loss4  = self.loss(l4,targets,masks)
            loss5  = self.loss(l5,targets,masks)
            loss = (loss1+loss2+loss3+loss4+loss5)/5
            f1_1  = self.monitor_metrics(l1,targets,masks)
            f1_2  = self.monitor_metrics(l2,targets,masks)
            f1_3  = self.monitor_metrics(l3,targets,masks)
            f1_4  = self.monitor_metrics(l4,targets,masks)
            f1_5  = self.monitor_metrics(l5,targets,masks)
            f1 = (f1_1+f1_2+f1_3+f1_4+f1_5)/5
            return logits,loss,{'f1':f1}
        return logits,loss,{}

if __name__ == '__main__':
    num_jobs = 12
    args = arg_passer()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_csv(args.csv)
    len_df = len(df)
    kfold  = KFold(n_splits = 5,random_state = 42,shuffle=True)
    for enum,(tr_idx,val_idx) in enumerate(kfold.split(range(len_df))):
        if enum == args.fold:
            df_train = df.loc[tr_idx]
            df_val = df.loc[val_idx]
            print(len(df_train))
    training_samples = data_to_token(df_train,tokenizer,args,num_jobs = num_jobs)
    valid_samples = data_to_token(df_val,tokenizer,args,num_jobs = num_jobs)

    training_dataset = FeedbackDataset(training_samples,tokenizer,args.max_length)
    num_train_steps = int(len(training_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    model = FeedbackModel(
        model_name = args.model,
        epochs = args.epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        num_labels = args.num_labels,
        steps_pr_epoch = len(training_dataset) / args.batch_size
    )
    os.makedirs(args.output,exist_ok = True)
    es = EarlyStopping(
        model_path=os.path.join(args.output, f"model_{args.fold}.bin"),
        valid_df=df_val,
        valid_samples=valid_samples,
        batch_size=args.valid_batch_size,
        patience=5,
        mode="max",
        delta=0.001,
        save_weights_only=True,
        tokenizer=tokenizer)
    model.fit(
        training_dataset,
        train_bs=args.batch_size,
        device="cuda",
        epochs=args.epochs,
        callbacks=[es],
        fp16=True,
        accumulation_steps=args.accumulation_steps)
    print('EXIT')

    