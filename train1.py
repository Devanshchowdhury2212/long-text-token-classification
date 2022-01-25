import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn import functional as F
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from utils1 import prepare_training_data,tags
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=str, default="../model", required=False)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    return parser.parse_args()

class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer,token,attention,target):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.token = token
        self.attention = attention
        self.target = target

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "ids": torch.tensor(self.token[idx], dtype=torch.long),
            "mask": torch.tensor(self.attention[idx], dtype=torch.long),
            "targets": torch.tensor(self.target[idx], dtype=torch.long),
        }

class FeedbackModel(tez.Model):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_after = "batch"

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        sch = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.num_train_steps),
            num_training_steps=self.num_train_steps,
            num_cycles=1,
            last_epoch=-1,
        )
        return sch

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

    def monitor_metrics(self, outputs, targets, attention_mask):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, self.num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": f1_score}

    def forward(self, ids, mask, token_type_ids=None, targets=None):

        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets, attention_mask=mask)
            loss2 = self.loss(logits2, targets, attention_mask=mask)
            loss3 = self.loss(logits3, targets, attention_mask=mask)
            loss4 = self.loss(logits4, targets, attention_mask=mask)
            loss5 = self.loss(logits5, targets, attention_mask=mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            f1_1 = self.monitor_metrics(logits1, targets, attention_mask=mask)["f1"]
            f1_2 = self.monitor_metrics(logits2, targets, attention_mask=mask)["f1"]
            f1_3 = self.monitor_metrics(logits3, targets, attention_mask=mask)["f1"]
            f1_4 = self.monitor_metrics(logits4, targets, attention_mask=mask)["f1"]
            f1_5 = self.monitor_metrics(logits5, targets, attention_mask=mask)["f1"]
            f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
            metric = {"f1": f1}
            return logits, loss, metric

        return logits, loss, {}


if __name__ == "__main__":
    NUM_JOBS = 12
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token = np.load('./results/tokens_1024.npy')
    targets = np.load('./results/targets_1024.npy')
    attention = np.load('./results/attention_1024.npy')
    training_samples = prepare_training_data(args,token,targets)
    train_dataset = FeedbackDataset(training_samples, args.max_len, tokenizer,token,attention,targets)
    num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    
    model = FeedbackModel(
        model_name=args.model,
        num_train_steps=num_train_steps,
        learning_rate=args.lr,
        num_labels=len(tags) - 1,
        steps_per_epoch=len(train_dataset) / args.batch_size,
    )
    model.fit(
        train_dataset,
        train_bs=args.batch_size,
        device="cuda",
        epochs=args.epochs,
        # callbacks=[es],
        fp16=True,
        accumulation_steps=args.accumulation_steps,
    )