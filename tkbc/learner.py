# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, ComplEx_RoPE
from regularizers import N3, Lambda3

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx','ComplEx_RoPE'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)

parser.add_argument(
    '--time_eval', default=False, action="store_true",
    help="Do Time Prediction, rather than entity prediction"
)

parser.add_argument(
    '--gpu', default=False, action="store_true",
    help="Use gpu or cpu"
)


args = parser.parse_args()

dataset = TemporalDataset(args.dataset)

sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, args.rank),
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'ComplEx_RoPE': ComplEx_RoPE(sizes, args.rank)
}[args.model]
num_params = sum(p.numel() for p in model.parameters())
print(num_params)
if args.gpu:
    model = model.cuda()


opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = N3(args.emb_reg)
time_reg = Lambda3(args.time_reg)

for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)


    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}

    def avg_time(mrrs, maes, t_mrrs):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        return {'MRR': mrrs, 'MAE': maes, 't_mrr':t_mrrs}

    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        if args.time_eval:
            valid, test = [
                avg_time(*dataset.eval_time(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test']
            ]
            #print("train: MRR-T ", train['MRR'])
            print("valid: MRR-T", valid['MRR'])
            print("test: MRR-T", test['MRR'])
            #print("train: MAE ", train['MAE'])
            print("valid: MAE", valid['MAE'])
            print("test: MAE", test['MAE'])
            #print("train: tmrr ", train['t_mrr'])
            print("valid: tmrr", valid['t_mrr'])
            print("test: tmrr", test['t_mrr'])
        else:
            if dataset.has_intervals():
                valid, test, train = [
                    dataset.eval(model, split, -1 if split != 'train' else 50000)
                    for split in ['valid', 'test', 'train']
                ]
                print("train: ", train['MRR'])
                print("valid: ", valid)
                print("test: ", test)

            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("train: ", train['MRR'], train['hits@[1,3,10]'])
                print("valid: ", valid['MRR'], valid['hits@[1,3,10]'])
                print("test: ", test['MRR'], test['hits@[1,3,10]'])


