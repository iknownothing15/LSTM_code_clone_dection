import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import random
from scripts.preprocess import read_data,init,parser_single,read_dict
from scripts.model import MyModule,EMBEDDING_DIM,HIDDEN_DIM,prepare_sequence
from scripts.evaluate import evaluate,evaluate_single
from scripts.train import train

torch.random.manual_seed(1)
random.seed(114514)

if __name__ == '__main__':
    # init()
    # training_pairs,test_pairs,word_dict=read_data()
    # train(training_pairs,word_dict)
    # evaluate(test_pairs+training_pairs,word_dict)
    word_dict=read_dict()
    evaluate_single('./data/inference/1.cpp','./data/inference/2.cpp',word_dict)