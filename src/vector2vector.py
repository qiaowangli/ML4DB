#!/Users/royli/miniforge3/envs/pytorch_m1/bin/python3

import torch
import torch.nn as nn
torch.manual_seed(1)

def word_embedding(tokenized_list,total_length, embedding_size):
  embedding = nn.Embedding(total_length, embedding_size)
  input = torch.LongTensor(tokenized_list)
  return embedding(input)

