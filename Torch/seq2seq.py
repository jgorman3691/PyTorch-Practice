import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # To print to TensorBoard

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
   return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
   return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenize_english, lower=True, init_token='sos', eos_token='<sos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de','.en', fields=(german,english)))

german.build_vocab(train_data, max_size=20000, min_freq=2)
english.build_vocab(train_data, max_size=20000, min_freq=2)

class Encoder(nn.Module):
   def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
      super(Encoder, self).__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers

      self.dropout = nn.Dropout(p)
      self.embedding = nn.Embedding(input_size, embedding_size)
      self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
   
   def forward(self, x):
      # x shape: (seq_length, N)
      # embedding shape adds embedding_size dimension to end of shape


      embedding = self.dropout(self.embedding(x))
      outputs, (hidden, cell) = self.rnn(embedding)

      return hidden, cell

class Decoder(nn.Module):
   def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layer, d)
      super(Decoder, self).__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers

      self.dropout = nn.Dropout(d)
      self.embedding = nn.Embedding(input_size, embedding_size)
      self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=d)
      self.fcc = nn.Linear(hidden_size, output_size)

   def forward(self, x, hidden, cell):
      # Shape of x is 
      x = x.unsqueeze(0)

      embedding = self.dropout(self.embedding(x))

      outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

      predictions = self.fc(outputs)
      predictions = predictions.squeeze3(0)
      