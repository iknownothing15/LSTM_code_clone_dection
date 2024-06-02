import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
EMBEDDING_DIM=32
HIDDEN_DIM=16
class MyModule(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,target_size):
        super(MyModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_scores = functional.log_softmax(lstm_out.view(len(sentence), -1), dim=1)
        return tag_scores
    
def prepare_sequence(sentence,word_dict):
    # idxs=[word_dict[word] for word in sentence]
    idxs=[]
    for word in sentence:
        if word in word_dict:
            idxs.append(word_dict[word])
        else:
            idxs.append(0)
    return torch.tensor(idxs)