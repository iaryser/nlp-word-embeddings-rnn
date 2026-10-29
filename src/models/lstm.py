import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    
    def __init__(self, embedding_matrix, input_dim,
                 hidden_dim, classifier_hidden_dim,
                 output_dim,
                 bidirectional, dropout_rate):
        super().__init__()
        
        self._num_directions = 2 if bidirectional else 1
        self._num_layers = 2
        self._hidden_dim = hidden_dim
          
        self.embeddings = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            self._num_layers,
            dropout=dropout_rate,
            batch_first=True, #so we can pass tensor([batch_size, seq_len, emebedding_dim])
            bidirectional=bidirectional
        )
        
        self.linear1 = nn.Linear(self._num_directions * hidden_dim, classifier_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(classifier_hidden_dim, output_dim)
        
    def _prepare_embeddings(self, batch):
        sequences = batch["input_ids"]
        lengths = batch["lengths"].cpu()
                
        embeddings = self.embeddings(sequences)

        packed_embeddings = pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        return packed_embeddings
    
    def forward(self, batch):
        x = self._prepare_embeddings(batch)
        
        packed_output, (hn, cn) = self.lstm(x) #h0 and c0 will be initilaized automatically if not provided
        #hn has size: [num_layers * 2, batch_size, hidden_dim]
        
        if self._num_directions == 2:
            final_hidden = torch.cat([hn[-2], hn[-1]], dim=1) #forward, backward
        else:
            final_hidden = hn[-1]
                
        x = self.linear1(final_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.linear2(x)

        