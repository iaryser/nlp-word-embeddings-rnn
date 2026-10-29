import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    
    def __init__(self, embedding_matrix, input_dim,
                 hidden_dim, classifier_hidden_dim,
                 output_dim, bidirectional, dropout_rate,
                 input_mode,
                 ):
        super().__init__()
        
        if input_mode not in ("single", "pair"):
            raise ValueError(f"Unsupported input mode: {input_mode}")
        
        self._num_directions = 2 if bidirectional else 1
        self._num_layers = 2
        self._hidden_dim = hidden_dim
        self._input_mode = input_mode
          
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
        
        linear_input_dim = self._num_directions * hidden_dim
        if self._input_mode == "pair":
            linear_input_dim *= 2

        self.linear1 = nn.Linear(linear_input_dim, classifier_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(classifier_hidden_dim, output_dim)
        
    def _prepare_embeddings(self, batch):
        if "single" == self._input_mode:
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
        
        else:
            seq_1 = batch["sequence_1"]
            seq_2 = batch["sequence_2"]
            len_1 = batch["lengths_1"]
            len_2 = batch["lengths_2"]
            
            embeddings_1 = self.embeddings(seq_1)
            embeddings_2 = self.embeddings(seq_2)
            
            packed_embeddings_1 = pack_padded_sequence(
                            embeddings_1,
                            len_1,
                            batch_first=True,
                            enforce_sorted=False
            )
            
            packed_embeddings_2 = pack_padded_sequence(
                embeddings_2,
                len_2,
                batch_first=True,
                enforce_sorted=False
            )

            return packed_embeddings_1, packed_embeddings_2
    
    def forward(self, batch):
        if "single" == self._input_mode:
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
        
        else:
            seq_1, seq_2 = self._prepare_embeddings(batch)
            
            packed_output_1, (hn_seq_1, cn_seq_1) = self.lstm(seq_1) 
            packed_output_2, (hn_seq_2, cn_seq_2) = self.lstm(seq_2)
            
            if self._num_directions == 2:
                final_hidden_seq_1 = torch.cat([hn_seq_1[-2], hn_seq_1[-1]], dim=1)
                final_hidden_seq_2 = torch.cat([hn_seq_2[-2], hn_seq_2[-1]], dim=1)
            else:
                final_hidden_seq_1 = hn_seq_1[-1]
                final_hidden_seq_2 = hn_seq_2[-1]
                
            final_hidden = torch.concat(
                [final_hidden_seq_1, final_hidden_seq_2],
                dim=1
            )
            
            x = self.linear1(final_hidden)
            x = self.relu(x)
            x = self.dropout(x)
            
            return self.linear2(x)
             

        