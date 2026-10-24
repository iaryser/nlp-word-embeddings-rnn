import torch
import torch.nn as nn

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_matrix, input_dim, inner_dim, output_dim, dropout_rate):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0
        )

        self.linear1 = nn.Linear(input_dim, inner_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(inner_dim, output_dim)

    def _mean_pooling(self, token_ids):
        embeddings = self.embeddings(token_ids)

        summed = embeddings.sum(dim=1) #across seq_len
        lengths = (token_ids != 0).sum(dim=1).clamp(min=1) #No div by zero

        return summed / lengths.unsqueeze(-1) #additional dim so the division works

    def forward(self, batch):
        sentence_vectors = []

        for key in [
            "observation_1",
            "hypothesis_1",
            "hypothesis_2",
            "observation_2"
        ]:
            vector = self._mean_pooling(batch[key])
            sentence_vectors.append(vector)

        x = torch.cat(sentence_vectors, dim=1) #[batch_size, 4 * embedding_dim]

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.linear2(x)