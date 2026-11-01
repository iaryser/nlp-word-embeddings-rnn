import torch
import torch.nn as nn

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_matrix, input_dim, inner_dim, output_dim, dropout_rate,
                 input_mode="joint"):
        super().__init__()

        if input_mode not in ("joint", "pair"):
            raise ValueError(f"Unsupported input mode: {input_mode}")
        self._input_mode = input_mode

        self.embeddings = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0
        )

        # Pair mode scores each pooled candidate with the same scalar-output MLP.
        self.linear1 = nn.Linear(
            embedding_matrix.size(1) if input_mode == "pair" else input_dim,
            inner_dim
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(inner_dim, 1 if input_mode == "pair" else output_dim)

    def _mean_pooling(self, token_ids):
        embeddings = self.embeddings(token_ids)
        if self._input_mode == "pair":
            embeddings = embeddings.masked_fill((token_ids == 0).unsqueeze(-1), 0)

        summed = embeddings.sum(dim=1) #across seq_len
        lengths = (token_ids != 0).sum(dim=1).clamp(min=1) #No div by zero

        return summed / lengths.unsqueeze(-1) #additional dim so the division works

    def forward(self, batch):
        if self._input_mode == "pair":
            x = torch.stack([
                self._mean_pooling(batch["sequence_1"]),
                self._mean_pooling(batch["sequence_2"])
            ], dim=1)  # [batch_size, 2, embedding_dim]
            x = self.linear1(x)
            x = self.relu(x)
            x = self.dropout(x)
            return self.linear2(x).squeeze(-1)  # [batch_size, 2]

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
