# ===== Task 3: Xây dựng mô hình RNN =====
import torch
import torch.nn as nn


class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word_to_ix["<PAD>"])

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)
        logits = self.fc(rnn_out)
        return logits


# Khởi tạo mô hình
vocab_size = len(word_to_ix)
num_tags = len(tag_to_ix)

model = SimpleRNNForTokenClassification(
    vocab_size=vocab_size,
    tagset_size=num_tags,
    embed_dim=128,
    hidden_dim=128
)

print(model)
