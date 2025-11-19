# ===== Task 2: Dataset, collate_fn và DataLoader =====
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = word_to_ix["<PAD>"]
PAD_TAG = -100


class POSDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]

        words = [w for w, t in sent]
        tags = [t for w, t in sent]

        word_ids = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in words]
        tag_ids = [self.tag_to_ix[t] for t in tags]

        return torch.tensor(word_ids, dtype=torch.long), \
            torch.tensor(tag_ids, dtype=torch.long)


def collate_fn(batch):
    """
    batch = list of (word_tensor, tag_tensor)
    ta cần pad cả 2 theo độ dài lớn nhất trong batch.
    """
    word_seqs = [item[0] for item in batch]
    tag_seqs = [item[1] for item in batch]

    padded_words = pad_sequence(word_seqs, batch_first=True, padding_value=PAD_IDX)

    padded_tags = pad_sequence(tag_seqs, batch_first=True, padding_value=PAD_TAG)

    return padded_words, padded_tags


# Tạo dataset
train_dataset = POSDataset(train_sents, word_to_ix, tag_to_ix)
dev_dataset = POSDataset(dev_sents, word_to_ix, tag_to_ix)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

print("Train batches:", len(train_loader))
print("Dev batches:", len(dev_loader))

# Kiểm tra 1 batch mẫu
for X_batch, y_batch in train_loader:
    print("Words:", X_batch.shape)
    print("Tags :", y_batch.shape)
    break
