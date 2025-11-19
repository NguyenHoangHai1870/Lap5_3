# ===== Task 1: Load .conllu và build vocab =====
import os
from collections import Counter

def load_conllu(file_path):
    sentences = []
    cur_words = []
    cur_tags = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_words:
                    sentences.append(list(zip(cur_words, cur_tags)))
                    cur_words, cur_tags = [], []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            idx = parts[0]
            if "-" in idx or "." in idx:
                continue
            word = parts[1]
            upos = parts[3]
            cur_words.append(word)
            cur_tags.append(upos)
    if cur_words:
        sentences.append(list(zip(cur_words, cur_tags)))
    return sentences

def build_vocabs(train_sentences, min_freq=1):
    counter = Counter()
    tag_set = set()
    for sent in train_sentences:
        for w, t in sent:
            counter[w] += 1
            tag_set.add(t)
    words = [w for w, c in counter.items() if c >= min_freq]
    word_to_ix = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(words, start=2):
        word_to_ix[w] = i
    tag_to_ix = {tag: i for i, tag in enumerate(sorted(tag_set))}
    return word_to_ix, tag_to_ix

data_dir = "/content/UD_English-EWT"
train_file = os.path.join(data_dir, "en_ewt-ud-train.conllu")
dev_file   = os.path.join(data_dir, "en_ewt-ud-dev.conllu")

# Kiểm tra file tồn tại
assert os.path.exists(train_file), f"Không tìm thấy train file: {train_file}"
assert os.path.exists(dev_file), f"Không tìm thấy dev file: {dev_file}"

# Load dữ liệu
train_sents = load_conllu(train_file)
dev_sents   = load_conllu(dev_file)

print("Số câu (train):", len(train_sents))
print("Số câu (dev):", len(dev_sents))
print()

# In 3 câu mẫu đầu (word, tag)
print("Ví dụ 3 câu đầu (word, UPOS):")
for i, s in enumerate(train_sents[:3], 1):
    print(f"Sentence {i}: {s[:20]}")

# Build vocabs
word_to_ix, tag_to_ix = build_vocabs(train_sents, min_freq=1)
print("\nKích thước từ vựng (word_to_ix):", len(word_to_ix))
print("Kích thước nhãn (tag_to_ix):", len(tag_to_ix))
print("\nMột vài nhãn (tag_to_ix) sample:", list(tag_to_ix.items())[:10])
