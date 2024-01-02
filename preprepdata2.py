from torch.utils.data import Dataset
import re
import unicodedata
import io
import torch
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoConfig, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

MAXLEN = 768
EPOCHS = 4
LR = 5e-4
EPS = 1e-8
WARMUP_STEPS = 1e2
SEED = 2020

SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

tokenizer.add_special_tokens(SPECIAL_TOKENS)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.strip())
    if "[serious]" in w or "(serious)" in w or "{serious}" in w:
        return "", False
    w = w.replace("[serious]", "").replace("(serious)", "").replace("{serious}", "") \
        .replace("[nsfw]", "").replace("(nsfw)", "").replace("{nsfw}", "")

    # w = re.sub(r"([0-9?.!,¿':+%&/()=\-*\"£$])", r" \1 ", w)
    # w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.0-9!,¿':+%&/()=|\-*_\"£$<>]+", " ", w)

    w = w.strip()
    return w, True


##14
lns = io.open("cutted/cutted_data-14-.txt", encoding='UTF-8').read().strip() \
    .replace('<end_of_data>', "").split(
    '\n')



def token_split(tokenized):
    returned = []
    r = []
    if len(tokenized) > 768:
        # assert tokenized[:1024] <= 1024
        returned.append(tokenizer.decode(tokenized[:768]))
        r = token_split(tokenized[576:])
    elif len(tokenized) <= 768:
        # assert len(tokenizer.encode(tokenizer.decode(tokenized)).ids) <= 1024
        returned.append(tokenizer.decode(tokenized))
    if len(r) != 0:
        for i in r:
            # assert len(i) <= 1024
            returned.append(i)
    return returned


lines = []


for i in lns:
    if "http" in i:
        continue
    if "http" in i:
        print(i)
    lines.append(i)

lines2 = []
for i in range(len(lines)):
    temp2 = tokenizer(lines[i])["input_ids"]
    if len(temp2) <= 768:
        lines2.append("<|BOS|>" + tokenizer.decode(temp2) + "<end_of_data>")
    else:
        temp3 = token_split(temp2)
        timm = 0
        for i in temp3:
            if timm == 0:
                lines2.append("<|BOS|>" + i + "<end_of_data>")
            else:
                lines2.append(i + "<end_of_data>")
            timm+=1


import random

random.shuffle(lines2)

with open("re-retraining-1-.txt", "a", encoding="utf-8") as f:
    for i in lines2:
        l, s = preprocess_sentence(i)
        if s:
            f.write(l)

f.close()
