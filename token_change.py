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


"""def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = w.replace('\"', '\'').replace("\n", "").replace("\r", "") \
        .replace("[serious]", "").replace("(serious)", "").replace("{serious}", "") \
        .replace("[nsfw]", "").replace("(nsfw)", "").replace("{nsfw}", "")

    w = re.sub(r"([0-9?.!,¿':+%&/()=\-*\"£$])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.0-9!,¿':+%&/()=\-*_\"£$<>]+", " ", w)

    w = w.strip()
    w = '<|BOS|> ' + w + ' <|EOS|>'
    return w"""

##0, 15, 20, 13
lns = io.open("cutted/cutted_data-0-.txt", encoding='UTF-8').read().strip() \
    .replace('<end_of_data>', "").replace("<end_of_entry>", " <|SEP|> ").replace("<end_of_input>", " <|SEP|> ").split(
    '<end_of_data>')


for i in lns[:2000]:
    print(i)
    print("================/////////////////////=====================")

