import io
from transformers import  AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

tokenizer.add_special_tokens(SPECIAL_TOKENS)


lines = io.open("re-retraining-1-.txt", encoding='UTF-8').read().strip() \
    .replace("’", "'").replace("<end_of_entry>", " <|SEP|> ").replace("<end_of_line>", "\n")\
    .replace("<end_of_input>", " <|SEP|> ").strip().split('<end_of_data>')


for i in lines:
    print(i)
    print(tokenizer.tokenize(i))