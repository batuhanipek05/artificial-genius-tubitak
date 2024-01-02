from torch.utils.data import Dataset
import re
import unicodedata
import io
import torch
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
import webbrowser

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


config = AutoConfig.from_pretrained("gpt2",
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)


tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelForPreTraining.from_pretrained("gpt2", config=config)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load("train_step_last/pytorch_model.bin"))
model.cuda()

state = True

dialogue = ""
dialogue_mode = False

while state:

    ent = input("Ask me anything!:")

    if ent == "exit()":
        state = False
        continue
    if ent == "dialogue()":
        dialogue_mode = True
        continue
    if ent == "dialogue_exit()":
        dialogue_mode = False
        dialogue = ""
        continue


    if "search on google" in ent:
        search = ent.replace("search on google ", "").replace("search on google ", "")\
            .replace("search on Google ", "").replace(" ", "+").replace("search on Google ", "").strip()
        print("searching " + search.replace("+", " ") + " on google")
        webbrowser.open('https://www.google.com/search?q=' + search, new=2)
        continue
    if dialogue_mode:
        inp = " " + ent + " " + SPECIAL_TOKENS['sep_token']
        dialogue += inp
        inp = dialogue
    else:
        inp = SPECIAL_TOKENS['bos_token'] + ent + " " + SPECIAL_TOKENS['sep_token']

    generated = torch.tensor(tokenizer.encode(inp)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)
    model.eval()

    sent_num = int(input("how many answers do you want?:"))

    sample_outputs = model.generate(generated,
                                    do_sample=True,
                                    min_length=100,
                                    max_length=MAXLEN,
                                    top_k=30,
                                    top_p=0.7,
                                    temperature=0.9,
                                    repetition_penalty=2.0,
                                    num_return_sequences=sent_num
                                    )
    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=False)

        # a = len(title) + len(','.join(keywords))
        if "<|EOS|>" in text:
            text = text.split("<|EOS|>")[0]
        if i == 0:
            dialogue += text.replace("<|PAD|>", "").replace(inp, "") + " " + SPECIAL_TOKENS['sep_token']
        print("{}: {}\n\n".format(i + 1, text.replace("<|PAD|>", "")))


    a = input("was my answer right:")
    if a == "no":
        correct = input("what is the right answer:")
        with open("adjustments.txt", 'a', encoding='utf-8') as f:
            f.write("<|BOS|>" + ent + " <end_of_entry> " + correct + "<end_of_data>")
