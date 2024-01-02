from torch.utils.data import Dataset
import re
import unicodedata
import io
import torch
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoConfig, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

config = AutoConfig.from_pretrained("gpt2",
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)

MAXLEN = 768
EPOCHS = 4
LR = 3e-4
EPS = 1e-10
WARMUP_STEPS = 1e2
SEED = 2020

SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

tokenizer.add_special_tokens(SPECIAL_TOKENS)


model = AutoModelForPreTraining.from_pretrained("gpt2", config=config)
model.resize_token_embeddings(len(tokenizer))

#model.load_state_dict(torch.load("train_step_last/pytorch_model.bin")) #uncomment in case of retraining

model.cuda()


#loading the dataset and retokenization
lines = io.open("current_training_dataset.txt", encoding='UTF-8').read().strip() \
    .replace("<end_of_entry>", " <|SEP|> ").replace("<end_of_line>", "\n")\
    .replace("<end_of_input>", " <|SEP|> ").strip().split('<end_of_data>')


print(len(lines))


class myDataset(Dataset): 

    def __init__(self, lines, tokenizer, randomize=True):
        title, text, keywords = [], [], []
        for i in lines:
            text.append(i)

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.title = title
        self.text = text
        self.keywords = keywords


    def __len__(self):
        return len(self.text)


    def __getitem__(self, i):
        input = self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = tokenizer(input,
                                   truncation=True,
                                   max_length=MAXLEN,
                                   padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        """
        """

        return {'label': torch.tensor(input_ids[1:]),
                'input_ids': torch.tensor(input_ids[:-1]),
                'attention_mask': torch.tensor(attention_mask[1:])}

print(len(lines))

dataset_test = myDataset(lines, tokenizer=tokenizer)
dataset2 = myDataset(lines, tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="fine_tuned_model",
    num_train_epochs=2,  
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    fp16=True,
    fp16_opt_level='01',
    warmup_steps=WARMUP_STEPS,
    learning_rate=LR,
    adam_epsilon=EPS,
    weight_decay=0.01,
    save_total_limit=1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset2,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    compute_metrics=True
)

trainer.train()
trainer.save_model()
