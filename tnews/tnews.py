import json

import torch
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from utils import DataGenerator

labels = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "106",
    "107",
    "108",
    "109",
    "110",
    "112",
    "113",
    "114",
    "115",
    "116",
]
num_labels = len(labels)
max_length = 128
batch_size = 32
epochs = 3
model_name_or_path = "hfl/chinese-roberta-wwm-ext"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cache_dir = "/mnt/f/hf/models"

def load_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            text, label = l["sentence"], l.get("label", "100")
            D.append((text, labels.index(label)))
    return D


train_data = load_data("tnews_public/train.json")
valid_data = load_data("tnews_public/dev.json")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=num_labels, cache_dir=cache_dir
)
model.to(device)


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_text, batch_labels = [], []
        for is_end, (text, label) in self.sample(random):
            batch_text.append(text)
            batch_labels.append(label)
            if len(batch_text) == self.batch_size or is_end:
                batch_data = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    padding=True,
                    max_length=max_length,
                    truncation=True,
                )
                batch_data["labels"] = torch.tensor(batch_labels, dtype=torch.long)
                yield batch_data
                batch_text, batch_labels = [], []


train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def train(model, device, train_generator, lr_scheduler, optimizer):
    model.train()
    progress_bar = tqdm(
        range(len(train_generator)),
        leave=False,
        desc="Training: ",
    )
    for batch in train_generator.forfit():
        batch.to(device)
        output = model(**batch)
        loss = output[0]
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if progress_bar.n >= len(train_generator):
            break


def eval(model, device, valid_generator, epoch):
    model.eval()
    correct = 0
    total = 0
    progress_bar = tqdm(
        range(len(valid_generator)),
        leave=False,
        desc="Evaluating: ",
    )
    with torch.no_grad():
        for batch in valid_generator:
            batch.to(device)
            labels = batch.pop("labels")
            output = model(**batch)
            pred = output[0].argmax(dim=-1)
            correct += pred.eq(labels).sum().item()
            total += pred.size(0)
            progress_bar.update(1)
    print(
        "#Epoch {:02d} | Valid: Accuracy: {}/{} ({:.0f}%)".format(
            epoch, correct, total, 100.0 * correct / total
        )
    )


optimizer = AdamW(model.parameters(), lr=5e-5)
max_train_steps = epochs * len(train_generator)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * max_train_steps),
    num_training_steps=max_train_steps,
)
for epoch in range(1, epochs + 1):
    train(model, device, train_generator, lr_scheduler, optimizer)
    eval(model, device, valid_generator, epoch)

# log!
# Epoch 01 | Valid: Accuracy: 5687/10000 (57%)
# Epoch 02 | Valid: Accuracy: 5781/10000 (58%)
# Epoch 03 | Valid: Accuracy: 5808/10000 (58%)
