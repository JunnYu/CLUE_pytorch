import json
import re
from itertools import groupby

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from transformers import (
    Adafactor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from transformers.optimization import Adafactor

from utils import DataGenerator

num_labels = 10
max_length = 64
batch_size = 24
epochs = 5
model_name_or_path = "hfl/chinese-roberta-wwm-ext"
cache_dir = "/mnt/f/hf/models"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sample_split(texts, answers, candidates):
    """将样本分隔为只有一个答案的样本，并截断长度"""
    results = []
    for i, a in enumerate(answers):
        texts_a, texts_b = texts[: i + 1], texts[i + 1 :]
        offset = 3 + 4 + 1 + 4 * (len(texts_a) + len(texts_b) - 2)
        while True:
            l_a = sum([len(t) for t in texts_a])
            l_b = sum([len(t) for t in texts_b])
            if l_a + l_b > max_length - offset:
                if l_a > l_b:
                    if len(texts_a[0]) > 1:
                        texts_a[0] = texts_a[0][1:]
                    else:
                        texts_a = texts_a[1:]
                        offset -= 4
                else:
                    if len(texts_b[-1]) > 1:
                        texts_b[-1] = texts_b[-1][:-1]
                    else:
                        texts_b = texts_b[:-1]
                        offset -= 4
            else:
                break
        results.append((texts_a, texts_b, a, candidates))
    return results


def load_data(q_file, a_file=None):
    """加载数据
    格式：[(左文本, 右文本, 答案id, 候选词集)]
    """
    D = []
    with open(q_file) as fq:
        if a_file is not None:
            A = json.load(open(a_file))
        for i, l in enumerate(fq):
            l = json.loads(l)
            assert len(l["candidates"]) == num_labels
            for c in l["content"]:
                texts = re.split("#idiom\d{6}#", c)
                keys = re.findall("#idiom\d{6}#", c)
                if a_file is None:
                    answers = [(i, k, 0) for k in keys]
                else:
                    answers = [(i, k, A[k]) for k in keys]
                D.extend(sample_split(texts, answers, l["candidates"]))

    return D


# 加载数据集
train_data = load_data("chid_public/train.json", "chid_public/train_answer.json")
valid_data = load_data("chid_public/dev.json", "chid_public/dev_answer.json")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=1, cache_dir=cache_dir
)
model.to(device)


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_text_a, batch_text_b, batch_labels = [], [], []
        for is_end, (ta, tb, (_, _, a), cs) in self.sample(random):
            textb = ""
            for i, t in enumerate(ta):
                textb += t
                if i != len(ta) - 1:
                    textb += "[MASK][MASK][MASK][MASK]"
            textb += "[unused1]"
            for i, t in enumerate(tb):
                textb += t
                if i != len(tb) - 1:
                    textb += "[MASK][MASK][MASK][MASK]"

            for c in cs:
                batch_text_a.append(c)
                batch_text_b.append(textb)
                batch_labels.append(a)
            if len(batch_text_a) == self.batch_size * num_labels or is_end:
                batch = tokenizer(
                    batch_text_a, batch_text_b, return_tensors="pt", padding=True
                )
                batch["labels"] = torch.tensor(batch_labels, dtype=torch.long)
                yield batch
                batch_text_a, batch_text_b, batch_labels = [], [], []


train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def multichoice_crossentropy(y_pred, y_true):
    """多项选择的交叉熵"""
    y_true = y_true[::num_labels]
    y_pred = y_pred.reshape(-1, num_labels)
    return F.cross_entropy(y_pred, y_true)


def train(model, device, train_generator, lr_scheduler, optimizer):
    model.train()
    progress_bar = tqdm(
        range(len(train_generator)),
        leave=False,
        desc="Training: ",
    )
    for batch in train_generator.forfit():
        batch.to(device)
        labels = batch.pop("labels")
        output = model(**batch)
        loss = multichoice_crossentropy(output[0], labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if progress_bar.n >= len(train_generator):
            break


def eval(model, device, valid_generator, epoch):
    model.eval()
    progress_bar = tqdm(
        range(len(valid_generator)),
        leave=False,
        desc="Evaluating: ",
    )
    total, correct = 0, 0
    logits = np.empty((0, num_labels))
    with torch.no_grad():
        for batch in valid_generator:
            batch.pop("labels")
            batch.to(device)
            output = model(**batch)
            y_pred = output[0].reshape(-1, num_labels).cpu().numpy()
            logits = np.concatenate([logits, y_pred], axis=0)
            progress_bar.update(1)

    for _, g in groupby(valid_data, key=lambda d: d[2][0]):
        y_true = np.array([d[2][2] for d in g])
        costs = -logits[total : total + len(y_true)]
        y_pred = linear_sum_assignment(costs)[1]
        total += len(y_true)
        correct += (y_true == y_pred).sum()

    print(
        "#Epoch {:02d} | Valid: Accuracy: {}/{} ({:.0f}%)".format(
            epoch, correct, total, 100.0 * correct / total
        )
    )


optimizer = Adafactor(model.parameters(), lr=5e-4, beta1=0.9, relative_step=False)
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
