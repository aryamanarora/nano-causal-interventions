# CODE IS DEPRECATED, WON'T WORK!

import random
from collections import defaultdict

from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from patch.models.gpt2 import create_gpt2, GPT2
from patch.utils import *
from copy import deepcopy

# load model and dataset
config, tokenizer, gpt = create_gpt2(name="gpt2")
model = GPT2(config, gpt, verbose=False)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

# tokenize dataset
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

# get top 200 tokens in dataset
counts = defaultdict(int)
for x in tqdm(dataset):
    for tok in x["input_ids"]:
        counts[tok] += 1
top_200 = [x[0] for x in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:200]]
print([tokenizer.decode(x) for x in top_200[:10]])
top_200 = set(top_200)

# get loss mask for repeats
def get_repeats(x):
    seen = {}
    res = []
    need = []
    for i, tok in enumerate(x["input_ids"]):
        res.append(True)
        # if i + 1 >= len(x["input_ids"]):
        #     res.append(False)
        #     break
        # next = x["input_ids"][i + 1]
        # res.append(tok in seen and x["input_ids"][seen[tok] + 1] == next and next not in top_200)
        # if res[-1]:
        #     need.append(seen[tok] + 1)
        # else:
        #     need.append(-1)
        # seen[tok] = i
    x["repeats"] = res
    x["need"] = torch.IntTensor(need)
    return x
dataset = dataset.map(get_repeats)
print(dataset[0])

# get inputs with len >= 100 and generate shuffled pairs
good_indices = []
for i in tqdm(range(len(dataset))):
    if len(dataset[i]["input_ids"]) < 101: continue
    good_indices.append(i)

random.shuffle(good_indices)
other = good_indices[::]
random.shuffle(other)

inputs = [(
    {
        "input_ids": torch.IntTensor(dataset[i]["input_ids"][:100]),
        "output_ids": torch.IntTensor(dataset[i]["input_ids"][1:101]),
        "text": dataset[i]["text"],
        "mask": torch.BoolTensor(dataset[i]["repeats"][:100]),
        "need": torch.IntTensor(dataset[i]["need"][:100])
    },
    {"input_ids": torch.IntTensor(dataset[j]["input_ids"][:100]), "output_ids": torch.IntTensor(dataset[j]["input_ids"][1:101]), "text": dataset[j]["text"]}
) for i, j in zip(good_indices, other)]

inputs_batched = []

batch_size = 10
batches = 10

for i in tqdm(range(len(inputs) // batch_size)):
    batch = []
    for j in range(i * batch_size, min(len(inputs), (i + 1) * batch_size)):
        batch.append(inputs[j])
    batch = (
        {
            "input_ids": torch.stack([x[0]["input_ids"] for x in batch]),
            "output_ids": torch.stack([x[0]["output_ids"] for x in batch]),
            "mask": torch.stack([x[0]["mask"] for x in batch]),
            "need": torch.stack([x[0]["need"] for x in batch])
        },
        {"input_ids": torch.stack([x[1]["input_ids"] for x in batch]), "output_ids": torch.stack([x[1]["output_ids"] for x in batch])}
    )
    inputs_batched.append(batch)

# scrub head 1.0
kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

saved = []
start_toks = []
caches = [{} for _ in range(batches)]
keep_cache = False

res = {}
with torch.inference_mode():
    for batch, x in enumerate(inputs_batched[:batches]):
        m = x[0]["mask"]
        expected = x[0]["output_ids"][m].long()

        model.cache = caches[batch]
        res1, cache = model(x, lambda x: 0, lambda x: False, clear_cache=False)
        caches[batch] = cache

        res1 = res1.hidden_states
        distrib1 = embed_to_distrib(model, res1, logits=False)[m]
        print(distrib1.shape, end=" ")
        indices_dim1 = torch.arange(distrib1.size(0))
        start_toks.append(format_token(tokenizer, x[0]["input_ids"][m][0]))
        print(f"{start_toks[-1]:>20}", end=": ")
        top_vals(tokenizer, distrib1[0], n=1)
        distrib1 = distrib1[indices_dim1, expected]
        saved.append(distrib1)

    for layer in range(0, config.n_layer):
        for head in range(config.n_head):
            metric = 0
            ct = 0
            for batch, x in enumerate(inputs_batched[:batches]):
                def which(path):
                    if f'a{layer}.head{head}' in path: return 1
                    return 0
                
                def which_compose(path):
                    if 'a10.head7' in path and f'a{layer}.head{head}' in path: return 1
                    return 0
                
                def branch(path):
                    if path[-1] == f'f{config.n_layer - 1}': return True
                    if f'f{config.n_layer - 1}' not in path:
                        if all([p[0] not in ('a', 'f') for p in path[:-1]]): return True
                        if path[-1] == f'a{layer}.head': return True
                    return False
                
                def branch_compose(path):
                    if 'a10.head7' not in path:
                        if path[-1] == f'f{config.n_layer - 1}': return True
                        if f'f{config.n_layer - 1}' not in path:
                            if all([p[0] not in ('a', 'f') for p in path[:-1]]): return True
                            if path[-1] == f'a10.head': return True
                    else:
                        if len(path) == 5: return True
                        if path[-1] == f'a{layer}.head': return True
                        return False

                    return False
                
                m = x[0]["mask"]
                expected = x[0]["output_ids"][m].long()

                # get outputs and convert to distrib over vocab
                model.cache = deepcopy(caches[batch])
                res2, cache = model(x, which, branch, clear_cache=False)
                res2 = res2.hidden_states
                distrib2 = embed_to_distrib(model, res2, logits=False)[m]

                # get the output probs between the two
                d2 = distrib2[torch.arange(distrib2.size(0)), expected]

                # get difference
                # diff = (d2 - saved[batch])
                # ct += d2.size(0)
                # metric += diff.sum().item()

                # print(f"{start_toks[batch]:>20} {diff[0]:>10.5f}", end=": ")
                # top_vals(tokenizer, distrib2[0], n=1)
                
                # get KL divergence between distribs
                loss = kl(distrib1, distrib2)
                metric += loss.item()

            res[(layer, head)] = metric / ct
            print(f"{layer:<5} {head:<5} {metric / ct:>20.5f}")

# make array
arr = np.zeros((config.n_layer, config.n_head))
for (row, col), value in res.items():
    arr[row, col] = value

# plot
fig, ax = plt.subplots()
im = ax.imshow(arr)

# add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

plt.show()