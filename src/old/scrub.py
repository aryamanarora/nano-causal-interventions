from transformers import GPTNeoXForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import datasets
import torch
import torch.nn as nn
import random
from tqdm import tqdm

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")

# load and tokenize wikitext-2
wikitext = datasets.load_dataset("wikitext", "wikitext-2-v1", split="test")

def tokenization(example):
    return tokenizer(example["text"])

wikitext = wikitext.map(tokenization)
print(wikitext[1])

# scrub layer 0 loss calculation
scrubbed = 0.0
for i in tqdm(range(100)):
    orig = torch.LongTensor(wikitext[i]['input_ids']).reshape(1, -1)
    other = wikitext[random.randint(0, len(wikitext))]['input_ids']
    while len(other) < orig.shape[1]:
        other = wikitext[random.randint(0, len(wikitext))]['input_ids']
    other = torch.LongTensor(other[:orig.shape[1]]).reshape(1, -1)

    def scrub(module, args):
        new_other = model.gpt_neox.layers[0](other, attention_mask=torch.ones_like(other))[0]
        print(new_other.shape, args[0].shape)

    model.gpt_neox.layers[1].register_forward_pre_hook(scrub)

    if orig.shape[1] == 0:
        continue
    loss_fn = nn.CrossEntropyLoss()
    outputs = model(input_ids=orig)

    # get logits from last hidden state
    loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), other.view(-1))
    scrubbed += loss.item()
print(scrubbed)

# unscrubbed loss calculation
unscrubbed = 0.0
for i in tqdm(range(100)):
    orig = torch.LongTensor(wikitext[i]['input_ids']).reshape(1, -1)
    if orig.shape[1] == 0: continue
    loss_fn = nn.CrossEntropyLoss()
    outputs = model(input_ids=orig)

    # get logits from last hidden state
    loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), orig.view(-1))
    unscrubbed += loss.item()
print(unscrubbed)

# baseline loss calculation
baseline = 0.0
for i in tqdm(range(100)):
    orig = torch.LongTensor(wikitext[i]['input_ids']).reshape(1, -1)
    other = wikitext[random.randint(0, len(wikitext) - 1)]['input_ids']
    while len(other) < orig.shape[1]:
        other = wikitext[random.randint(0, len(wikitext) - 1)]['input_ids']
    other = torch.LongTensor(other[:orig.shape[1]]).reshape(1, -1)

    if orig.shape[1] == 0:
        continue
    loss_fn = nn.CrossEntropyLoss()
    outputs = model(input_ids=orig)

    # get logits from last hidden state
    loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), other.view(-1))
    baseline += loss.item()
print(baseline)




# # plot attention heads in each layers
# layers = len(attns1)
# heads = attns1[0].shape[1]
# fig, axs = plt.subplots(layers, heads)
# for i in range(layers):
#     for j in range(heads):
#         axs[i][j].imshow(attns1[i][0, j, :, :].detach().numpy())

# # clear plot
# plt.clf()

# layers = len(attns1)
# heads = attns1[0].shape[1]
# fig, axs = plt.subplots(layers, heads)
# for i in range(layers):
#     for j in range(heads):
#         axs[i][j].imshow(attns1[i][0, j, :, :].detach().numpy() - attns2[i][0, j, :, :].detach().numpy())

plt.show()