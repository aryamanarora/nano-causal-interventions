import torch

def print_path(tup, depth=0):
    if isinstance(tup, str):
        print('    ' * depth, tup, sep='')
        return
    print('    ' * depth, tup[-1], sep='')
    for i in range(len(tup) - 1):
        print_path(tup[i], depth + 1)

lsm = torch.nn.LogSoftmax(dim=2)
sm = torch.nn.Softmax(dim=2)

def format_token(tokenizer, tok):
    return tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n")

def embed_to_distrib(model, embed, log=False, logits=False):
    with torch.inference_mode():
        vocab = torch.matmul(embed, model.model.wte.weight.t())
        if logits: return vocab
        return lsm(vocab) if log else sm(vocab)

def top_vals(tokenizer, res, n=10):
    top_values, top_indices = torch.topk(res, n)
    # print(f"{'Index':<20} Value")
    for i in range(len(top_values)):
        tok = format_token(tokenizer, top_indices[i].item())
        print(f"{tok:<20} {top_values[i].item()}")

class ReturnValue():
    def __init__(self, hidden_states, path, outputs=None):
        self.hidden_states = hidden_states
        self.path = path[::]
        self.outputs = outputs
    
    def __str__(self):
        return f"ReturnValue(\n    h={self.hidden_states.shape},\n    p={self.path})"

