from patch.models.gpt2 import create_gpt2, GPT2

config, tokenizer, gpt = create_gpt2(name="distilgpt2")
model = GPT2(config, gpt, verbose=False) # set to True for logs, inspecting cache accesses

inputs = [
    tokenizer("The capital of Spain is", return_tensors="pt"),
    tokenizer("The capital of Italy is", return_tensors="pt")
]

def branch(path):
    if path[-1] == "a1": return True
    return False

def which(path):
    if "a1" in path: return 1
    return 0

res, cache = model(inputs, which, branch)
dot = res.visualise_path()

# save as png
dot.format = 'png'
dot.render('test.gv', view=True)