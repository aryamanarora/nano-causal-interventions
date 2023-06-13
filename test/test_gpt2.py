from patch.models.gpt2 import create_gpt2, GPT2
import torch

config, tokenizer, gpt = create_gpt2(name="distilgpt2")
inputs = [tokenizer("Hello sus man", return_tensors="pt"), tokenizer("Hi sus man", return_tensors="pt")]
model = GPT2(config, gpt, verbose=True)
with torch.no_grad():
    true = gpt(inputs[1].input_ids).last_hidden_state

def test_unscrubbed():
    """Make sure raw model and our wrapped model give the same results"""
    with torch.no_grad():
        res, _ = model(inputs, lambda x: 1, lambda x: False)
    assert (res.hidden_states == true).all()

def test_branch():
    """Make sure we can branch on a2"""
    def which(path):
        if 'a2' in path: return 1
        return 1
    
    def branch(path):
        if path[-1] == 'a2': return True
        if path[-1] == 'a2.head': return 'positions'
        return False
    
    with torch.no_grad():
        res, _ = model(inputs, which, branch)
    assert (res.hidden_states == true).all()

def test_branch2():
    """Make sure we can branch on f2.head"""
    def which(path):
        if 'f2' in path: return 1
        return 1
    
    def branch(path):
        if path[-1] == 'f2': return True
        if path[-1] == 'f2.head': return 'positions'
        return False
    
    with torch.no_grad():
        res, _ = model(inputs, which, branch)
    assert (res.hidden_states == true).all()

def test_attn_lastpos():
    """Intervening on last position in attn should not change final output for preceding positions"""
    def which(path):
        if 'a2.head.pos2' in path: return 0
        return 1
    
    def branch(path):
        if path[-1] == 'a2': return True
        if path[-1] == 'a2.head': return 'positions'
        return False
    
    with torch.no_grad():
        res, _ = model(inputs, which, branch)
    assert (res.hidden_states[:, :2] == true[:, :2]).all()
    assert (res.hidden_states[:, 2:] != true[:, 2:]).any()

def test_attn_last2pos():
    """Intervening on last 2 position in attn should not change final output for preceding positions"""
    def which(path):
        if 'a2.head.pos2' in path: return 0
        if 'a2.head.pos1' in path: return 0
        return 1
    
    def branch(path):
        if path[-1] == 'a2': return True
        if path[-1] == 'a2.head': return 'positions'
        return False
    
    with torch.no_grad():
        res, _ = model(inputs, which, branch)
    assert (res.hidden_states[:, :1] == true[:, :1]).all()
    assert (res.hidden_states[:, 1:] != true[:, 1:]).any()

def test_ffn_lastpos():
    """Intervening on last position in ffn should not change final output for preceding positions"""
    def which(path):
        if 'f2.head.pos2' in path: return 0
        return 1
    
    def branch(path):
        if path[-1] == 'f2': return True
        if path[-1] == 'f2.head': return 'positions'
        return False
    
    with torch.no_grad():
        res, _ = model(inputs, which, branch)
    assert (res.hidden_states[:, :2] == true[:, :2]).all()
    assert (res.hidden_states[:, 2:] != true[:, 2:]).any()

def test_ffn2_lastpos():
    """Intervening on last position in 2 ffns should not change final output for preceding positions"""
    def which(path):
        if 'f2.head.pos2' in path: return 0
        if 'f1.head.pos2' in path: return 0
        return 1
    
    def branch(path):
        if path[-1] == 'f2': return True
        if path[-1] == 'f2.head': return 'positions'
        if 'f2' not in path:
            if path[-1] == 'f1': return True
            if path[-1] == 'f1.head': return 'positions'
        return False
    
    with torch.no_grad():
        res, _ = model(inputs, which, branch)
    assert (res.hidden_states[:, :2] == true[:, :2]).all()
    assert (res.hidden_states[:, 2:] != true[:, 2:]).any()
