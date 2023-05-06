from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch
import torch.nn as nn
from torchview import draw_graph
from functools import partial
import timeit
from utils import *

class GPT2(nn.Module):
    def __init__(self, config, model: GPT2Model, verbose: bool = False):
        super().__init__()
        self.config = config
        self.model = model
        self.verbose = verbose
        self.cache = {}

        self.which = None
        self.branch = None
    
    def embed_input(self, path) -> ReturnValue:
        name = "emb"
        path.append(name)

        # the meat of path patching: picking which input to use!
        input_choice = self.which(path)
        if self.verbose: print(' '.join(path), input_choice)
        child_path = ((input_choice,), name)
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]
        input_ids = self.inputs[input_choice]['input_ids']

        # metadata about inputs
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        # make ids for positional embeddings
        position_ids = torch.arange(0, input_shape[-1] + 0, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # embed and add position embeddings
        inputs_embeds = self.model.wte(input_ids)
        position_embeds = self.model.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # apply dropout
        hidden_states = self.model.drop(hidden_states)
        result = ReturnValue(hidden_states, child_path)
        if self.store_cache: self.cache[child_path] = result
        return result
    
    def block_attn_heads(self, path, i, results_given=None) -> ReturnValue:
        name = f"a{i}.head"
        path.append(name)
        if self.verbose: print(' '.join(path))

        input_func = self.embed_input if i == 0 else partial(self.block_ffn, i=i-1)

        # get attn params
        cur_block = self.model.h[i]
        head_mask = self.model.get_head_mask(None, self.config.n_layer)[i]
        num_heads, head_dim = cur_block.attn.num_heads, cur_block.attn.head_dim

        # compute inputs (cache as much as possible!)
        results = []
        child_path = tuple()

        if results_given is not None:
            results.append(results_given)
            child_path = (results_given.path, name)
        else:

            # branching in attn layer (residual and attn are separate)
            if self.branch(path):
                for head in range(num_heads):
                    head_name = name + str(head)

                    # branching of qkv in attn head
                    if self.branch(path + [head_name]):
                        results.append({
                            'q': input_func(path + [head_name, head_name + '.q']),
                            'k': input_func(path + [head_name, head_name + '.k']),
                            'v': input_func(path + [head_name, head_name + '.v'])
                        })
                        if results[-1]['q'] == results[-1]['k'] == results[-1]['v']:
                            child_path += ((results[-1]['q'].path, head_name),)
                        else:
                            child_path += ((results[-1]['q'].path, results[-1]['k'].path, results[-1]['v'].path, head_name),)
                    else:
                        results.append(input_func(path + [head_name]))
                        child_path += ((results[-1].path, head_name),)

                child_path += (name,)
            else:
                # no branching in attn layer
                result = input_func(path)
                results = [result for _ in range(num_heads)]
                child_path = (result.path, name)

        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # ln + attn
        # qkv: shape of (batch_size, num_heads, seq_len, head_dim)
        q, k, v = [], [], []
        if self.branch(path) and results_given is None:
            for head in range(num_heads):
                head_name = name + str(head)

                query, key, value = None, None, None
                if isinstance(results[head], ReturnValue):
                    hidden_states = cur_block.ln_1(results[head].hidden_states)
                    query, key, value = cur_block.attn.c_attn(hidden_states).split(cur_block.attn.split_size, dim=2)
                else:
                    query, _, _ = cur_block.attn.c_attn(cur_block.ln_1(results[head]['q'].hidden_states)).split(cur_block.attn.split_size, dim=2)
                    _, key, _ = cur_block.attn.c_attn(cur_block.ln_1(results[head]['k'].hidden_states)).split(cur_block.attn.split_size, dim=2)
                    _, _, value = cur_block.attn.c_attn(cur_block.ln_1(results[head]['v'].hidden_states)).split(cur_block.attn.split_size, dim=2)

                query = cur_block.attn._split_heads(query, num_heads, head_dim)
                key = cur_block.attn._split_heads(key, num_heads, head_dim)
                value = cur_block.attn._split_heads(value, num_heads, head_dim)

                if head == 0:
                    q = torch.zeros_like(query)
                    k = torch.zeros_like(key)
                    v = torch.zeros_like(value)

                q[:, head, :, :] = query[:, head, :, :]
                k[:, head, :, :] = key[:, head, :, :]
                v[:, head, :, :] = value[:, head, :, :]
        else:
            hidden_states = cur_block.ln_1(results[0].hidden_states)

            query, key, value = cur_block.attn.c_attn(hidden_states).split(cur_block.attn.split_size, dim=2)
            query = cur_block.attn._split_heads(query, num_heads, head_dim)
            key = cur_block.attn._split_heads(key, num_heads, head_dim)
            value = cur_block.attn._split_heads(value, num_heads, head_dim)
            q, k, v = query, key, value

        # attn probs and outputs
        attn_output, attn_weights = cur_block.attn._attn(q, k, v, None, head_mask)
        attn_output = cur_block.attn._merge_heads(attn_output, num_heads, head_dim)
        attn_output = cur_block.attn.c_proj(attn_output)

        # dropout
        attn_output = cur_block.attn.resid_dropout(attn_output)

        # result
        result = ReturnValue(attn_output, child_path, attn_weights)
        if self.store_cache: self.cache[child_path] = result
        return result
    
    def block_attn(self, path, i) -> ReturnValue:
        name = f"a{i}"
        path.append(name)
        if self.verbose: print(' '.join(path))

        input_func = self.embed_input if i == 0 else partial(self.block_ffn, i=i - 1)
        results, child_path = [], []
        if self.branch(path):
            results = [input_func(path[:-1]), self.block_attn_heads(path[::], i)]
            child_path = (results[0].path, results[1].path, name)
        else:
            results = [input_func(path[::])]
            results.append(self.block_attn_heads(path[::], i, results_given=results[0]))
            child_path = (results[0].path, name)
        
        # cache
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # get attn params
        cur_block = self.model.h[i]
        head_mask = self.model.get_head_mask(None, self.config.n_layer)[i]

        # ln + attn
        residual = results[0].hidden_states
        attn_output, attn_weights = results[1].hidden_states, results[1].outputs
        outputs = (None, attn_weights)
        del outputs

        # residual connection
        hidden_states = attn_output + residual
        result = ReturnValue(hidden_states, child_path, None)
        if self.store_cache: self.cache[child_path] = result
        return result

    def block_ffn(self, path, i) -> ReturnValue:
        name = f"f{i}"
        path.append(name)
        if self.verbose: print(' '.join(path))

        input_func = partial(self.block_attn, i=i)
        results, child_path = [None, None], None
        if self.branch(path):
            results[0] = input_func(path[:-1])
            results[1] = input_func(path)
            child_path = ((results[0].path, results[1].path), name)
        else:
            results[0] = input_func(path)
            results[1] = results[0]
            child_path = ((results[0].path), name)

        # cache
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # get ffn params
        cur_block = self.model.h[i]

        # ln + attn
        residual = results[0].hidden_states
        hidden_states = cur_block.ln_2(results[1].hidden_states)
        feed_forward_hidden_states = cur_block.mlp(hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        outputs = None
        result = ReturnValue(hidden_states, child_path, outputs)
        if self.store_cache: self.cache[child_path] = result
        return result

    def final_ln(self, path) -> ReturnValue:
        name = "ln_final"
        path.append(name)
        if self.verbose: print(' '.join(path))

        # just the final layer norm after all blocks
        result = self.block_ffn(path[::], self.config.n_layer - 1)
        hidden_states = self.model.ln_f(result.hidden_states)

        result = ReturnValue(hidden_states, (result.path, name), result.outputs)
        return result
    
    def forward(self, inputs, which, branch, store_cache=True, clear_cache=True) -> ReturnValue:
        with torch.inference_mode():
            self.store_cache = store_cache
            self.which = which
            self.branch = branch
            self.inputs = inputs
            if clear_cache:
                self.cache = {}
            result = self.final_ln([])
            self.which = None
            self.branch = None
            self.inputs = None
        return result, self.cache

def create_gpt2(name="gpt2", revision="main"):
    config = GPT2Config.from_pretrained(name, revision=revision)
    tokenizer = GPT2Tokenizer.from_pretrained(name, revision=revision)
    gpt = GPT2Model.from_pretrained(name, config=config)
    print("loaded model")
    return config, tokenizer, gpt

def sanity_check():
    config, tokenizer, gpt = create_gpt2()
    inputs = [tokenizer("Hello sus man", return_tensors="pt"), tokenizer("Hi sus man", return_tensors="pt")]
    model = GPT2(config, gpt, verbose=True)
    true = gpt(inputs[1].input_ids).last_hidden_state
    res, cache = model(inputs, lambda x: 1, lambda x: False)
    assert (res.hidden_states == true).all()
    print("sanity check passed")

def branching_check():
    config, tokenizer, gpt = create_gpt2()
    inputs = [tokenizer("Hello sus man", return_tensors="pt"), tokenizer("Hi sus man", return_tensors="pt")]
    true = gpt(inputs[1].input_ids).last_hidden_state
    model = GPT2(config, gpt, verbose=True)

    def which(path):
        if 'a1' in path: return 1
        elif 'f0' in path: return 1
        return 0
    
    def branch(path):
        if path[-1] == 'a1': return True
        elif path[-1] == 'f0': return True
        return False
    
    def run():
        res, cache = model(inputs, which, branch, store_cache=True)
        print(len(cache))
        for key in list(sorted(cache.keys(), key=lambda x: len(x))):
            print(key)
        print(res.hidden_states - true)
    t = timeit.timeit(run, number=1)
    print(f"Time: {t:.5f} s")

def main():
    sanity_check()
    branching_check()

if __name__ == "__main__":
    main()