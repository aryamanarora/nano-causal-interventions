from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch
import torch.nn as nn
from torchview import draw_graph
from functools import partial
import timeit
from patch.utils import *
from copy import deepcopy

class GPT2(nn.Module):
    def __init__(self, config, model: GPT2Model, verbose: bool = False):
        super().__init__()
        self.config = config
        self.model = model
        self.verbose = verbose
        self.cache = {}

        self.which = lambda x: 0
        self.branch = lambda x: False
    
    def embed_input(self, path, evaluate=True) -> ReturnValue:
        name = "emb"
        path.append(name)

        # the meat of path patching: picking which input to use!
        input_choice = self.which(path)
        if self.verbose: print(' '.join(path), input_choice)
        child_path = Path((input_choice,), name)
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
        child_path = Path([], name)

        if results_given is not None:
            results.append(results_given)
            child_path = Path((results_given.path,), name)
        else:

            # branching in attn layer (residual and attn are separate)
            if self.branch(path) in ['heads', True]:
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
                            child_path.children.append(Path((results[-1]['q'].path,), head_name))
                        else:
                            child_path.children.append(Path((results[-1]['q'].path, results[-1]['k'].path, results[-1]['v'].path,), head_name))
                    else:
                        results.append(input_func(path + [head_name]))
                        child_path.children.append(Path((results[-1].path,), head_name))

            elif self.branch(path) == 'positions':
                for pos in range(self.input_len):
                    results.append(input_func(path + [f'{name}.pos{pos}']))
                    child_path.children.append(Path((results[-1].path,), f'{name}.pos{pos}'))
            else:
                # no branching in attn layer
                result = input_func(deepcopy(path))
                results = [result for _ in range(num_heads)]
                child_path = Path((result.path,), name)

        child_path = Path(tuple(child_path.children), child_path.parent)
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # ln + attn
        # qkv: shape of (batch_size, num_heads, seq_len, head_dim)
        q, k, v = [], [], []
        if self.branch(path) in ['heads', True] and results_given is None:
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
            hidden_states = results[0].hidden_states
            if self.branch(path) == 'positions':
                assert len(results) == self.input_len
                assert hidden_states.shape[1] == self.input_len
                for pos in range(self.input_len):
                    hidden_states[:, pos, :] = results[pos].hidden_states[:, pos, :]
            hidden_states = cur_block.ln_1(hidden_states)

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
        results, child_path = [], Path(tuple(), name)
        if self.branch(path):
            results = [input_func(path[:-1]), self.block_attn_heads(deepcopy(path), i)]
            child_path = Path((Path((results[0].path,), f'{name}.resid'), Path((results[1].path,), f'{name}.head')), name)
        else:
            results = [input_func(deepcopy(path))]
            results.append(self.block_attn_heads(deepcopy(path), i, results_given=results[0]))
            child_path = Path((results[0].path,), name)
        
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
    
    def block_ffn_head(self, path, i, results_given=None) -> ReturnValue:
        name = f"f{i}.head"
        path.append(name)
        if self.verbose: print(' '.join(path))
        
        input_func = partial(self.block_attn, i=i)
        results, child_path = [], Path([], name)

        if results_given is not None:
            results.append(results_given)
            child_path = Path((results_given.path,), name)
        else:
            if self.branch(path) == 'positions':
                for pos in range(self.input_len):
                    results.append(input_func(path + [name + f'.pos{pos}']))
                    child_path.children.append(Path((results[-1].path,), f'{name}.pos{pos}'))
            else:
                results.append(input_func(deepcopy(path)))
                child_path = Path((results[0].path,), name)
        child_path = Path(tuple(child_path.children), child_path.parent)

        # cache
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # get ffn params
        cur_block = self.model.h[i]
        
        # compute
        if self.branch(path) == 'positions' and results_given is None:
            for pos in range(self.input_len):
                results[0].hidden_states[:, pos, :] = results[pos].hidden_states[:, pos, :]
        hidden_states = cur_block.ln_2(results[0].hidden_states)
        feed_forward_hidden_states = cur_block.mlp(hidden_states)
        
        # result
        result = ReturnValue(feed_forward_hidden_states, child_path, None)
        if self.store_cache: self.cache[child_path] = result
        return result

    def block_ffn(self, path, i) -> ReturnValue:
        name = f"f{i}"
        path.append(name)
        if self.verbose: print(' '.join(path))

        ffn_func = partial(self.block_ffn_head, i=i)
        residual_func = partial(self.block_attn, i=i)
        results, child_path = [None, None], Path(tuple(), name)
        if self.branch(path):
            results[0] = residual_func(path[:-1])
            results[1] = ffn_func(deepcopy(path))
            child_path = Path((Path((results[0].path,), f'{name}.resid'), Path((results[1].path,), f'{name}.head')), name)
        else:
            results[0] = residual_func(path[:-1])
            results[1] = ffn_func(deepcopy(path), results_given=results[0])
            child_path = Path((results[0].path,), name)

        # cache
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # residual connection
        hidden_states = results[0].hidden_states + results[1].hidden_states
        outputs = None
        result = ReturnValue(hidden_states, child_path, outputs)
        if self.store_cache: self.cache[child_path] = result
        return result

    def final_ln(self, path) -> ReturnValue:
        name = "ln_final"
        path.append(name)
        if self.verbose: print(' '.join(path))

        # just the final layer norm after all blocks
        result = self.block_ffn(deepcopy(path), self.config.n_layer - 1)
        hidden_states = self.model.ln_f(result.hidden_states)

        result = ReturnValue(hidden_states, Path((result.path,), name), result.outputs)
        return result
    
    def forward(self, inputs, which, branch, store_cache=True, clear_cache=True) -> ReturnValue:
        with torch.inference_mode():
            self.store_cache = store_cache
            self.which = which
            self.branch = branch
            self.inputs = inputs
            self.input_len = len(inputs[0].input_ids[0])
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
        if 'a11.head.pos1' in path: return 1
        return 0
    
    def branch(path):
        if path[-1] == 'a11': return True
        if path[-1] == 'a11.head': return 'positions'
        return False
    
    def run():
        res, cache = model(inputs, which, branch, store_cache=True)
        dot = res.visualise_path()
        dot.render('test.gv', view=True)
        input()
        print(len(cache))
        print(res.hidden_states - true)
    t = timeit.timeit(run, number=1)
    print(f"Time: {t:.5f} s")

def main():
    # sanity_check()
    branching_check()

if __name__ == "__main__":
    main()