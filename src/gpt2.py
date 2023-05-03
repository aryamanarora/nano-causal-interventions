from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch
import torch.nn as nn
from torchview import draw_graph

def combine_paths(path1: list, path2: list) -> list:
    if path1 == path2: return path1
    return path1 + path2

class ReturnValue():
    def __init__(self, hidden_states, path, outputs=None):
        self.hidden_states = hidden_states
        self.path = path[::]
        self.outputs = outputs

class GPT2(nn.Module):
    def __init__(self, config, model: GPT2Model, verbose: bool = False):
        super().__init__()
        self.config = config
        self.model = model
        self.verbose = verbose
        self.cache = {}

        self.which = None
        self.branch = None
    
    def embed_input(self, inputs, path) -> ReturnValue:
        name = "embed_input"
        path.append(name)
        if self.verbose: print(' '.join(path))

        # the meat of path patching: picking which input to use!
        input_choice = self.which(path)
        child_path = [(name, input_choice)]
        tup = tuple(child_path)
        if tup in self.cache:
            if self.verbose: print("    using cache")
            return self.cache[tup]
        input_ids = inputs[input_choice].input_ids

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
        self.cache[tup] = result
        path.pop()
        return result
    
    def block_attn(self, inputs, path, i) -> ReturnValue:
        name = f"attn{i}"
        path.append(name)
        if self.verbose: print(' '.join(path))

        result1, result2 = None, None
        if not self.branch(path):
            if i == 0: result1 = self.embed_input(inputs, path)
            else: result1 = self.block_ffn(inputs, path, i - 1)
            result2 = result1
        elif i == 0:
            result1 = self.embed_input(inputs, path[:-1])
            result2 = self.embed_input(inputs, path)
        else:
            result1 = self.block_ffn(inputs, path[:-1], i - 1)
            result2 = self.block_ffn(inputs, path, i - 1)
        
        # cache
        child_path = combine_paths(result1.path, result2.path) + [name]
        tup = tuple(child_path)
        if tup in self.cache:
            if self.verbose: print("    using cache")
            return self.cache[tup]

        # get attn params
        cur_block = self.model.h[i]
        head_mask = self.model.get_head_mask(None, self.config.n_layer)[i]

        # ln + attn
        residual = result1.hidden_states
        hidden_states = cur_block.ln_1(result2.hidden_states)
        attn_outputs = cur_block.attn(
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=head_mask,
            use_cache=False,
            output_attentions=True,
        )

        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # residual connection
        hidden_states = attn_output + residual
        result = ReturnValue(hidden_states, child_path, outputs)
        self.cache[tup] = result
        path.pop()
        return result

    def block_ffn(self, inputs, path, i) -> ReturnValue:
        name = f"ffn{i}"
        path.append(name)
        if self.verbose: print(' '.join(path))

        result1, result2 = None, None
        if not self.branch(path):
            result1 = self.block_attn(inputs, path, i)
            result2 = result1
        else:
            result1 = self.block_attn(inputs, path[:-1], i)
            result2 = self.block_attn(inputs, path, i)

        # cache
        child_path = combine_paths(result1.path, result2.path) + [name]
        tup = tuple(child_path)
        if tup in self.cache:
            if self.verbose: print("    using cache")
            return self.cache[tup]

        # get ffn params
        cur_block = self.model.h[i]

        # ln + attn
        residual = result1.hidden_states
        hidden_states = cur_block.ln_2(result2.hidden_states)
        feed_forward_hidden_states = cur_block.mlp(hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        outputs = (hidden_states,) + result2.outputs
        result = ReturnValue(hidden_states, child_path, outputs)
        self.cache[tup] = result
        path.pop()
        return result

    def final_ln(self, inputs, path) -> ReturnValue:
        name = "final_ln"
        path.append(name)
        if self.verbose: print(' '.join(path))

        # just the final layer norm after all blocks
        result = self.block_ffn(inputs, path, self.config.n_layer - 1)
        hidden_states = self.model.ln_f(result.hidden_states)

        result = ReturnValue(hidden_states, result.path + [name], result.outputs)
        path.pop()
        return result
    
    def forward(self, inputs, which, branch) -> ReturnValue:
        self.which = which
        self.branch = branch
        result = self.final_ln(inputs, [])
        self.cache = {}
        return result

def create_gpt2():
    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt = GPT2Model.from_pretrained("gpt2", config=config)
    inputs = [tokenizer("Hello sus man", return_tensors="pt"), tokenizer("Hi sus man", return_tensors="pt")]

    # model_graph = draw_graph(model, input_data=inputs, save_graph=True, filename="graph.png")
    model = GPT2(config, gpt, verbose=False)
    true = gpt(inputs[1].input_ids).last_hidden_state
    res = model(inputs, lambda x: 1, lambda x: False).hidden_states
    assert (res == true).all()
    print("sanity check passed")

    # all paths via attn0 get the 0 input
    # res3 = model(inputs, lambda x: 0 if 'attn0' in x else 1, lambda x: x[-1] == 'attn0')

    # all paths via attn0 and then attn1 get the 0 input
    def which(path):
        if 'attn5' in path and 'attn4' in path: return 0
        return 1
    
    def branch(path):
        if path[-1] == 'attn5': return True
        if 'attn5' in path and path[-1] == 'attn4': return True
        return False

    res4 = model(inputs, which, branch)
    print(res4.hidden_states)

create_gpt2()