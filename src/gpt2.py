from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch
import torch.nn as nn
from torchview import draw_graph

class GPT2(nn.Module):
    def __init__(self, config, model: GPT2Model, verbose: bool = False):
        super().__init__()
        self.config = config
        self.model = model
        self.verbose = verbose
        self.cache = {}
    
    def embed_input(self, inputs, which, path):
        path.append("embed_input")
        if self.verbose: print(path)

        # the meat of path patching: picking which input to use!
        input_choice = which(path)
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
        path.pop()
        return hidden_states
    
    def block_attn(self, inputs, which, path, i):
        path.append(f"attn{i}")
        if self.verbose: print(path)

        hidden_states1, outputs1 = None, None
        hidden_states2, outputs2 = None, None
        if which(path[:-1]) == which(path):
            if i == 0: hidden_states1 = self.embed_input(inputs, which, path)
            else: hidden_states1, outputs1 = self.block_ffn(inputs, which, path, i - 1)
            hidden_states2, outputs2 = hidden_states1, outputs1
        elif i == 0:
            hidden_states1 = self.embed_input(inputs, which, path[:-1])
            hidden_states2 = self.embed_input(inputs, which, path)
        else:
            hidden_states1, outputs1 = self.block_ffn(inputs, which, path[:-1], i - 1)
            hidden_states2, outputs2 = self.block_ffn(inputs, which, path, i - 1)

        # get attn params
        cur_block = self.model.h[i]
        head_mask = self.model.get_head_mask(None, self.config.n_layer)[i]

        # ln + attn
        residual = hidden_states1
        hidden_states = cur_block.ln_1(hidden_states2)
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
        path.pop()
        return hidden_states, outputs

    def block_ffn(self, inputs, which, path, i):
        path.append(f"ffn{i}")
        if self.verbose: print(path)

        hidden_states1, outputs1 = None, None
        hidden_states2, outputs2 = None, None
        if which(path[:-1]) == which(path):
            hidden_states1, outputs1 = self.block_attn(inputs, which, path, i)
            hidden_states2, outputs2 = hidden_states1, outputs1
        else:
            hidden_states1, outputs1 = self.block_attn(inputs, which, path[:-1], i)
            hidden_states2, outputs2 = self.block_attn(inputs, which, path, i)

        # get ffn params
        cur_block = self.model.h[i]

        # ln + attn
        residual = hidden_states1
        hidden_states = cur_block.ln_2(hidden_states2)
        feed_forward_hidden_states = cur_block.mlp(hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        outputs = (hidden_states,) + outputs1[1:]
        path.pop()
        return hidden_states, outputs

    def final_ln(self, inputs, which, path):
        path.append("final_ln")
        if self.verbose: print(path)

        # just the final layer norm after all blocks
        hidden_states, output = self.block_ffn(inputs, which, path, self.config.n_layer - 1)
        hidden_states = self.model.ln_f(hidden_states)

        path.pop()
        return hidden_states
    
    def forward(self, inputs, which):
        hidden_states = self.final_ln(inputs, which, [])
        self.cache = {}
        return hidden_states

def create_gpt2():
    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt = GPT2Model.from_pretrained("gpt2", config=config)
    inputs = [tokenizer("Hello sus man", return_tensors="pt"), tokenizer("Hi sus man", return_tensors="pt")]

    # model_graph = draw_graph(model, input_data=inputs, save_graph=True, filename="graph.png")
    model = GPT2(config, gpt, True)
    res = model(inputs, lambda x: 1)
    res2 = gpt(inputs[1].input_ids).last_hidden_state
    res3 = model(inputs, lambda x: 1 if 'attn0' in x else 0)

    # check equality
    print(res - res2)
    print(res - res3)
    assert (res == res2).all()
    assert not (res == res3).all()

create_gpt2()