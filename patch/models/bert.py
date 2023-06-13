# TODO: everything!!!!

from transformers import BertModel, BertTokenizer, BertConfig, BertLayer
import torch
import torch.nn as nn
from torchview import draw_graph
from functools import partial
import timeit
from utils import *
import math

class BERT(nn.Module):
    def __init__(self, config: BertConfig, model: BertModel, verbose: bool = False):
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

        # path patching!
        input_choice = self.which(path)
        if self.verbose: print(' '.join(path), input_choice)
        child_path = ((input_choice,), name)
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]
        input_ids = self.inputs[input_choice].input_ids

        # metadata about inputs
        input_shape = input_ids.size()
        device = input_ids.device
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # embed
        embedding_output = self.model.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        # return
        result = ReturnValue(embedding_output, child_path)
        if self.store_cache: self.cache[child_path] = result
        return result
    
    def block_attn_heads(self, path, i, results_given=None) -> ReturnValue:
        # path structure
        name = f"a{i}.head"
        path.append(name)
        if self.verbose: print(' '.join(path))
        input_func = self.embed_input if i == 0 else partial(self.block_ffn, i=i-1)

        # get attn params
        cur_block: BertLayer = self.model.encoder.layer[i]
        attn = cur_block.attention
        num_heads = attn.self.num_attention_heads

        # compute which inputs will be needed (cache as much as possible!)
        results = []
        child_path = tuple()

        if results_given is not None:
            results.append(results_given)
            child_path = (results_given.path, name)
        else:
            # branching in attn layer (residual and attn are separate)
            if self.branch(path):
                # diff input for each head
                for head in range(num_heads):
                    head_name = name + str(head)

                    # branching of qkv in attn head
                    if self.branch(path + [head_name]):
                        results.append({
                            'q': input_func(path + [head_name, head_name + '.q']),
                            'k': input_func(path + [head_name, head_name + '.k']),
                            'v': input_func(path + [head_name, head_name + '.v'])
                        })
                        child_path += ((results[-1]['q'].path, results[-1]['k'].path, results[-1]['v'].path, head_name),)
                    else:
                        results.append(input_func(path + [head_name]))
                        child_path += ((results[-1].path, head_name),)
                child_path += (name,)
            # no branching in attn layer
            else:
                result = input_func(path)
                results = [result for _ in range(num_heads)]
                child_path = (result.path, name)
        
        # access cache
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
                    hidden_states = results[head].hidden_states
                    query = attn.self.transpose_for_scores(attn.self.query(hidden_states))
                    key = attn.self.transpose_for_scores(attn.self.key(hidden_states))
                    value = attn.self.transpose_for_scores(attn.self.value(hidden_states))
                else:
                    query = attn.self.transpose_for_scores(attn.self.query(results[head]['q'].hidden_states))
                    key = attn.self.transpose_for_scores(attn.self.key(results[head]['k'].hidden_states))
                    value = attn.self.transpose_for_scores(attn.self.value(results[head]['v'].hidden_states))

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

            query = attn.self.transpose_for_scores(attn.self.query(hidden_states))
            key = attn.self.transpose_for_scores(attn.self.key(hidden_states))
            value = attn.self.transpose_for_scores(attn.self.value(hidden_states))
            q, k, v = query, key, value

        # attn probs and outputs
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attn.self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attn.self.dropout(attention_probs)

        # value
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # result
        result = ReturnValue((context_layer,), child_path, attn_weights)
        if self.store_cache: self.cache[child_path] = result
        return result