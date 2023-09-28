from transformers import BertModel, BertTokenizer, BertConfig, BertLayer
import torch
import torch.nn as nn
from functools import partial
from patch.utils import *
from copy import deepcopy
import math

class BERT(nn.Module):
    def __init__(self, config: BertConfig, model: BertModel, verbose: bool = False):
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
        # path
        name = f"a{i}.head"
        path.append(name)
        if self.verbose: print(' '.join(path))
        input_func = self.embed_input if i == 0 else partial(self.block_ffn, i=i-1)

        # get attn params
        cur_block: BertLayer = self.model.encoder.layer[i]
        attn = cur_block.attention
        num_heads = attn.self.num_attention_heads

        # compute inputs (cache as much as possible!)
        results = []
        child_path = Path([], name)

        if results_given is not None:
            results.append(results_given)
            child_path = Path((results_given.path,), name)
        else:

            # branch heads
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
            # branch at positions
            elif self.branch(path) == 'positions':
                for pos in range(self.input_len):
                    results.append(input_func(path + [f'{name}.pos{pos}']))
                    child_path.children.append(Path((results[-1].path,), f'{name}.pos{pos}'))
            # no branching in attn layer
            else:
                result = input_func(deepcopy(path))
                results = [result for _ in range(num_heads)]
                child_path = Path((result.path,), name)

        # check cache
        child_path = Path(tuple(child_path.children), child_path.parent)
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # ln + attn (basically combine stuff to make the correct qkv)
        # qkv: shape of (batch_size, num_heads, seq_len, head_dim)
        q, k, v = [], [], []
        if self.branch(path) in ['heads', True] and results_given is None:
            # combine outputs per-head
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

                if head == 0:
                    q = torch.zeros_like(query)
                    k = torch.zeros_like(key)
                    v = torch.zeros_like(value)

                q[:, head, :, :] = query[:, head, :, :]
                k[:, head, :, :] = key[:, head, :, :]
                v[:, head, :, :] = value[:, head, :, :]
        else:
            hidden_states = results[0].hidden_states

            # combine outputs for each position
            if self.branch(path) == 'positions':
                assert len(results) == self.input_len
                assert hidden_states.shape[1] == self.input_len
                for pos in range(self.input_len):
                    hidden_states[:, pos, :] = results[pos].hidden_states[:, pos, :]

            # qkv
            query = attn.self.transpose_for_scores(attn.self.query(hidden_states))
            key = attn.self.transpose_for_scores(attn.self.key(hidden_states))
            value = attn.self.transpose_for_scores(attn.self.value(hidden_states))
            q, k, v = query, key, value

        # attn probs
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        if attn.self.position_embedding_type == "relative_key" or attn.self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = attn.self.distance_embedding(distance + attn.self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=q.dtype)  # fp16 compatibility

            if attn.self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", q, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif attn.self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", q, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", k, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # scale scores, get probs
        attention_scores = attention_scores / math.sqrt(attn.self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attn.self.dropout(attention_probs)

        # attn output
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn.self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # result + cache
        result = ReturnValue(context_layer, child_path, attention_probs)
        if self.store_cache: self.cache[child_path] = result
        return result
    
    def block_attn(self, path, i) -> ReturnValue:
        # path
        name = f"a{i}"
        path.append(name)
        if self.verbose: print(' '.join(path))
        attn_func = partial(self.block_attn_heads, i=i)
        residual_func = self.embed_input if i == 0 else partial(self.block_ffn, i=i - 1)

        # get inputs
        results, child_path = [], Path(tuple(), name)
        if self.branch(path):
            results = [residual_func(path[:-1]), attn_func(deepcopy(path))]
            child_path = Path((Path((results[0].path,), f'{name}.resid'), Path((results[1].path,), f'{name}.head')), name)
        else:
            results = [residual_func(deepcopy(path))]
            results.append(attn_func(deepcopy(path), results_given=results[0]))
            child_path = Path((results[0].path,), name)
        
        # cache
        if child_path in self.cache:
            if self.verbose: print("    using cache", child_path)
            return self.cache[child_path]

        # ln + attn
        residual = results[0].hidden_states
        attn_output, attn_weights = results[1].hidden_states, results[1].outputs
        outputs = (None, attn_weights)
        del outputs

        # residual connection
        hidden_states = attn_output + residual

        # result + cache
        result = ReturnValue(hidden_states, child_path, None)
        if self.store_cache: self.cache[child_path] = result
        return result
    
    def block_ffn_head(self, path, i, results_given=None) -> ReturnValue:
        # path
        name = f"f{i}.head"
        path.append(name)
        if self.verbose: print(' '.join(path))
        input_func = partial(self.block_attn, i=i)

        # get inputs
        results, child_path = [], Path([], name)

        # given inputs from ffn
        if results_given is not None:
            results.append(results_given)
            child_path = Path((results_given.path,), name)
        # split by position
        elif self.branch(path) == 'positions':
            for pos in range(self.input_len):
                results.append(input_func(path + [name + f'.pos{pos}']))
                child_path.children.append(Path((results[-1].path,), f'{name}.pos{pos}'))
        # have to compute input myself
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
        # path
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
        # path
        name = "ln_final"
        path.append(name)
        if self.verbose: print(' '.join(path))

        # just the final layer norm after all blocks
        result = self.block_ffn(deepcopy(path), self.config.n_layer - 1)
        hidden_states = self.model.ln_f(result.hidden_states)

        # result
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
    
def create_bert(name="bert-base-uncased", revision="main"):
    config = BertConfig.from_pretrained(name, revision=revision)
    tokenizer = BertTokenizer.from_pretrained(name, revision=revision)
    gpt = BertModel.from_pretrained(name, config=config)
    print("loaded model")
    return config, tokenizer, gpt