from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch
import torch.nn as nn
from torchview import draw_graph

class GPT2(nn.Module):
    def __init__(self, config, model: GPT2Model):
        super().__init__()
        self.config = config
        self.model = model
    
    def embed_input(self, inputs, which, path):
        path.append("embed_input")
        print(path)
        input_ids = inputs[which(path)].input_ids

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        past_length = 0

        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.model.wte(input_ids)
        position_embeds = self.model.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.model.drop(hidden_states)
        return hidden_states

    def block(self, inputs, which, path, i):
        print(path)
        if i == 0:
            return self.embed_input(inputs, which, path + [f"block{i}"])
        
        hidden_states = self.block(inputs, which, path + [f"block{i}"], i - 1)
        head_mask = self.model.get_head_mask(None, self.config.n_layer)

        cur_block = self.model.h[i]
        outputs = cur_block(
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=head_mask[i],
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=None,
        )
        
        hidden_states = outputs[0]
        return hidden_states

    def final_ln(self, inputs, which, path):
        print(path)
        hidden_states = self.block(inputs, which, path + ["final_ln"], self.config.n_layer - 1)
        hidden_states = self.model.ln_f(hidden_states)
        return hidden_states
    
    def forward(self, inputs, which):
        hidden_states = self.final_ln(inputs, which, [])
        return hidden_states

def create_gpt2():
    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt = GPT2Model.from_pretrained("gpt2", config=config)
    inputs = [tokenizer("Hello", return_tensors="pt"), tokenizer("Hi", return_tensors="pt")]

    # model_graph = draw_graph(model, input_data=inputs, save_graph=True, filename="graph.png")
    model = GPT2(config, gpt)
    model.forward(inputs, lambda x: 1)

create_gpt2()