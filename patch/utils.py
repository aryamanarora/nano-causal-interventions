import torch
import graphviz
from collections import namedtuple, defaultdict
from tqdm import tqdm
from functools import partial
import pandas as pd

from plotnine import ggplot, geom_tile, aes, facet_wrap, theme, element_text
from functools import wraps

lsm = torch.nn.LogSoftmax(dim=2)
sm = torch.nn.Softmax(dim=2)

Path = namedtuple("Path", "children parent")

class LoggingDict(dict):
    def __getitem__(self, key):
        print(f"    key: {key} ({super().__getitem__(key).hidden_states.sum()})")
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        match = True
        if super().get(key) is not None:
            match = (super().__getitem__(key).hidden_states == value.hidden_states).all()
        print(f"    {'!!' if not match else ''}key: {key} ({value.hidden_states.sum()})")
        super().__setitem__(key, value)

def verbose(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        name = f.__name__ + (("_" + str(kwargs["i"])) if "i" in kwargs else "")
        if self.verbose:
            print(f'+ {name:<20}{args}, {kwargs}')
        res = f(self, *args, **kwargs)
        if self.verbose:
            print(f'- {name:<20}{args}, {kwargs}')
        return res
    return wrapper

class ReturnValue:
    def __init__(self, hidden_states: torch.Tensor, path: Path, outputs=None):
        self.hidden_states: torch.Tensor = hidden_states
        self.path: Path = path[::]
        self.outputs = outputs

    def __repr__(self):
        return f"ReturnValue(h={self.hidden_states.shape},p={self.path})"

    def __str__(self):
        return f"ReturnValue(h={self.hidden_states.shape},p={self.path})"

    def _get_nodes(self, path, parent="final"):
        edges = set()
        if isinstance(path, str) or isinstance(path, int):
            edges.add((parent, path))
        elif isinstance(path[-1], str):
            if path not in self.ct[path[-1]]:
                self.ct[path[-1]][path] = len(self.ct[path[-1]])
            child = f"{path[-1]}||{self.ct[path[-1]][path]}"
            edges.add((parent, child))
            for p in path[0]:
                edges.update(self._get_nodes(p, child))
        else:
            edges.update(self._get_nodes(path[-1], parent))
        return edges

    def get_nodes(self):
        self.ct = defaultdict(lambda: defaultdict(int))
        return self._get_nodes(self.path)

    def visualise_path(self):
        dot = graphviz.Digraph(comment="Circuit")
        edges = self.get_nodes()
        nodes = set([e[0] for e in edges] + [e[1] for e in edges])
        for n in nodes:
            dot.node(str(n), str(n).split("||")[0])
        for e in edges:
            dot.edge(str(e[0]), str(e[1]))
        return dot


# utilities for printing nicely


def print_path(tup, depth=0):
    if isinstance(tup, str):
        print("    " * depth, tup, sep="")
        return
    print("    " * depth, tup[-1], sep="")
    for i in range(len(tup) - 1):
        print_path(tup[i], depth + 1)


def format_token(tokenizer, tok):
    return tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n")


# utilities for converting between embeddings and distributions


def embed_to_distrib(model, embed, log=False, logits=False):
    """Convert an embedding to a distribution over the vocabulary"""
    with torch.inference_mode():
        vocab = torch.matmul(embed, model.model.wte.weight.t())
        if logits:
            return vocab
        return lsm(vocab) if log else sm(vocab)


def top_vals(tokenizer, res, n=10):
    """Pretty print the top n values of a distribution over the vocabulary"""
    top_values, top_indices = torch.topk(res, n)
    for i in range(len(top_values)):
        tok = format_token(tokenizer, top_indices[i].item())
        print(f"{tok:<20} {top_values[i].item()}")


# simple causal interventions that are very common


def which_node_pos(path, node, pos):
    if f"{node}.head.pos{pos}" in path or f"{node}.pos{pos}" in path:
        return 1
    return 0


def branch_node_pos(path, node, pos):
    if path[-1] == f"{node}":
        return "positions"
    if path[-1] == f"{node}.head":
        return "positions"
    return False


def intervene_node_and_pos(
    model,
    tokenizer,
    inputs,
    tokens: list[str],
    which=which_node_pos,
    branch=branch_node_pos,
    plot=False,
    pause=False,
    nodes=None,
    pos=None
):
    # nodes
    if nodes is None:
        layers = model.config.n_layer
        nodes = ["none"]
        for l in range(layers - 1, -1, -1):
            nodes.append(f"f{l}")
            nodes.append(f"a{l}")

    # pos
    if pos is None:
        pos = list(range(len(inputs[0].input_ids[0])))

    # get tokens
    tokens = tokenizer.encode("".join(tokens))
    data = []

    # for each pos and layer in the model, intervene
    for i, node in enumerate(tqdm(nodes)):
        for j in pos:
            if pause:
                print(f"layer={node}, pos={j}")

            # run intervention
            res, cache = model(
                inputs,
                partial(which, node=node, pos=j),
                partial(branch, node=node, pos=j),
                store_cache=True,
                clear_cache=True,
            )

            # store probs
            distrib = embed_to_distrib(model, res.hidden_states, logits=False)
            for token in tokens:
                data.append(
                    {
                        "token": format_token(tokenizer, token),
                        "prob": float(distrib[0][-1][token]),
                        "layer": node,
                        "pos": j,
                        "id": i,
                    }
                )
            
            # visualise
            if pause:
                dot = res.visualise_path()
                dot.format = 'png'
                dot.render('test.gv', view=True)
                input()
                
            # don't repeat nones
            if node == "none":
                break

    # return as dataframe
    df = pd.DataFrame(data)

    # make plot
    if plot:
        df["layer"] = df["layer"].astype("category")
        df["token"] = df["token"].astype("category")
        df["layer"] = df["layer"].cat.set_categories(nodes[::-1], ordered=True)

        g = (
            ggplot(df)
            + geom_tile(aes(x="pos", y="layer", fill="prob", color="prob"))
            + facet_wrap("~token")
            + theme(axis_text_x=element_text(rotation=90))
        )
        return df, nodes, g

    return df, nodes
