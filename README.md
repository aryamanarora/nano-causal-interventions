# nano-path-patching

Extremely simple implementation of [path patching](https://github.com/redwoodresearch/rust_circuit_public), for cases where the functionality of the whole library is not necessary. Written all in PyTorch.

Basic idea: we rewrite the `model.forward()` call to support picking between two inputs based on the path we are on in the computational graph (the core principle underlying the fully-fledged implementation of path patching). For example, you may call `GPT2.forward(inputs=[input1, input2], which=lambda path: 'attn4.5' in path)`, which will patch all inputs downstream of attention head 5 in layer 4.

Currently under implementation.