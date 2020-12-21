# OCaml Bindings to ONNX Runtime

## Getting started
1. Download ONNX Runtime from [here](https://github.com/microsoft/onnxruntime/releases).
2. Add the following lines to your `.bash_profile` or `.zshrc`:
```sh
export ONNX_RUNTIME_DIR="path/to/onnxruntime-xxx-xxx-1.x.x"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:path/to/onnxruntime-xxx-xxx-1.x.x"
```
3. Install these bindings with:
```sh
cd ocaml-onnxruntime
dune build @install
dune install
```
