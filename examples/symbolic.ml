open Owl
open Owl_symbolic
open Onnxruntime

let model =
  let x = Op.(sin (variable ~shape:[| 3; 3 |] "X")) in
  let g = SymGraph.make_graph [| x |] "sym_graph" in
  let model = ONNX_Engine.of_symbolic g in
  let encoder = Pbrt.Encoder.create () in
  Owl_symbolic_specs.PB.encode_model_proto model encoder;
  Pbrt.Encoder.to_bytes encoder


let forward x =
  let env = create_env LOGGING_LEVEL_ERROR "sine" in
  let sess = create_session_from_bytes env model in
  let meminfo = create_cpu_memory_info ALLOCATOR_TYPE_DEVICE MEMTYPE_DEFAULT in
  let x = of_bigarray meminfo x in
  run sess [ x ] |> List.hd |> to_bigarray Float32


let () =
  let x = Dense.Matrix.S.ones 3 3 in
  let y = forward x in
  Dense.Matrix.S.print y
