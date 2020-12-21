module C = Configurator.V1

let () =
  let onnx_runtime_dir =
    match Sys.getenv_opt "ONNX_RUNTIME_DIR" with
    | Some x -> x
    | None   -> failwith "ONNX_RUNTIME_DIR not found - please see README.md."
  in
  C.main ~name:"onnxruntime" (fun _ ->
      let libs = [ Printf.(sprintf "-L/%s/lib/" onnx_runtime_dir); "-lonnxruntime" ] in
      let cflags = [ Printf.(sprintf "-I/%s/include/" onnx_runtime_dir) ] in
      let conf : C.Pkg_config.package_conf = { cflags; libs } in
      C.Flags.write_sexp "c_flags.sexp" conf.cflags;
      C.Flags.write_sexp "c_library_flags.sexp" conf.libs;
      C.Flags.write_lines "c_flags" conf.cflags;
      C.Flags.write_lines "c_library_flags" conf.libs)
