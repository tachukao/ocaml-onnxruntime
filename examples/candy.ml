open Owl
open Onnxruntime
module N = Dense.Ndarray.S

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir

let forward x =
  let env = create_env LOGGING_LEVEL_ERROR "candy" in
  let sess = create_session env (in_dir "candy.onnx") in
  let meminfo = create_cpu_memory_info ALLOCATOR_TYPE_DEVICE MEMTYPE_DEFAULT in
  let x = of_bigarray meminfo x in
  run sess [ x ] |> List.hd |> to_bigarray Float32


let img_to_arr x =
  let x = Array.init 224 (fun i -> Array.init 224 (fun j -> Rgb24.get x i j)) in
  let r = Array.map Array.(map (fun x -> Color.(x.r) |> Int.to_float)) x |> N.of_arrays in
  let g = Array.map Array.(map (fun x -> Color.(x.g) |> Int.to_float)) x |> N.of_arrays in
  let b = Array.map Array.(map (fun x -> Color.(x.b) |> Int.to_float)) x |> N.of_arrays in
  N.stack ~axis:0 [| r; g; b |] |> fun x -> N.(expand x 4)


let arr_to_img x =
  let x = N.map (fun x -> if x >= 255. then 255. else if x < 0. then 0. else x) x in
  let img = Rgb24.create 224 224 in
  for i = 0 to 224 - 1 do
    for j = 0 to 224 - 1 do
      let r = N.get x [| 0; 0; i; j |] |> Float.to_int in
      let g = N.get x [| 0; 1; i; j |] |> Float.to_int in
      let b = N.get x [| 0; 2; i; j |] |> Float.to_int in
      Rgb24.set img i j Color.{ r; g; b }
    done
  done;
  img


let () =
  let img =
    match Jpeg.load (in_dir "amber.jpg") [] with
    | Rgb24 x -> Rgb24.resize None x 224 224
    | _       -> failwith "Jpeg files are Rgb24"
  in
  Jpeg.save (in_dir "amber_before.jpg") [] (Rgb24 img);
  let x = img_to_arr img in
  let y = forward x in
  let img' = arr_to_img y in
  Jpeg.save (in_dir "amber_after.jpg") [] (Rgb24 img')
