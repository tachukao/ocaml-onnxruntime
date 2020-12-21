open Ctypes
open Wrapper

type 'a structured = ('a, [ `Struct ]) Ctypes_static.structured

type logging_level = Typs.LoggingLevel.t =
  | LOGGING_LEVEL_VERBOSE
  | LOGGING_LEVEL_INFO
  | LOGGING_LEVEL_WARNING
  | LOGGING_LEVEL_ERROR
  | LOGGING_LEVEL_FATAL

type allocator_type = Typs.AllocatorType.t =
  | ALLOCATOR_TYPE_INVALID
  | ALLOCATOR_TYPE_DEVICE
  | ALLOCATOR_TYPE_ARENA

type memtype = Typs.MemType.t =
  | MEMTYPE_CPU_INPUT
  | MEMTYPE_CPU_OUTPUT
  | MEMTYPE_CPU
  | MEMTYPE_DEFAULT

type onnx_tensor_element = Typs.TensorElement.t =
  | ONNX_UNDEFINED
  | ONNX_FLOAT
  | ONNX_UINT8
  | ONNX_INT8
  | ONNX_UINT16
  | ONNX_INT16
  | ONNX_INT32
  | ONNX_INT64
  | ONNX_STRING
  | ONNX_BOOL
  | ONNX_FLOAT16
  | ONNX_DOUBLE
  | ONNX_UINT32
  | ONNX_UINT64
  | ONNX_COMPLEX64
  | ONNX_COMPLEX128
  | ONNX_BFLOAT16

let to_onnx_tensor_element : type a b. (a, b) Bigarray.kind -> onnx_tensor_element
  = function
  | Float32        -> ONNX_FLOAT
  | Float64        -> ONNX_DOUBLE
  | Int8_unsigned  -> ONNX_UINT8
  | Int8_signed    -> ONNX_INT8
  | Int16_unsigned -> ONNX_INT16
  | Int16_signed   -> ONNX_INT16
  | Int32          -> ONNX_INT32
  | Int64          -> ONNX_INT64
  | Int            -> ONNX_INT64
  | Complex32      -> ONNX_COMPLEX64
  | Complex64      -> ONNX_COMPLEX128
  | Char           -> ONNX_STRING
  | Nativeint      -> failwith "Nativeint is not support in ONNX Tensor Element Type"


let sizeof_kind : type a b. (a, b) Bigarray.kind -> int = function
  | Float32        -> sizeof float
  | Float64        -> sizeof double
  | Int8_unsigned  -> sizeof uint8_t
  | Int8_signed    -> sizeof int8_t
  | Int16_unsigned -> sizeof uint16_t
  | Int16_signed   -> sizeof int16_t
  | Int32          -> sizeof int32_t
  | Int64          -> sizeof int64_t
  | Int            -> sizeof int
  | Complex32      -> sizeof complex32
  | Complex64      -> sizeof complex64
  | Char           -> sizeof char
  | Nativeint      -> failwith "Nativeint is not support in ONNX Tensor Element Type"


type ortenv = Typs.Env.t structured ptr ptr
type ortsess = Typs.Session.t structured ptr ptr
type ortvalue = Typs.Value.t structured ptr
type ortrunops = Typs.RunOptions.t structured ptr ptr
type ortsessops = Typs.SessionOptions.t structured ptr ptr
type ortmeminfo = Typs.MemoryInfo.t structured ptr ptr
type orttensor_type_and_shape_info = Typs.TensorTypeAndShapeInfo.t structured ptr ptr
type ortallocator = Typs.Allocator.t structured ptr ptr

let ort_api_version = Typs.ort_api_version

(* global API: not exposed to the user *)
let g_ort =
  let api_base = !@(Bindings.get_api_base ()) in
  let f = getf api_base Typs.get_api in
  let g = uint32_t @-> returning (ptr Typs.Api.t) in
  let get_api = coerce (static_funptr g) (Foreign.funptr g) f in
  get_api Unsigned.UInt32.(of_int ort_api_version)


let abort_on_error status =
  if not (is_null status)
  then (
    let g = ptr Typs.Status.t @-> returning string in
    let get =
      getf !@g_ort Typs.Api.get_error_message
      |> coerce (static_funptr g) (Foreign.funptr g)
    in
    let msg = get status in
    failwith msg)


let default_allocator : ortallocator =
  let g = ptr (ptr Typs.Allocator.t) @-> returning (ptr Typs.Status.t) in
  let get =
    getf !@g_ort Typs.Api.get_allocator_with_default_options
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  let alloc = allocate (ptr Typs.Allocator.t) (from_voidp Typs.Allocator.t null) in
  get alloc |> abort_on_error;
  alloc


let create_env =
  let release_env =
    let release_env = getf !@g_ort Typs.Api.release_env in
    let g = ptr Typs.Env.t @-> returning void in
    coerce (static_funptr g) (Foreign.funptr g) release_env
  in
  fun logging_level logid ->
    let env =
      allocate
        ~finalise:(fun env -> release_env !@env)
        (ptr Typs.Env.t)
        (from_voidp Typs.Env.t null)
    in
    let create_env = getf !@g_ort Typs.Api.create_env in
    let g =
      Typs.LoggingLevel.t
      @-> string
      @-> ptr (ptr Typs.Env.t)
      @-> returning (ptr Typs.Status.t)
    in
    let create_env = coerce (static_funptr g) (Foreign.funptr g) create_env in
    (* TODO: implement abort on ERROR *)
    create_env logging_level logid env |> abort_on_error;
    env


let create_session_options : unit -> ortsessops =
  let release_session_options =
    let release_session_options = getf !@g_ort Typs.Api.release_session_options in
    let g = ptr Typs.SessionOptions.t @-> returning void in
    coerce (static_funptr g) (Foreign.funptr g) release_session_options
  in
  fun () ->
    let sess_options =
      allocate
        ~finalise:(fun sess_options -> release_session_options !@sess_options)
        (ptr Typs.SessionOptions.t)
        (from_voidp Typs.SessionOptions.t null)
    in
    let g = ptr (ptr Typs.SessionOptions.t) @-> returning (ptr Typs.Status.t) in
    let create =
      getf !@g_ort Typs.Api.create_session_options
      |> coerce (static_funptr g) (Foreign.funptr g)
    in
    (* TODO: implement abort on ERROR *)
    create sess_options |> abort_on_error;
    sess_options


let release_session =
  let release_session = getf !@g_ort Typs.Api.release_session in
  let g = ptr Typs.Session.t @-> returning void in
  coerce (static_funptr g) (Foreign.funptr g) release_session


let create_session : ?sessops:ortsessops -> ortenv -> string -> ortsess =
 fun ?sessops env model_path ->
  let sess =
    allocate
      ~finalise:(fun sess -> release_session !@sess)
      (ptr Typs.Session.t)
      (from_voidp Typs.Session.t null)
  in
  let g =
    ptr Typs.Env.t
    @-> string
    @-> ptr Typs.SessionOptions.t
    @-> ptr (ptr Typs.Session.t)
    @-> returning (ptr Typs.Status.t)
  in
  let sess_options =
    match sessops with
    | Some x -> x
    | None   -> create_session_options ()
  in
  let create_session =
    getf !@g_ort Typs.Api.create_session |> coerce (static_funptr g) (Foreign.funptr g)
  in
  (* TODO: implement abort on ERROR *)
  create_session !@env model_path !@sess_options sess |> abort_on_error;
  sess


let create_session_from_bytes : ?sessops:ortsessops -> ortenv -> bytes -> ortsess =
 fun ?sessops env model ->
  let sess =
    allocate
      ~finalise:(fun sess -> release_session !@sess)
      (ptr Typs.Session.t)
      (from_voidp Typs.Session.t null)
  in
  let g =
    ptr Typs.Env.t
    @-> ptr void
    @-> size_t
    @-> ptr Typs.SessionOptions.t
    @-> ptr (ptr Typs.Session.t)
    @-> returning (ptr Typs.Status.t)
  in
  let create =
    getf !@g_ort Typs.Api.create_session_from_array
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  let sess_options =
    match sessops with
    | Some x -> x
    | None   -> create_session_options ()
  in
  let size = Bytes.length model |> Unsigned.Size_t.of_int in
  let model = Bytes.to_string model |> CArray.of_string |> CArray.start |> to_voidp in
  create !@env model size !@sess_options sess |> abort_on_error;
  sess


let session_num_inputs : ortsess -> int =
  let g = ptr Typs.Session.t @-> ptr size_t @-> returning (ptr Typs.Status.t) in
  let get =
    getf !@g_ort Typs.Api.session_num_inputs
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun sess ->
    let i = allocate size_t Unsigned.Size_t.(of_int 0) in
    get !@sess i |> abort_on_error;
    Unsigned.Size_t.to_int !@i


let session_num_outputs : ortsess -> int =
  let g = ptr Typs.Session.t @-> ptr size_t @-> returning (ptr Typs.Status.t) in
  let get =
    getf !@g_ort Typs.Api.session_num_outputs
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun sess ->
    let i = allocate size_t Unsigned.Size_t.(of_int 0) in
    get !@sess i |> abort_on_error;
    Unsigned.Size_t.to_int !@i


let session_nth_input_name : ortsess -> int -> ortallocator -> string =
  let g =
    ptr Typs.Session.t
    @-> size_t
    @-> ptr Typs.Allocator.t
    @-> ptr string
    @-> returning (ptr Typs.Status.t)
  in
  let get =
    getf !@g_ort Typs.Api.session_nth_input_name
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun sess i alloc ->
    let name = allocate string "" in
    get !@sess Unsigned.Size_t.(of_int i) !@alloc name |> abort_on_error;
    !@name


let session_nth_output_name : ortsess -> int -> ortallocator -> string =
  let g =
    ptr Typs.Session.t
    @-> size_t
    @-> ptr Typs.Allocator.t
    @-> ptr string
    @-> returning (ptr Typs.Status.t)
  in
  let get =
    getf !@g_ort Typs.Api.session_nth_output_name
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun sess i alloc ->
    let name = allocate string "" in
    get !@sess Unsigned.Size_t.(of_int i) !@alloc name |> abort_on_error;
    !@name


let session_input_names ?(alloc = default_allocator) sess =
  let n_inputs = session_num_inputs sess in
  List.init n_inputs (fun i -> session_nth_input_name sess i alloc)


let session_output_names ?(alloc = default_allocator) sess =
  let n_inputs = session_num_outputs sess in
  List.init n_inputs (fun i -> session_nth_output_name sess i alloc)


let create_cpu_memory_info : allocator_type -> memtype -> ortmeminfo =
  let release =
    let release = getf !@g_ort Typs.Api.release_memory_info in
    let g = ptr Typs.MemoryInfo.t @-> returning void in
    coerce (static_funptr g) (Foreign.funptr g) release
  in
  fun allotype memtype ->
    let memory_info =
      allocate
        ~finalise:(fun x -> release !@x)
        (ptr Typs.MemoryInfo.t)
        (from_voidp Typs.MemoryInfo.t null)
    in
    let g =
      Typs.AllocatorType.t
      @-> Typs.MemType.t
      @-> ptr (ptr Typs.MemoryInfo.t)
      @-> returning (ptr Typs.Status.t)
    in
    let create =
      getf !@g_ort Typs.Api.create_cpu_memory_info
      |> coerce (static_funptr g) (Foreign.funptr g)
    in
    (* TODO: implement abort on ERROR *)
    create allotype memtype memory_info |> abort_on_error;
    memory_info


let release_tensor =
  let release = getf !@g_ort Typs.Api.release_value in
  let g = ptr Typs.Value.t @-> returning void in
  coerce (static_funptr g) (Foreign.funptr g) release


let of_bigarray
    : type a b. ortmeminfo -> (a, b, Bigarray.c_layout) Bigarray.Genarray.t -> ortvalue
  =
 fun meminfo x ->
  let shp = Bigarray.Genarray.dims x in
  let n = Array.fold_left ( * ) 1 shp in
  let n_dims = Array.length shp in
  let shp =
    shp
    |> Array.map Int64.of_int
    |> Array.to_list
    |> CArray.of_list int64_t
    |> CArray.start
  in
  let tensor =
    allocate
      ~finalise:(fun x -> release_tensor !@x)
      (ptr Typs.Value.t)
      (from_voidp Typs.Value.t null)
  in
  let xv = Ctypes.bigarray_start genarray x in
  let g =
    ptr Typs.MemoryInfo.t
    @-> ptr void
    @-> size_t
    @-> ptr int64_t
    @-> size_t
    @-> Typs.TensorElement.t
    @-> ptr (ptr Typs.Value.t)
    @-> returning (ptr Typs.Status.t)
  in
  let create =
    getf !@g_ort Typs.Api.create_tensor_with_data_as_ort_value
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  let kind = Bigarray.Genarray.kind x in
  let tensor_elt = to_onnx_tensor_element kind in
  let status =
    create
      !@meminfo
      (to_voidp xv)
      Unsigned.Size_t.(of_int (n * sizeof_kind kind))
      shp
      Unsigned.Size_t.(of_int n_dims)
      tensor_elt
      tensor
  in
  abort_on_error status;
  !@tensor


let is_tensor : ortvalue -> bool =
  let g = ptr Typs.Value.t @-> ptr int @-> returning (ptr Typs.Status.t) in
  let is_tensor =
    getf !@g_ort Typs.Api.is_tensor |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun x ->
    let i = allocate int 0 in
    is_tensor x i |> abort_on_error;
    !@i = 1


let create_tensor_type_and_shape_info : unit -> orttensor_type_and_shape_info =
  let release =
    let release = getf !@g_ort Typs.Api.release_tensor_type_and_shape_info in
    let g = ptr Typs.TensorTypeAndShapeInfo.t @-> returning void in
    coerce (static_funptr g) (Foreign.funptr g) release
  in
  fun () ->
    let tensor_type_and_shape_info =
      allocate
        ~finalise:(fun x -> release !@x)
        (ptr Typs.TensorTypeAndShapeInfo.t)
        (from_voidp Typs.TensorTypeAndShapeInfo.t null)
    in
    let g = ptr (ptr Typs.TensorTypeAndShapeInfo.t) @-> returning (ptr Typs.Status.t) in
    let create =
      getf !@g_ort Typs.Api.create_tensor_type_and_shape_info
      |> coerce (static_funptr g) (Foreign.funptr g)
    in
    (* TODO: implement abort on ERROR *)
    create tensor_type_and_shape_info |> abort_on_error;
    tensor_type_and_shape_info


let _get_dimensions_count : orttensor_type_and_shape_info -> Unsigned.Size_t.t =
  let g =
    ptr Typs.TensorTypeAndShapeInfo.t @-> ptr size_t @-> returning (ptr Typs.Status.t)
  in
  let get =
    getf !@g_ort Typs.Api.get_dimensions_count
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun info ->
    let size = allocate size_t Unsigned.Size_t.(of_int 0) in
    get !@info size |> abort_on_error;
    !@size


let _get_dimensions : orttensor_type_and_shape_info -> Unsigned.Size_t.t -> int array =
  let g =
    ptr Typs.TensorTypeAndShapeInfo.t
    @-> ptr int64_t
    @-> size_t
    @-> returning (ptr Typs.Status.t)
  in
  let get =
    getf !@g_ort Typs.Api.get_dimensions |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun info ndims ->
    let shp = CArray.make int64_t Unsigned.Size_t.(to_int ndims) in
    get !@info CArray.(start shp) ndims |> abort_on_error;
    shp |> CArray.to_list |> List.map Int64.to_int |> Array.of_list


let _get_tensor_element_type : orttensor_type_and_shape_info -> onnx_tensor_element =
  let g =
    ptr Typs.TensorTypeAndShapeInfo.t
    @-> ptr Typs.TensorElement.t
    @-> returning (ptr Typs.Status.t)
  in
  let get =
    getf !@g_ort Typs.Api.get_tensor_element_type
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun info ->
    let element_type = allocate Typs.TensorElement.t ONNX_UNDEFINED in
    get !@info element_type |> abort_on_error;
    !@element_type


let get_tensor_type_and_shape : ortvalue -> onnx_tensor_element * int array =
  let g =
    ptr Typs.Value.t
    @-> ptr (ptr Typs.TensorTypeAndShapeInfo.t)
    @-> returning (ptr Typs.Status.t)
  in
  let get =
    getf !@g_ort Typs.Api.get_tensor_type_and_shape
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  fun x ->
    let info = create_tensor_type_and_shape_info () in
    get x info |> abort_on_error;
    let ndims = _get_dimensions_count info in
    let dims = _get_dimensions info ndims in
    _get_tensor_element_type info, dims


let to_bigarray
    :  ('a, 'b) Bigarray.kind -> ortvalue
    -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  =
 fun kind tensor ->
  let g = ptr Typs.Value.t @-> ptr (ptr void) @-> returning (ptr Typs.Status.t) in
  let get =
    getf !@g_ort Typs.Api.get_tensor_mutable_data
    |> coerce (static_funptr g) (Foreign.funptr g)
  in
  let _, shp = get_tensor_type_and_shape tensor in
  let x = Bigarray.Genarray.create kind Bigarray.c_layout shp in
  let y = bigarray_start genarray x in
  let z = allocate (ptr void) (to_voidp y) in
  get tensor z |> abort_on_error;
  let z = !@z |> from_voidp (reference_type y) in
  bigarray_of_ptr genarray shp kind z


let create_run_options : unit -> ortrunops =
  let release =
    let release = getf !@g_ort Typs.Api.release_run_options in
    let g = ptr Typs.RunOptions.t @-> returning void in
    coerce (static_funptr g) (Foreign.funptr g) release
  in
  fun () ->
    let run_options =
      allocate
        ~finalise:(fun x -> release !@x)
        (ptr Typs.RunOptions.t)
        (from_voidp Typs.RunOptions.t null)
    in
    let g = ptr (ptr Typs.RunOptions.t) @-> returning (ptr Typs.Status.t) in
    let create =
      getf !@g_ort Typs.Api.create_run_options
      |> coerce (static_funptr g) (Foreign.funptr g)
    in
    create run_options |> abort_on_error;
    run_options


let run =
  let g =
    ptr Typs.Session.t
    @-> ptr Typs.RunOptions.t
    @-> ptr string
    @-> ptr (ptr Typs.Value.t)
    @-> size_t
    @-> ptr string
    @-> size_t
    @-> ptr (ptr Typs.Value.t)
    @-> returning (ptr Typs.Status.t)
  in
  let run = getf !@g_ort Typs.Api.run |> coerce (static_funptr g) (Foreign.funptr g) in
  fun ?runops ?inp_names ?out_names sess inps ->
    let inp_names =
      match inp_names with
      | Some inp_names -> inp_names
      | None           -> session_input_names sess
    in
    let out_names =
      match out_names with
      | Some out_names -> out_names
      | None           -> session_output_names sess
    in
    let runoptions =
      match runops with
      | Some x -> x
      | None   -> create_run_options ()
    in
    let inp_len = List.length inp_names in
    let out_len = List.length out_names in
    let inps = CArray.of_list (ptr Typs.Value.t) inps |> CArray.start in
    let tensor =
      allocate_n ~count:out_len ~finalise:(fun x -> release_tensor !@x) (ptr Typs.Value.t)
    in
    let inp_names = CArray.(start (of_list string inp_names)) in
    let out_names = CArray.(start (of_list string out_names)) in
    run
      !@sess
      !@runoptions
      inp_names
      inps
      (Unsigned.Size_t.of_int inp_len)
      out_names
      (Unsigned.Size_t.of_int out_len)
      tensor
    |> abort_on_error;
    CArray.from_ptr tensor out_len |> CArray.to_list
