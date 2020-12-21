open Bigarray
open Wrapper

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

type ortenv
type ortsess
type ortsessops
type ortvalue
type ortrunops
type ortmeminfo
type ortallocator

(** api version extracted from onnxruntime_c_api.h *)
val ort_api_version : int

(** default ortallocator *)
val default_allocator : ortallocator

(** [to_onnx_tensor_element kind] returns the corresponding ONNX tensor element type. *)
val to_onnx_tensor_element : ('a, 'b) Bigarray.kind -> onnx_tensor_element

(** [create_env logging_level name] creates a new environment called [name], which logs according to [logging_level]. *)
val create_env : logging_level -> string -> ortenv

(** [create_session ?sessops env file ] creates a new session in [env] with a model loaded from path [file]. If [sessops] is not provided, then no session options are passed. *)
val create_session : ?sessops:ortsessops -> ortenv -> string -> ortsess

(** [create_session_from_bytes ?sessops env model ] creates a new session in [env] with a ONNX model [model] in bytes array format. If [sessops] is None, then no session options are passed. *)
val create_session_from_bytes : ?sessops:ortsessops -> ortenv -> bytes -> ortsess

(** [session_num_inputs sess] returns the number of inputs session [sess]. *)
val session_num_inputs : ortsess -> int

(** [session_num_outputs sess] returns the number of outputs for session [sess]. *)
val session_num_outputs : ortsess -> int

(** [session_nth_input_name sess i] returns the [i] th input name of [sess]. *)
val session_nth_input_name : ortsess -> int -> ortallocator -> string

(** [session_nth_output_name sess i] returns the [i] th output name of [sess] *)
val session_nth_output_name : ortsess -> int -> ortallocator -> string

(** [session_input_names ?alloc sess ] returns the input names in [sess]. [default_allocator] is used if [alloc] is not provided. *)
val session_input_names : ?alloc:ortallocator -> ortsess -> string list

(** [session_output_names ?alloc sess ] returns the output names of [sess]. [default_allocator] is used if [alloc] is not provided. *)
val session_output_names : ?alloc:ortallocator -> ortsess -> string list

(** [create_session_options ()] returns a null ortsessops . *)
val create_session_options : unit -> ortsessops

(** [create_cpu_memory_info ]*)
val create_cpu_memory_info : allocator_type -> memtype -> ortmeminfo

(** [create_run_options ()] returns a null ortrunops *)
val create_run_options : unit -> ortrunops

(** [of_bigarray meminfo x] converts Genarray [x] into an ortvalue with memory information [meminfo]. *)
val of_bigarray : ortmeminfo -> ('a, 'b, c_layout) Genarray.t -> ortvalue

(** [to_bigarray kind x] converts an ortvalue into a [kind] Genarray of. *)
val to_bigarray : ('a, 'b) kind -> ortvalue -> ('a, 'b, c_layout) Genarray.t

(** [get_tensor_type_and_shape x] returns the ONNX tensor element type and shape of [x]. *)
val get_tensor_type_and_shape : ortvalue -> onnx_tensor_element * int array

(** [is_tensor x] returns true if [x] is a ORT tensor, false otherwise. *)
val is_tensor : ortvalue -> bool

(** [run sess ?runops ?inp_names ?out_names sess inps ] returns the output of executing [sess] with inputs [inps]. If [inp_names] and [out_names] are not provided, they are computed using [session_input_names] and [session_output_names] respectively. One can provide optional run options [runops]. *)
val run
  :  ?runops:ortrunops
  -> ?inp_names:string list
  -> ?out_names:string list
  -> ortsess
  -> ortvalue list
  -> ortvalue list
