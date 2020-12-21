open Ctypes

module Bindings (S : Cstubs.Types.TYPE) = struct
  open S

  let ort_api_version = constant "ORT_API_VERSION" int

  module TensorElement = struct
    type t =
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

    let onnx_undefined = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED" int64_t
    let onnx_float = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT" int64_t
    let onnx_uint8 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8" int64_t
    let onnx_int8 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8" int64_t
    let onnx_uint16 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16" int64_t
    let onnx_int16 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16" int64_t
    let onnx_int32 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32" int64_t
    let onnx_int64 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64" int64_t
    let onnx_string = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING" int64_t
    let onnx_bool = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL" int64_t
    let onnx_f16 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16" int64_t
    let onnx_f64 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE" int64_t
    let onnx_uint32 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32" int64_t
    let onnx_uint64 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64" int64_t
    let onnx_complex64 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64" int64_t
    let onnx_complex128 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128" int64_t
    let onnx_bf16 = S.constant "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16" int64_t

    let t =
      S.enum
        "ONNXTensorElementDataType"
        ~typedef:true
        [ ONNX_UNDEFINED, onnx_undefined
        ; ONNX_FLOAT, onnx_float
        ; ONNX_UINT8, onnx_uint8
        ; ONNX_INT8, onnx_int8
        ; ONNX_UINT16, onnx_uint16
        ; ONNX_INT16, onnx_int16
        ; ONNX_INT32, onnx_int32
        ; ONNX_INT64, onnx_int64
        ; ONNX_STRING, onnx_string
        ; ONNX_BOOL, onnx_bool
        ; ONNX_FLOAT16, onnx_f16
        ; ONNX_DOUBLE, onnx_f64
        ; ONNX_UINT32, onnx_uint32
        ; ONNX_UINT64, onnx_uint64
        ; ONNX_COMPLEX64, onnx_complex64
        ; ONNX_COMPLEX128, onnx_complex128
        ; ONNX_BFLOAT16, onnx_bf16
        ]
        ~unexpected:(fun _ -> failwith "unexpected ONNX tensor element data type enum")
  end

  module Type = struct
    type t =
      | UNKNOWN
      | TENSOR
      | SEQUENCE
      | MAP
      | OPAQUE
      | SPARSETENSOR

    let onnx_unknown = constant "ONNX_TYPE_UNKNOWN" int64_t
    let onnx_tensor = constant "ONNX_TYPE_TENSOR" int64_t
    let onnx_sequence = constant "ONNX_TYPE_SEQUENCE" int64_t
    let onnx_map = constant "ONNX_TYPE_MAP" int64_t
    let onnx_opaque = constant "ONNX_TYPE_OPAQUE" int64_t
    let onnx_sparse_tensor = constant "ONNX_TYPE_SPARSETENSOR" int64_t

    let t =
      S.enum
        "ONNXType"
        ~typedef:true
        [ UNKNOWN, onnx_unknown
        ; TENSOR, onnx_tensor
        ; SEQUENCE, onnx_sequence
        ; MAP, onnx_map
        ; OPAQUE, onnx_opaque
        ; SPARSETENSOR, onnx_sparse_tensor
        ]
        ~unexpected:(fun _ -> failwith "unexpected ONNX tensor element data type enum")
  end

  module LoggingLevel = struct
    type t =
      | LOGGING_LEVEL_VERBOSE
      | LOGGING_LEVEL_INFO
      | LOGGING_LEVEL_WARNING
      | LOGGING_LEVEL_ERROR
      | LOGGING_LEVEL_FATAL

    let verbose = constant "ORT_LOGGING_LEVEL_VERBOSE" int64_t
    let info = constant "ORT_LOGGING_LEVEL_INFO" int64_t
    let warning = constant "ORT_LOGGING_LEVEL_WARNING" int64_t
    let error = constant "ORT_LOGGING_LEVEL_ERROR" int64_t
    let fatal = constant "ORT_LOGGING_LEVEL_FATAL" int64_t

    let t =
      S.enum
        ~typedef:true
        "OrtLoggingLevel"
        [ LOGGING_LEVEL_VERBOSE, verbose
        ; LOGGING_LEVEL_INFO, info
        ; LOGGING_LEVEL_WARNING, warning
        ; LOGGING_LEVEL_ERROR, error
        ; LOGGING_LEVEL_FATAL, fatal
        ]
        ~unexpected:(fun _ -> failwith "unexpected ort logging level enum")
  end

  module ErrorCode = struct
    type t =
      | OK
      | FAIL
      | INVALID_ARGUMENT
      | NO_SUCH_FILE
      | NO_MODEL
      | ENGINE_ERROR
      | RUNTIME_EXCEPTION
      | INVALID_PROTOBUF
      | MODEL_LOADED
      | NOT_IMPLEMENTED
      | INVALID_GRAPH
      | EP_FAIL

    let ok = S.constant "ORT_OK" int64_t
    let fail = S.constant "ORT_FAIL" int64_t
    let invalid_argument = S.constant "ORT_INVALID_ARGUMENT" int64_t
    let no_such_file = S.constant "ORT_NO_SUCHFILE" int64_t
    let no_model = S.constant "ORT_NO_MODEL" int64_t
    let engine_error = S.constant "ORT_ENGINE_ERROR" int64_t
    let runtime_exception = S.constant "ORT_RUNTIME_EXCEPTION" int64_t
    let invalid_protobuf = S.constant "ORT_INVALID_PROTOBUF" int64_t
    let model_loaded = S.constant "ORT_MODEL_LOADED" int64_t
    let not_implemented = S.constant "ORT_NOT_IMPLEMENTED" int64_t
    let invalid_graph = S.constant "ORT_INVALID_GRAPH" int64_t
    let ep_fail = S.constant "ORT_EP_FAIL" int64_t

    let t =
      S.enum
        "OrtErrorCode"
        ~typedef:true
        [ OK, ok
        ; FAIL, fail
        ; INVALID_ARGUMENT, invalid_argument
        ; NO_SUCH_FILE, no_such_file
        ; NO_MODEL, no_model
        ; ENGINE_ERROR, engine_error
        ; RUNTIME_EXCEPTION, runtime_exception
        ; INVALID_PROTOBUF, invalid_protobuf
        ; MODEL_LOADED, model_loaded
        ; NOT_IMPLEMENTED, not_implemented
        ; INVALID_GRAPH, invalid_graph
        ; EP_FAIL, ep_fail
        ]
        ~unexpected:(fun _ -> failwith "unexpected Ort Error Code enum")
  end

  module Status = struct
    (* OrtStatus *)
    type t

    let t : t structure typ = structure "OrtStatus"
  end

  module Env = struct
    (* OrtEnv *)
    type t

    let t : t structure typ = structure "OrtEnv"
  end

  module MemoryInfo = struct
    (* OrtMemoryInfo*)
    type t

    let t : t structure typ = structure "OrtMemoryInfo"
  end

  module Session = struct
    (* OrtSession *)
    type t

    let t : t structure typ = structure "OrtSession"
  end

  module Value = struct
    (* OrtValue *)
    type t

    let t : t structure typ = structure "OrtValue"
  end

  module RunOptions = struct
    (* OrtRunOptions*)
    type t

    let t : t structure typ = structure "OrtRunOptions"
  end

  module TensorTypeAndShapeInfo = struct
    (* OrtTensorTypeAndShapeInfo *)
    type t

    let t : t structure typ = structure "OrtTensorTypeAndShapeInfo"
  end

  module SessionOptions = struct
    (* OrtSessionOptions *)
    type t

    let t : t structure typ = structure "OrtSessionOptions"
  end

  module Allocator = struct
    (* OrtAllocator *)
    type t

    let t : t structure typ = structure "OrtAllocator"
    let version = field t "version" uint32_t
  end

  module AllocatorType = struct
    (* OrtAllocatorType *)
    type t =
      | ALLOCATOR_TYPE_INVALID
      | ALLOCATOR_TYPE_DEVICE
      | ALLOCATOR_TYPE_ARENA

    let invalid = constant "Invalid" int64_t
    let device_allocator = constant "OrtDeviceAllocator" int64_t
    let arena_allocator = constant "OrtArenaAllocator" int64_t

    let t =
      S.enum
        ~typedef:true
        "OrtAllocatorType"
        [ ALLOCATOR_TYPE_INVALID, invalid
        ; ALLOCATOR_TYPE_DEVICE, device_allocator
        ; ALLOCATOR_TYPE_ARENA, arena_allocator
        ]
        ~unexpected:(fun _ -> failwith "unexpected Ort Allocator type enum")
  end

  module MemType = struct
    (* OrtMemType *)
    type t =
      | MEMTYPE_CPU_INPUT
      | MEMTYPE_CPU_OUTPUT
      | MEMTYPE_CPU
      | MEMTYPE_DEFAULT

    let cpu_input = constant "OrtMemTypeCPUInput" int64_t
    let cpu_output = constant "OrtMemTypeCPUOutput" int64_t
    let cpu = constant "OrtMemTypeCPU" int64_t
    let default = constant "OrtMemTypeDefault" int64_t

    let t =
      S.enum
        ~typedef:true
        "OrtMemType"
        [ MEMTYPE_CPU_INPUT, cpu_input
        ; MEMTYPE_CPU_OUTPUT, cpu_output
        ; MEMTYPE_CPU, cpu
        ; MEMTYPE_DEFAULT, default
        ]
        ~unexpected:(fun _ -> failwith "unexpected Ort Allocator type enum")
  end

  module Api = struct
    (* OrtApi *)
    type t

    let t : t structure typ = structure "OrtApi"

    let create_env =
      field
        t
        "CreateEnv"
        (static_funptr
           (LoggingLevel.t @-> string @-> ptr (ptr Env.t) @-> returning (ptr Status.t)))


    let create_status =
      field
        t
        "CreateStatus"
        (static_funptr (ErrorCode.t @-> string @-> returning (ptr Status.t)))


    let get_error_code =
      field t "GetErrorCode" (static_funptr (ptr Status.t @-> returning ErrorCode.t))


    let get_error_message =
      field t "GetErrorMessage" (static_funptr (ptr Status.t @-> returning string))


    (* Session *)

    let create_session =
      field
        t
        "CreateSession"
        (static_funptr
           (ptr Env.t
           @-> string
           @-> ptr SessionOptions.t
           @-> ptr (ptr Session.t)
           @-> returning (ptr Status.t)))


    let create_session_from_array =
      field
        t
        "CreateSessionFromArray"
        (static_funptr
           (ptr Env.t
           @-> ptr void
           @-> size_t
           @-> ptr SessionOptions.t
           @-> ptr (ptr Session.t)
           @-> returning (ptr Status.t)))


    let session_num_inputs =
      field
        t
        "SessionGetInputCount"
        (static_funptr (ptr Session.t @-> ptr size_t @-> returning (ptr Status.t)))


    let session_num_outputs =
      field
        t
        "SessionGetOutputCount"
        (static_funptr (ptr Session.t @-> ptr size_t @-> returning (ptr Status.t)))


    let session_nth_input_name =
      field
        t
        "SessionGetInputName"
        (static_funptr
           (ptr Session.t
           @-> size_t
           @-> ptr Allocator.t
           @-> ptr string
           @-> returning (ptr Status.t)))


    let session_nth_output_name =
      field
        t
        "SessionGetOutputName"
        (static_funptr
           (ptr Session.t
           @-> size_t
           @-> ptr Allocator.t
           @-> ptr string
           @-> returning (ptr Status.t)))


    let run =
      field
        t
        "Run"
        (static_funptr
           (ptr Session.t
           @-> ptr RunOptions.t
           @-> ptr string
           @-> ptr (ptr Value.t)
           @-> size_t
           @-> ptr string
           @-> size_t
           @-> ptr (ptr Value.t)
           @-> returning (ptr Status.t)))


    (* Create *)

    let create_session_options =
      field
        t
        "CreateSessionOptions"
        (static_funptr (ptr (ptr SessionOptions.t) @-> returning (ptr Status.t)))


    let create_memory_info =
      field
        t
        "CreateMemoryInfo"
        (static_funptr
           (string
           @-> AllocatorType.t
           @-> int
           @-> MemType.t
           @-> ptr (ptr MemoryInfo.t)
           @-> returning (ptr Status.t)))


    let create_cpu_memory_info =
      field
        t
        "CreateCpuMemoryInfo"
        (static_funptr
           (AllocatorType.t
           @-> MemType.t
           @-> ptr (ptr MemoryInfo.t)
           @-> returning (ptr Status.t)))


    let create_run_options =
      field
        t
        "CreateRunOptions"
        (static_funptr (ptr (ptr RunOptions.t) @-> returning (ptr Status.t)))


    (* Tensor *)

    let create_tensor_with_data_as_ort_value =
      field
        t
        "CreateTensorWithDataAsOrtValue"
        (static_funptr
           (ptr MemoryInfo.t
           @-> ptr void
           @-> size_t
           @-> ptr int64_t
           @-> size_t
           @-> TensorElement.t
           @-> ptr (ptr Value.t)
           @-> returning (ptr Status.t)))


    let is_tensor =
      field
        t
        "IsTensor"
        (static_funptr (ptr Value.t @-> ptr int @-> returning (ptr Status.t)))


    let create_tensor_type_and_shape_info =
      field
        t
        "CreateTensorTypeAndShapeInfo"
        (static_funptr (ptr (ptr TensorTypeAndShapeInfo.t) @-> returning (ptr Status.t)))


    let get_tensor_mutable_data =
      field
        t
        "GetTensorMutableData"
        (static_funptr (ptr Value.t @-> ptr (ptr void) @-> returning (ptr Status.t)))


    let get_tensor_type_and_shape =
      field
        t
        "GetTensorTypeAndShape"
        (static_funptr
           (ptr Value.t
           @-> ptr (ptr TensorTypeAndShapeInfo.t)
           @-> returning (ptr Status.t)))


    let get_tensor_element_type =
      field
        t
        "GetTensorElementType"
        (static_funptr
           (ptr TensorTypeAndShapeInfo.t
           @-> ptr TensorElement.t
           @-> returning (ptr Status.t)))


    let get_dimensions_count =
      field
        t
        "GetDimensionsCount"
        (static_funptr
           (ptr TensorTypeAndShapeInfo.t @-> ptr size_t @-> returning (ptr Status.t)))


    let get_dimensions =
      field
        t
        "GetDimensions"
        (static_funptr
           (ptr TensorTypeAndShapeInfo.t
           @-> ptr int64_t
           @-> size_t
           @-> returning (ptr Status.t)))


    let get_allocator_with_default_options =
      field
        t
        "GetAllocatorWithDefaultOptions"
        (static_funptr (ptr (ptr Allocator.t) @-> returning (ptr Status.t)))


    (* RELEASE *)

    let release_env = field t "ReleaseEnv" (static_funptr (ptr Env.t @-> returning void))

    let release_memory_info =
      field t "ReleaseMemoryInfo" (static_funptr (ptr MemoryInfo.t @-> returning void))


    let release_status =
      field t "ReleaseStatus" (static_funptr (ptr Status.t @-> returning void))


    let release_session =
      field t "ReleaseSession" (static_funptr (ptr Session.t @-> returning void))


    let release_value =
      field t "ReleaseValue" (static_funptr (ptr Value.t @-> returning void))


    let release_run_options =
      field t "ReleaseRunOptions" (static_funptr (ptr RunOptions.t @-> returning void))


    let release_tensor_type_and_shape_info =
      field
        t
        "ReleaseTensorTypeAndShapeInfo"
        (static_funptr (ptr TensorTypeAndShapeInfo.t @-> returning void))


    let release_session_options =
      field
        t
        "ReleaseSessionOptions"
        (static_funptr (ptr SessionOptions.t @-> returning void))


    let () = seal t
  end

  type ortapibase

  let ortapibase : ortapibase structure typ = structure "OrtApiBase"

  let get_api =
    field ortapibase "GetApi" (static_funptr (uint32_t @-> returning (ptr Api.t)))


  let get_version_string =
    field ortapibase "GetVersionString" (static_funptr (void @-> returning string))


  let () = seal ortapibase
end
