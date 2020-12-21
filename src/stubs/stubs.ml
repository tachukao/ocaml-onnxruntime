open Ctypes
module Typs = Typs

module Bindings (F : FOREIGN) = struct
  open F

  let get_api_base = foreign "OrtGetApiBase" (void @-> returning (ptr Typs.ortapibase))
end
