defmodule NxPy.Dtype do
  @moduledoc """
  Maps Nx tensor types to and from NumPy dtype strings.

  NumPy is the only, hardcoded target (no behaviour — the previous
  `NxPy.PythonLib` pluggability layer was removed as part of the
  protocol/encoder refactor). Standard types map to native NumPy
  dtypes; `{:bf, 16}`, `{:f, 8}`, `{:f8_e4m3fn, 8}`, and the sub-byte
  integers (`{:s, 4}`, `{:s, 2}`, `{:u, 4}`, `{:u, 2}`) map to
  `ml_dtypes` strings.
  """

  @types [
    {{:f, 64}, "float64"},
    {{:f, 32}, "float32"},
    {{:f, 16}, "float16"},
    {{:bf, 16}, "bfloat16"},
    {{:f, 8}, "float8_e5m2"},
    {{:f8_e4m3fn, 8}, "float8_e4m3fn"},
    {{:s, 64}, "int64"},
    {{:s, 32}, "int32"},
    {{:s, 16}, "int16"},
    {{:s, 8}, "int8"},
    {{:u, 64}, "uint64"},
    {{:u, 32}, "uint32"},
    {{:u, 16}, "uint16"},
    {{:u, 8}, "uint8"},
    {{:c, 64}, "complex64"},
    {{:c, 128}, "complex128"}
  ]

  @subbyte_types [
    {{:s, 4}, "int4"},
    {{:s, 2}, "int2"},
    {{:u, 4}, "uint4"},
    {{:u, 2}, "uint2"}
  ]

  for {nx_type, numpy_type} <- @types ++ @subbyte_types do
    def to_numpy(unquote(nx_type)), do: unquote(numpy_type)
  end

  # Sub-byte types are intentionally absent on the return path: the encoder
  # widens them to int8/uint8 before Python sees them.
  for {nx_type, numpy_type} <- @types do
    def from_numpy(unquote(numpy_type)), do: unquote(nx_type)
  end
end
