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

  # Standard floats — native NumPy
  def to_numpy({:f, 64}), do: "float64"
  def to_numpy({:f, 32}), do: "float32"
  def to_numpy({:f, 16}), do: "float16"

  # Brain float and fp8 — via ml_dtypes
  def to_numpy({:bf, 16}), do: "bfloat16"
  def to_numpy({:f, 8}), do: "float8_e5m2"
  def to_numpy({:f8_e4m3fn, 8}), do: "float8_e4m3fn"

  # Signed integers — native NumPy
  def to_numpy({:s, 64}), do: "int64"
  def to_numpy({:s, 32}), do: "int32"
  def to_numpy({:s, 16}), do: "int16"
  def to_numpy({:s, 8}), do: "int8"

  # Unsigned integers — native NumPy
  def to_numpy({:u, 64}), do: "uint64"
  def to_numpy({:u, 32}), do: "uint32"
  def to_numpy({:u, 16}), do: "uint16"
  def to_numpy({:u, 8}), do: "uint8"

  # Complex — native NumPy
  def to_numpy({:c, 64}), do: "complex64"
  def to_numpy({:c, 128}), do: "complex128"

  # Sub-byte integers — via ml_dtypes (unpacked: 1 byte per element)
  def to_numpy({:s, 4}), do: "int4"
  def to_numpy({:s, 2}), do: "int2"
  def to_numpy({:u, 4}), do: "uint4"
  def to_numpy({:u, 2}), do: "uint2"

  # Inverse of to_numpy/1. Sub-byte types are intentionally absent on the
  # return path (they arrive widened as int8/uint8 — see module docs).
  def from_numpy("float64"), do: {:f, 64}
  def from_numpy("float32"), do: {:f, 32}
  def from_numpy("float16"), do: {:f, 16}
  def from_numpy("bfloat16"), do: {:bf, 16}
  def from_numpy("float8_e5m2"), do: {:f, 8}
  def from_numpy("float8_e4m3fn"), do: {:f8_e4m3fn, 8}
  def from_numpy("int64"), do: {:s, 64}
  def from_numpy("int32"), do: {:s, 32}
  def from_numpy("int16"), do: {:s, 16}
  def from_numpy("int8"), do: {:s, 8}
  def from_numpy("uint64"), do: {:u, 64}
  def from_numpy("uint32"), do: {:u, 32}
  def from_numpy("uint16"), do: {:u, 16}
  def from_numpy("uint8"), do: {:u, 8}
  def from_numpy("complex64"), do: {:c, 64}
  def from_numpy("complex128"), do: {:c, 128}
end
