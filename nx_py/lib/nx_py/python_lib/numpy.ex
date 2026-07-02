defmodule NxPy.PythonLib.Numpy do
  @moduledoc """
  NumPy Python library adapter for nx_py.

  Maps Nx tensor types to NumPy dtype strings. Standard types map directly.
  Brain float, fp8, and sub-byte integer types require the `ml_dtypes` Python package.
  """
  @behaviour NxPy.PythonLib

  @impl true
  # Standard floats — native NumPy
  def nx_to_dtype(%{type: {:f, 64}}), do: "float64"
  def nx_to_dtype(%{type: {:f, 32}}), do: "float32"
  def nx_to_dtype(%{type: {:f, 16}}), do: "float16"

  # Brain float and fp8 — via ml_dtypes
  def nx_to_dtype(%{type: {:bf, 16}}), do: "bfloat16"
  def nx_to_dtype(%{type: {:f, 8}}), do: "float8_e5m2"
  def nx_to_dtype(%{type: {:f8_e4m3fn, 8}}), do: "float8_e4m3fn"

  # Signed integers — native NumPy
  def nx_to_dtype(%{type: {:s, 64}}), do: "int64"
  def nx_to_dtype(%{type: {:s, 32}}), do: "int32"
  def nx_to_dtype(%{type: {:s, 16}}), do: "int16"
  def nx_to_dtype(%{type: {:s, 8}}), do: "int8"

  # Unsigned integers — native NumPy
  def nx_to_dtype(%{type: {:u, 64}}), do: "uint64"
  def nx_to_dtype(%{type: {:u, 32}}), do: "uint32"
  def nx_to_dtype(%{type: {:u, 16}}), do: "uint16"
  def nx_to_dtype(%{type: {:u, 8}}), do: "uint8"

  # Complex — native NumPy
  def nx_to_dtype(%{type: {:c, 64}}), do: "complex64"
  def nx_to_dtype(%{type: {:c, 128}}), do: "complex128"

  # Sub-byte integers — via ml_dtypes (unpacked: 1 byte per element)
  def nx_to_dtype(%{type: {:s, 4}}), do: "int4"
  def nx_to_dtype(%{type: {:s, 2}}), do: "int2"
  def nx_to_dtype(%{type: {:u, 4}}), do: "uint4"
  def nx_to_dtype(%{type: {:u, 2}}), do: "uint2"

  @ml_dtypes [{:bf, 16}, {:f, 8}, {:f8_e4m3fn, 8}, {:s, 4}, {:s, 2}, {:u, 4}, {:u, 2}]

  @impl true
  def requires_ml_dtypes?(%{type: t}) do
    t in @ml_dtypes
  end
end
