defmodule NxPy.Encoder do
  @moduledoc """
  Per-backend conversion of `Nx.Tensor` into a `Pythonx.Object` that wraps
  a `nx_py.NxTensor` Python instance.

  The `defimpl Pythonx.Encoder, for: Nx.Tensor` (in this same file) dispatches
  to `encode_exla/1` or `encode_binary/1` based on the concrete backend struct
  found at `tensor.data`. Other backends raise a clear `Protocol.UndefinedError`.
  """

  alias NxPy.Dtype

  @doc """
  Zero-copy encode for `EXLA.Backend`.

  Sub-byte integers (`{:s,4}`, `{:s,2}`, `{:u,4}`, `{:u,2}`) are widened to
  8-bit first (EXLA packs sub-byte values; NumPy/ml_dtypes stores them unpacked
  as 1 byte per element). `Nx.to_pointer(tensor, mode: :local)` then yields a
  raw address; a `Pythonx.eval` builds a NumPy view over that address via
  `ctypes`/`frombuffer` and wraps it in `nx_py.NxTensor`. Mutations in Python
  are visible on the DeviceBuffer — zero copies.
  """
  def encode_exla(tensor) do
    {tensor, _original_type} = maybe_widen(tensor)
    ptr = Nx.to_pointer(tensor, mode: :local)
    address = ptr.address
    data_size = ptr.data_size
    shape_tuple = Nx.shape(tensor)
    dtype_str = Dtype.to_numpy(tensor.type)

    {result, %{}} =
      Pythonx.eval(
        """
        import ctypes
        import ml_dtypes
        import numpy as np
        from nx_py import NxTensor
        buf = (ctypes.c_uint8 * data_size).from_address(address)
        arr = np.frombuffer(buf, dtype=np.dtype(dtype_str)).reshape(shape)
        NxTensor(arr)
        """,
        %{
          "address" => address,
          "data_size" => data_size,
          "shape" => shape_tuple,
          "dtype_str" => dtype_str
        }
      )

    result
  end

  @doc """
  Copy-based encode for `Nx.BinaryBackend`.

  Sub-byte integers (`{:s,4}`, `{:s,2}`, `{:u,4}`, `{:u,2}`) are widened to
  8-bit first, same as `encode_exla/1` — `Nx.to_binary/1` packs sub-byte
  values 2-per-byte, but NumPy/ml_dtypes' unpacked `int4`/`int2` dtypes
  expect 1 byte per element, so an un-widened sub-byte tensor would hand
  `frombuffer` half the bytes it needs to fill `shape`.

  `Nx.to_binary/1` produces bytes; `Pythonx.eval` builds a NumPy array via
  `frombuffer(...).copy()` (the `.copy()` breaks the alias with the BEAM-owned
  binary so Python owns its own buffer), reshapes it to the tensor's shape,
  and wraps it in `nx_py.NxTensor`. The reshape matters: `from_python/2`
  reads shape back off `obj.shape`, so a flat `frombuffer` result would
  round-trip a `{2, 3}` tensor back as `{6}`.
  """
  def encode_binary(tensor) do
    {tensor, _original_type} = maybe_widen(tensor)
    binary = Nx.to_binary(tensor)
    shape_tuple = Nx.shape(tensor)
    dtype_str = Dtype.to_numpy(tensor.type)

    {result, %{}} =
      Pythonx.eval(
        """
        import ml_dtypes
        import numpy as np
        from nx_py import NxTensor
        arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).copy().reshape(shape)
        NxTensor(arr)
        """,
        %{
          "data" => binary,
          "shape" => shape_tuple,
          "dtype_str" => dtype_str
        }
      )

    result
  end

  # Widen sub-byte integers to the nearest NumPy-compatible (8-bit) type.
  # Returns {widened_tensor, original_type} or {tensor, nil} if no widening needed.
  # NOTE: original_type is currently unused on the encode path (the caller cannot
  # carry it through to from_python/2; sub-byte narrowing is a documented non-goal
  # of this refactor). It is retained here for clarity and a possible future
  # revision that threads a retained handle through.
  defp maybe_widen(%Nx.Tensor{type: {:s, n}} = t) when n < 8, do: {Nx.as_type(t, {:s, 8}), t.type}
  defp maybe_widen(%Nx.Tensor{type: {:u, n}} = t) when n < 8, do: {Nx.as_type(t, {:u, 8}), t.type}
  defp maybe_widen(t), do: {t, nil}
end

defimpl Pythonx.Encoder, for: Nx.Tensor do
  @moduledoc """
  Makes any `Nx.Tensor` usable directly as a `Pythonx.eval` global.

  Dispatches on the concrete backend struct at `tensor.data`:

    * `%EXLA.Backend{}`     -> `NxPy.Encoder.encode_exla/1` (zero-copy)
    * `%Nx.BinaryBackend{}` -> `NxPy.Encoder.encode_binary/1` (copy-based)

  Other backends raise a `Protocol.UndefinedError` naming the offending struct.

  Requires an initialized Python env (`NxPy.init_python/1`), because the encoder
  imports `nx_py.NxTensor` from the bundled package. A missing package surfaces
  as a clear `Protocol.UndefinedError` naming the fix.
  """

  def encode(tensor, _encoder) do
    case tensor.data do
      %EXLA.Backend{} ->
        NxPy.Encoder.encode_exla(tensor)

      %Nx.BinaryBackend{} ->
        NxPy.Encoder.encode_binary(tensor)

      other ->
        raise Protocol.UndefinedError,
          protocol: @protocol,
          value: tensor,
          description:
            "nx_py does not support the #{inspect(other.__struct__)} backend yet. " <>
              "If you intended to use a supported backend, ensure the tensor is on " <>
              "EXLA.Backend or Nx.BinaryBackend. If Python is not initialized, call " <>
              "NxPy.init_python/0 first."
    end
  rescue
    e in Pythonx.Error ->
      # A Python-side error inside the encoder's internal eval (most commonly
      # ModuleNotFoundError when nx_py.NxTensor isn't installed because
      # NxPy.init_python/0 hasn't run) becomes a clear protocol error.
      if String.contains?(Exception.message(e), "ModuleNotFoundError") do
        raise Protocol.UndefinedError,
          protocol: @protocol,
          value: tensor,
          description:
            "cannot encode Nx.Tensor, because the nx_py Python package is not " <>
              "installed — call NxPy.init_python/0 first"
      else
        reraise e, __STACKTRACE__
      end
  end
end
