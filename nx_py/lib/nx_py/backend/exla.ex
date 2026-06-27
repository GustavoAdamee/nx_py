defmodule NxPy.Backend.EXLA do
  @moduledoc """
  Zero-copy backend for Nx tensors using EXLA's `to_pointer`/`from_pointer` API.

  Shares memory directly between EXLA DeviceBuffers and Python via raw pointers.
  Requires tensors to be on the `:host` client for `:local` mode pointer access.

  ## Sub-byte type handling

  Sub-byte integers (`{:s, 2}`, `{:s, 4}`, `{:u, 2}`, `{:u, 4}`) are widened to
  `{:s, 8}` / `{:u, 8}` before passing to Python, since NumPy/ml_dtypes uses an
  unpacked representation (1 byte per element) while EXLA packs sub-byte values.
  """
  @behaviour NxPy.Backend

  @impl true
  def to_python_ref(tensor) do
    {tensor, original_type} = maybe_widen(tensor)
    ptr = Nx.to_pointer(tensor, mode: :local)

    %{
      pointer: ptr.address,
      data: nil,
      data_size: ptr.data_size,
      dtype: nil,
      shape: Tuple.to_list(Nx.shape(tensor)),
      tensor_ref: tensor,
      mode: :zero_copy,
      original_type: original_type
    }
  end

  @impl true
  def from_python_ref(ref, _type, _shape) do
    # Python wrote directly into the DeviceBuffer — just return the tensor.
    # If sub-byte widening was applied, narrow back to the original type.
    maybe_narrow(ref.tensor_ref, ref[:original_type])
  end

  # Widen sub-byte integers to the nearest NumPy-compatible type.
  # Returns {widened_tensor, original_type} or {tensor, nil} if no widening needed.
  defp maybe_widen(%{type: {:s, n}} = t) when n < 8, do: {Nx.as_type(t, {:s, 8}), t.type}
  defp maybe_widen(%{type: {:u, n}} = t) when n < 8, do: {Nx.as_type(t, {:u, 8}), t.type}
  defp maybe_widen(t), do: {t, nil}

  defp maybe_narrow(tensor, nil), do: tensor
  defp maybe_narrow(tensor, original_type), do: Nx.as_type(tensor, original_type)
end
