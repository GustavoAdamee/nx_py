defmodule NxPy.Backend.Binary do
  @moduledoc """
  Copy-based backend for Nx tensors using the Nx BinaryBackend.

  Converts tensors to/from Erlang binaries. No zero-copy — data is copied
  at the Elixir/Python boundary.
  """
  @behaviour NxPy.Backend

  @impl true
  def to_python_ref(tensor) do
    binary = Nx.to_binary(tensor)

    %{
      pointer: nil,
      data: binary,
      data_size: byte_size(binary),
      dtype: nil,
      shape: Tuple.to_list(Nx.shape(tensor)),
      tensor_ref: tensor,
      mode: :copy_based
    }
  end

  @impl true
  def from_python_ref(ref, type, shape) do
    ref.data
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
  end
end
