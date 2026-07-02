defmodule NxPy.Backend do
  @moduledoc """
  Behaviour defining how an Nx backend converts tensors for Python interop.

  Each backend must implement:
  - `to_python_ref/1` — converts an Nx tensor into a reference map for Python
  - `from_python_ref/3` — reconstructs an Nx tensor from a reference map
  """

  @type python_ref :: %{
          pointer: non_neg_integer() | nil,
          data: binary() | nil,
          data_size: non_neg_integer(),
          dtype: String.t() | nil,
          shape: [pos_integer()],
          tensor_ref: Nx.Tensor.t(),
          mode: :zero_copy | :copy_based
        }

  @callback to_python_ref(tensor :: Nx.Tensor.t()) :: python_ref()
  @callback from_python_ref(
              ref :: python_ref(),
              type :: Nx.Tensor.type(),
              shape :: Nx.Tensor.shape()
            ) ::
              Nx.Tensor.t()
end
