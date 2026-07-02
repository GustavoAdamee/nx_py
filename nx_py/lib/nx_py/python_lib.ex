defmodule NxPy.PythonLib do
  @moduledoc """
  Behaviour defining how a Python library maps Nx tensor types to Python dtype strings.

  Each Python lib must implement:
  - `nx_to_dtype/1` — maps an Nx tensor's type to a Python dtype string
  - `requires_ml_dtypes?/1` — returns true if the tensor type needs ml_dtypes
  """

  @callback nx_to_dtype(tensor :: Nx.Tensor.t()) :: String.t()
  @callback requires_ml_dtypes?(tensor :: Nx.Tensor.t()) :: boolean()
end
