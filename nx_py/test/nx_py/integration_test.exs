defmodule NxPy.IntegrationTest do
  use ExUnit.Case, async: false

  # Single network-touching init for the whole file. Installs numpy, ml-dtypes,
  # and the bundled nx_py package (NxTensor class) into the Pythonx env.
  setup_all do
    Application.put_env(:nx, :default_backend, {EXLA.Backend, client: :host})
    NxPy.init_python()
    :ok
  end

  describe "Pythonx.Encoder for Nx.Tensor — Binary backend (copy-based)" do
    test "tensor passed as a global round-trips through numpy" do
      tensor = Nx.iota({5}, type: {:s, 32})

      {result, _} =
        Pythonx.eval(
          """
          arr = tensor.array        # NxTensor -> numpy array (copy via frombuffer().copy())
          out = arr * 10
          from nx_py import NxTensor
          NxTensor(out)
          """,
          %{"tensor" => tensor}
        )

      reconstructed = NxPy.from_python(result, backend: Nx.BinaryBackend)

      assert Nx.to_flat_list(reconstructed) == [0, 10, 20, 30, 40]
      assert reconstructed.type == {:s, 32}
      assert reconstructed.shape == {5}
    end

    test "2D f32 tensor round-trips with shape preserved" do
      tensor = Nx.iota({2, 3}, type: {:f, 32})

      {result, _} =
        Pythonx.eval(
          """
          arr = tensor.array
          out = arr * 2.0
          from nx_py import NxTensor
          NxTensor(out)
          """,
          %{"tensor" => tensor}
        )

      reconstructed = NxPy.from_python(result, backend: Nx.BinaryBackend)

      expected = Nx.multiply(Nx.iota({2, 3}, type: {:f, 32}), 2.0)
      assert Nx.to_flat_list(reconstructed) == Nx.to_flat_list(expected)
      assert reconstructed.shape == {2, 3}
    end
  end

  describe "Pythonx.Encoder for Nx.Tensor — EXLA backend (zero-copy, in-place)" do
    test "from_python_owned/3 retains the zero-copy buffer owner across garbage collection" do
      original = Nx.iota({5}, type: {:s, 32}) |> Nx.backend_transfer(EXLA.Backend)

      {result, _} = Pythonx.eval("tensor", %{"tensor" => original})
      owned = NxPy.from_python_owned(result, original, backend: EXLA.Backend)

      :erlang.garbage_collect()

      assert Nx.to_flat_list(elem(owned, 0)) == [0, 1, 2, 3, 4]
      assert elem(owned, 1) == original
    end

    test "in-place mutation of tensor.array writes into the DeviceBuffer" do
      tensor = Nx.iota({5}, type: {:s, 32}) |> Nx.backend_transfer(EXLA.Backend)

      {result, _} =
        Pythonx.eval(
          """
          arr = tensor.array       # zero-copy numpy view over the DeviceBuffer
          arr *= 10                # in-place — writes straight into the DeviceBuffer
          tensor
          """,
          %{"tensor" => tensor}
        )

      owned = NxPy.from_python_owned(result, tensor, backend: EXLA.Backend)
      reconstructed = elem(owned, 0)

      assert Nx.to_flat_list(reconstructed) == [0, 10, 20, 30, 40]

      assert elem(owned, 1) == tensor
    end

    test "separate output tensor preserves the input" do
      input_tensor = Nx.iota({1000}, type: {:f, 32}) |> Nx.backend_transfer(EXLA.Backend)

      output_tensor =
        Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {1000}) |> Nx.backend_transfer(EXLA.Backend)

      {result, _} =
        Pythonx.eval(
          """
          in_arr = input.array
          out_arr = output.array
          out_arr[:] = in_arr * 10
          output
          """,
          %{"input" => input_tensor, "output" => output_tensor}
        )

      owned = NxPy.from_python_owned(result, output_tensor, backend: EXLA.Backend)

      reconstructed = elem(owned, 0)
      expected = Nx.multiply(Nx.iota({1000}, type: {:f, 32}), 10.0)

      assert Nx.to_flat_list(reconstructed |> Nx.slice_along_axis(0, 5)) == [
               0.0,
               10.0,
               20.0,
               30.0,
               40.0
             ]

      assert Nx.to_flat_list(reconstructed) == Nx.to_flat_list(expected)

      # Input was NOT mutated (separate output tensor)
      assert Nx.to_flat_list(input_tensor |> Nx.slice_along_axis(0, 5)) == [
               0.0,
               1.0,
               2.0,
               3.0,
               4.0
             ]

      assert elem(owned, 1) == output_tensor
    end
  end

  describe "from_python/2 reads metadata off any NxTensor instance" do
    test "obj need not be the original — a freshly-built NxTensor works" do
      # Build an NxTensor entirely in Python, never having sent an Elixir tensor.
      {result, _} =
        Pythonx.eval(
          """
          import numpy as np
          from nx_py import NxTensor
          NxTensor(np.array([1, 2, 3, 4], dtype=np.int32))
          """,
          %{}
        )

      reconstructed = NxPy.from_python(result, backend: Nx.BinaryBackend)

      assert Nx.to_flat_list(reconstructed) == [1, 2, 3, 4]
      assert reconstructed.type == {:s, 32}
      assert reconstructed.shape == {4}
    end
  end
end
