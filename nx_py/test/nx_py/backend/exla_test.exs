defmodule NxPy.Backend.EXLATest do
  use ExUnit.Case, async: false

  @moduletag :exla

  describe "to_python_ref/1" do
    setup do
      Application.put_env(:nx, :default_backend, {EXLA.Backend, client: :host})
      :ok
    end

    test "returns zero-copy reference map for s32 tensor" do
      tensor = Nx.iota({5}, type: {:s, 32})
      ref = NxPy.Backend.EXLA.to_python_ref(tensor)

      assert ref.mode == :zero_copy
      assert is_integer(ref.pointer)
      assert ref.pointer > 0
      assert ref.data == nil
      assert ref.data_size == 5 * 4
      assert ref.shape == [5]
      assert ref.tensor_ref == tensor
      assert ref.dtype == nil
    end

    test "returns valid pointer for f32 tensor" do
      tensor = Nx.iota({3}, type: {:f, 32})
      ref = NxPy.Backend.EXLA.to_python_ref(tensor)

      assert is_integer(ref.pointer)
      assert ref.data_size == 3 * 4
    end
  end

  describe "from_python_ref/3" do
    setup do
      Application.put_env(:nx, :default_backend, {EXLA.Backend, client: :host})
      :ok
    end

    test "returns the original tensor reference (Python wrote in-place)" do
      tensor = Nx.iota({5}, type: {:s, 32})
      ref = NxPy.Backend.EXLA.to_python_ref(tensor)

      result = NxPy.Backend.EXLA.from_python_ref(ref, {:s, 32}, {5})

      # For zero-copy, the result IS the original tensor
      assert Nx.to_flat_list(result) == Nx.to_flat_list(tensor)
    end
  end
end
