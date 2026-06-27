defmodule NxPy.Backend.BinaryTest do
  use ExUnit.Case

  describe "to_python_ref/1" do
    test "returns copy-based reference map for s32 tensor" do
      tensor = Nx.iota({5}, type: {:s, 32})
      ref = NxPy.Backend.Binary.to_python_ref(tensor)

      assert ref.mode == :copy_based
      assert ref.pointer == nil
      assert ref.data != nil
      assert ref.data_size == 5 * 4
      assert ref.shape == [5]
      assert ref.tensor_ref == tensor
      assert ref.dtype == nil
    end

    test "returns correct data_size for f64 tensor" do
      tensor = Nx.iota({3}, type: {:f, 64})
      ref = NxPy.Backend.Binary.to_python_ref(tensor)

      assert ref.data_size == 3 * 8
    end
  end

  describe "from_python_ref/3" do
    test "reconstructs s32 tensor from binary data" do
      tensor = Nx.iota({5}, type: {:s, 32})
      ref = NxPy.Backend.Binary.to_python_ref(tensor)

      result = NxPy.Backend.Binary.from_python_ref(ref, {:s, 32}, {5})

      assert Nx.to_flat_list(result) == Nx.to_flat_list(tensor)
      assert result.shape == {5}
      assert result.type == {:s, 32}
    end

    test "reconstructs f32 tensor with 2D shape" do
      tensor = Nx.iota({2, 3}, type: {:f, 32})
      ref = NxPy.Backend.Binary.to_python_ref(tensor)

      result = NxPy.Backend.Binary.from_python_ref(ref, {:f, 32}, {2, 3})

      assert Nx.to_flat_list(result) == Nx.to_flat_list(tensor)
      assert result.shape == {2, 3}
    end

    test "reconstructs with updated data (simulating Python computation)" do
      tensor = Nx.tensor([1, 2, 3], type: {:s, 32})
      ref = NxPy.Backend.Binary.to_python_ref(tensor)

      # Simulate Python writing new data
      new_data = Nx.tensor([10, 20, 30], type: {:s, 32}) |> Nx.to_binary()
      ref = %{ref | data: new_data}

      result = NxPy.Backend.Binary.from_python_ref(ref, {:s, 32}, {3})

      assert Nx.to_flat_list(result) == [10, 20, 30]
    end
  end
end
