defmodule NxPy.EncoderTest do
  use ExUnit.Case, async: false

  describe "encode_binary/1 (no Pythonx — structural checks only)" do
    # encode_binary/1 calls Pythonx.eval internally to build the NxTensor
    # wrapper, so it requires an initialized Python env. These tests assert
    # the *inputs* to that eval are correct by inspecting what the function
    # would pass, without actually starting Python. The real round-trip
    # through Pythonx lives in test/nx_py/integration_test.exs.

    test "to_binary/1 + reshape shape are correct for a 2D s32 tensor" do
      tensor = Nx.iota({2, 3}, type: {:s, 32})
      binary = Nx.to_binary(tensor)
      shape_tuple = Nx.shape(tensor)

      assert byte_size(binary) == 2 * 3 * 4
      assert shape_tuple == {2, 3}
      # The encoder will pass shape as a Python tuple literal built from this,
      # and call .reshape(shape) before wrapping in NxTensor — see encoder.ex.
    end

    test "to_binary/1 byte size matches data_size for f64 1D" do
      tensor = Nx.iota({5}, type: {:f, 64})
      binary = Nx.to_binary(tensor)
      assert byte_size(binary) == 5 * 8
    end
  end

  describe "encode_pointer/1 (no Pythonx — structural checks only)" do
    setup do
      Application.put_env(:nx, :default_backend, {EXLA.Backend, client: :host})
      :ok
    end

    test "to_pointer(:local) returns a valid %Nx.Pointer{} for s32" do
      tensor = Nx.iota({5}, type: {:s, 32})
      ptr = Nx.to_pointer(tensor, mode: :local)

      assert %Nx.Pointer{} = ptr
      assert ptr.kind == :local
      assert is_integer(ptr.address) and ptr.address > 0
      assert ptr.data_size == 5 * 4
    end

    test "maybe_widen widens sub-byte s4 to s8 before to_pointer" do
      # Sub-byte types {:s,4}/{:s,2}/{:u,4}/{:u,2} are widened to 8-bit first,
      # because EXLA packs sub-byte values but NumPy/ml_dtypes stores them
      # unpacked (1 byte per element). The encoder's maybe_widen/1 must
      # produce an {:s,8} tensor whose pointer data_size is element_count * 1.
      tensor = Nx.iota({8}, type: {:s, 4}) |> Nx.backend_transfer(EXLA.Backend)
      widened = Nx.as_type(tensor, {:s, 8})
      ptr = Nx.to_pointer(widened, mode: :local)

      assert widened.type == {:s, 8}
      assert ptr.data_size == 8 * 1
    end
  end
end
