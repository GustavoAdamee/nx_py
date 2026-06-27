defmodule NxPy.ConfigTest do
  use ExUnit.Case

  setup do
    # Ensure the agent is running (started by app supervisor or manually)
    unless Process.whereis(NxPy.Config) do
      {:ok, _} = start_supervised(NxPy.Config)
    end

    # Reset state for test isolation
    NxPy.Config.set(%{})
    :ok
  end

  test "get returns empty map before configure" do
    assert NxPy.Config.get() == %{}
  end

  test "set and get round-trip" do
    config = %{backend: :some_backend, python_lib: :some_lib}
    NxPy.Config.set(config)
    assert NxPy.Config.get() == config
  end

  test "set overwrites previous config" do
    NxPy.Config.set(%{backend: :first, python_lib: :numpy})
    NxPy.Config.set(%{backend: :second, python_lib: :numpy})
    assert NxPy.Config.get().backend == :second
  end
end
