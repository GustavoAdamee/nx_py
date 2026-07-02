defmodule NxPy.Config do
  @moduledoc """
  Agent that stores the runtime configuration for nx_py.

  Holds the selected backend module and Python lib module.
  """
  use Agent

  def start_link(_opts) do
    Agent.start_link(fn -> %{} end, name: __MODULE__)
  end

  @doc "Returns the current config map."
  def get do
    Agent.get(__MODULE__, & &1)
  end

  @doc "Sets the config map."
  def set(config) do
    Agent.update(__MODULE__, fn _ -> config end)
  end
end
