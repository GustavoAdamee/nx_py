defmodule NxPy.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :nx_py,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {NxPy.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  defp deps do
    [
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true, ref: "4c741d35"},
      {:exla, github: "elixir-nx/nx", sparse: "exla", ref: "4c741d35"},
      {:pythonx, "~> 0.4.7"}
    ]
  end
end
