defmodule NxPyTest do
  use ExUnit.Case

  describe "build_toml/1" do
    test "generates TOML with local nx_py path dep, uv.sources, and reinstall-package" do
      toml = NxPy.build_toml()

      py_src = Application.app_dir(:nx_py, "priv/python")

      # The bundled nx_py package is the first dependency, as a local editable path
      assert toml =~ ~s("nx_py @ #{py_src}")
      assert toml =~ ~s(name = "nx_py_project")
      assert toml =~ ~s(requires-python = "==3.13.*")

      # tool.uv.sources table maps nx_py to the local path with editable = true
      assert toml =~ "[tool.uv.sources]"
      assert toml =~ "nx_py = { path = \"#{py_src}\", editable = true }"

      # reinstall-package ensures changes to the bundled package are picked up
      assert toml =~ "[tool.uv]"
      assert toml =~ "reinstall-package = [\"nx_py\"]"
    end

    test "appends extra_deps after the local nx_py dep" do
      toml = NxPy.build_toml(extra_deps: ["scipy==1.12.0", "pandas==2.2.0"])

      py_src = Application.app_dir(:nx_py, "priv/python")
      assert toml =~ ~s("nx_py @ #{py_src}")
      assert toml =~ ~s("scipy==1.12.0")
      assert toml =~ ~s("pandas==2.2.0")
    end

    test "overrides python version" do
      toml = NxPy.build_toml(python_version: "==3.12.*")

      assert toml =~ ~s(requires-python = "==3.12.*")
      refute toml =~ ~s(requires-python = "==3.13.*")
    end
  end
end
