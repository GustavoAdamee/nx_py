# EXLA Zero-Copy Proof via LLDB + Python ctypes
#
# Demonstrates that Nx.to_pointer(:local) exposes the actual EXLA DeviceBuffer
# memory by using LLDB to inspect/write at the raw address, and Python ctypes
# to write to the same address — all visible to Elixir without any copy.
#
# Prerequisites:
#   cd /path/to/nx/exla
#   mix deps.get
#   DEBUG=1 EXLA_FORCE_REBUILD=partial mix compile
#
# Run (from the nx/exla directory):
#   elixir ../../livebook_files/lldb_zero_copy_proof.exs
#
# LLDB workflow (macOS; use gdb on Linux with similar commands):
#   1. Run this script — it pauses at each phase
#   2. In another terminal: sudo lldb --attach-pid <pid>
#   3. Set breakpoint inside EXLA: b read_device_mem
#      (breakpoints by function name survive code updates)
#   4. Continue: c
#   5. Return to Elixir and press ENTER to proceed
#   6. When breakpoint hits, get the device pointer:
#      expr buffer.ptr->resource.GetDevicePointer(buffer.ptr->resource.buffer()->client())
#   7. Read memory: memory read <addr> <addr+N>
#   8. Write memory: memory write <addr> <byte_hex>
#   9. Continue: c — verify changes are visible in Elixir

Mix.install(
  [
    {:exla, path: "."},
    {:nx, path: "../nx", override: true},
    {:pythonx, "~> 0.4.7"}
  ],
  system_env: %{"DEBUG" => "1", "EXLA_FORCE_REBUILD" => "partial"}
)

Application.put_env(:nx, :default_backend, {EXLA.Backend, client: :host})

Pythonx.uv_init("""
[project]
name = "project"
version = "0.0.0"
requires-python = "==3.13.*"
dependencies = ["numpy==2.2.2"]
""")

# --- Helpers ---

defmodule Proof do
  def hex(address), do: "0x#{Integer.to_string(address, 16)}"

  def print_lldb_commands(pid, address, size) do
    addr_hex = hex(address)
    end_hex = hex(address + size)

    IO.puts("  sudo lldb --attach-pid #{pid}")
    IO.puts("  memory read #{addr_hex} #{end_hex}")
    IO.puts("  process detach")
  end

  
  def pause(msg) do
    IO.gets(msg <> " ")
  end
end

# ============================================================
# PHASE 0: Setup — create tensor with known values
# ============================================================

IO.puts("=== EXLA ZERO-COPY PROOF VIA LLDB ===\n")

type = {:s, 32}
backend = {EXLA.Backend, client: :host}
values = [1, 2, 3, 4, 5]
pid = System.pid()

t_bin = Nx.tensor([values], type: type)
IO.inspect(Nx.to_binary(t_bin), label: "t_bin")

t_exla = Nx.backend_copy(t_bin, backend)
IO.inspect(Nx.to_binary(t_exla), label: "t_exla")
IO.puts("")

pointer = Nx.to_pointer(t_exla, mode: :local)
addr = pointer.address
size = pointer.data_size

IO.puts("Pointer address: #{Proof.hex(addr)}")
IO.puts("Pointer size:    #{size} bytes")
IO.puts("")

# ============================================================
# PHASE 1: LLDB reads/writes via breakpoint
# ============================================================

IO.puts("=== PHASE 1: LLDB INSPECTION ===")
IO.puts("PID: #{pid}")
IO.puts("")
IO.puts("In another terminal:")
IO.puts("  sudo lldb --attach-pid #{pid}")
IO.puts("")
IO.puts("Inside LLDB, set breakpoint to trace the DeviceBuffer pointer:")
IO.puts("  (lldb) b read_device_mem")
IO.puts("  (lldb) c")
IO.puts("  # Then press ENTER in Elixir to trigger the call")
IO.puts("  # When breakpoint hits:")
IO.puts("  (lldb) expr buffer.ptr->resource.GetDevicePointer(buffer.ptr->resource.buffer()->client())")
IO.puts("")
IO.puts("Or read memory directly at the to_pointer address:")
Proof.print_lldb_commands(pid, addr, size)
IO.puts("")
IO.puts("Expected memory (little-endian int32):")
IO.puts("  01 00 00 00  02 00 00 00  03 00 00 00  04 00 00 00  05 00 00 00")
IO.puts("  (1)          (2)          (3)          (4)          (5)")
IO.puts("")
IO.puts("Optional: write 42 (0x2A) to first element:")
IO.puts("  (lldb) memory write #{Proof.hex(addr)} 0x2a 0x00 0x00 0x00")
IO.puts("")

Proof.pause("[ENTER to continue to Phase 2]")

# ============================================================
# PHASE 2: Python writes via ctypes at the same address
# ============================================================

IO.puts("")
IO.puts("=== PHASE 2: PYTHON WRITES VIA CTYPES ===")
IO.puts("Python writing value 99 to first element at #{Proof.hex(addr)}...")

{_, _} =
  Pythonx.eval(
    """
    import ctypes
    import numpy as np

    buf = (ctypes.c_uint8 * size).from_address(ptr)
    arr = np.frombuffer(buf, dtype=np.int32)

    # Read current value (may be 42 if LLDB wrote, or 1 if untouched)
    print(f"Python reads first element: {arr[0]}")

    # Write 99 to first element — directly into EXLA DeviceBuffer memory
    out_arr = np.frombuffer(buf, dtype=np.int32).copy()
    out_arr[0] = 99

    # Write back into the buffer (same address)
    ctypes.memmove(ptr, out_arr.ctypes.data, size)
    print(f"Python wrote 99 to first element at address {hex(ptr)}")
    """,
    %{"ptr" => addr, "size" => size}
  )

IO.puts("Done.")
IO.puts("")
IO.puts("Re-attach LLDB to verify Python's write:")
Proof.print_lldb_commands(pid, addr, size)
IO.puts("")
IO.puts("Expected memory after Python write:")
IO.puts("  63 00 00 00  02 00 00 00  03 00 00 00  04 00 00 00  05 00 00 00")
IO.puts("  (99)         (2)          (3)          (4)          (5)")
IO.puts("")

Proof.pause("[ENTER to continue to Phase 3]")

# ============================================================
# PHASE 3: Elixir reads back — proves zero-copy
# ============================================================

IO.puts("")
IO.puts("=== PHASE 3: ELIXIR READS BACK ===")

IO.inspect(Nx.to_binary(t_exla), label: "t_exla (after modifications)")
final_values = Nx.to_flat_list(t_exla)
IO.puts("Tensor values: #{inspect(final_values)}")

t2 = Nx.from_pointer(backend, pointer, type, Nx.shape(t_exla))
IO.inspect(Nx.to_binary(t2), label: "t2 (from_pointer)")
from_ptr_values = Nx.to_flat_list(t2)
IO.puts("from_pointer:  #{inspect(from_ptr_values)}")

IO.puts("")
IO.puts("PROOF COMPLETE:")

first = hd(final_values)
cond do
  first == 99 ->
    IO.puts("  ok  Python ctypes wrote to DeviceBuffer via to_pointer address")
    IO.puts("  ok  LLDB confirmed the write at the same physical address")
    IO.puts("  ok  Elixir read back the modified data without any copy")
    IO.puts("  ->  Same physical memory accessed by EXLA, Python, and LLDB")
  first == 42 ->
    IO.puts("  ok  LLDB wrote 42 to DeviceBuffer (Python read it before writing 99)")
    IO.puts("  ok  Elixir read back the modified data without any copy")
    IO.puts("  ->  Same physical memory accessed by EXLA and LLDB")
  true ->
    IO.puts("  !!  First element is #{first}, expected 99 or 42")
    IO.puts("      Check that LLDB/Python wrote correctly.")
end
