class NxTensor:
    """A thin wrapper around a numpy array, exposing the fields nx_py needs on the Elixir side."""

    def __init__(self, array):
        self.array = array

    @property
    def pointer(self):
        return self.array.ctypes.data

    @property
    def dtype(self):
        return str(self.array.dtype)

    @property
    def shape(self):
        return self.array.shape

    def to_nx(self, mode):
        metadata = (self.dtype, self.shape)

        if mode == "pointer":
            return (self.pointer, self.array.nbytes, *metadata)
        if mode == "binary":
            return (self.array.tobytes(), *metadata)

        raise ValueError(f"unsupported Nx conversion mode: {mode}")
