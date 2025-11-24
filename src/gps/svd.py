class SVD:
    def __init__(self, data):
        self.data = data
        self.U = None
        self.S = None
        self.Vt = None

    def decompose(self):
        # calculated the SVD of the matrix
        pass

    # TODO add a K parameter with a default of full rank
    def reconstruct(self):
        if self.U is None or self.S is None or self.Vt is None:
            raise ValueError("SVD has not been performed. Call decompose() first.")
        # reconstruct the matrix according to the given rank
        pass
