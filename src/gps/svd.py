import numpy as np

class SVD:
    def __init__(self, data):
        self.data = np.array(data)
        self.U = None
        self.S = None
        self.Vt = None

    def decompose(self):
        # compute full SVD
        self.U, self.S, self.Vt = np.linalg.svd(self.data, full_matrices=False)

    def reconstruct(self, k=None):
        if self.U is None or self.S is None or self.Vt is None:
            raise ValueError("SVD has not been performed. Call decompose() first.")

        # default: full rank reconstruction
        if k is None:
            k = len(self.S)

        # keep only top k singular values & vectors
        Uk = self.U[:, :k]
        Sk = np.diag(self.S[:k])
        Vtk = self.Vt[:k, :]

        return Uk @ Sk @ Vtk
