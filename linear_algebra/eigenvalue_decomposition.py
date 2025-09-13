import numpy as np
class EigenValueDecompositionOperations:
    def eigenvalue_decomposition(self, matrix:np.ndarray)->tuple:
        """
        Calculate eigenvalues and eigenvectors of a square matrix.

        Args:
        matrix: Square matrix (n × n)

        Returns:
        tuple: (eigenvalues, eigenvectors)
        - eigenvalues: 1D array of eigenvalues
        - eigenvectors: 2D array where each column is an eigenvector
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Eigenvalue decomposition only defined for square matrices")
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        return eigenvalues, eigenvectors




eigen_ops = EigenValueDecompositionOperations()

A = np.array([[3, 1],
              [0, 2]])

eigenvals, eigenvecs = eigen_ops.eigenvalue_decomposition(A)

print("Eigenvalues:", eigenvals)
# Expected: [3, 2] 

print("Eigenvectors:")
print(eigenvecs)
# Expected: Each column should be an eigenvector

# Verify the eigenvalue equation: A @ v = λ @ v
for i in range(len(eigenvals)):
    v = eigenvecs[:, i]  # i-th eigenvector (column)
    λ = eigenvals[i]     # i-th eigenvalue
    
    left_side = A @ v    # A × v
    right_side = λ * v   # λ × v
    
    print(f"Eigenvalue {λ:.3f}:")
    print(f"A @ v = {left_side}")
    print(f"λ @ v = {right_side}")
    print(f"Close? {np.allclose(left_side, right_side)}")
    print()