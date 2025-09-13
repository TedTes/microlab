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


    def principal_components(self, data:np.ndarray, n_components:int = None) -> dict:
        """
        Perform Principal Component Analysis (PCA).

        Args:
        data: Data matrix (samples × features)
        n_components: Number of components to keep (None = all)

        Returns:
        dict with 'components', 'eigenvalues', 'explained_variance_ratio'
        """

        # Step 1: Center data
        data_centered = data - np.mean(data, axis=0)

        # Step 2: Covariance matrix  
        cov_matrix = np.cov(data_centered.T)

        # Step 3: Eigendecomposition
        eigenvals, eigenvecs = self.eigenvalue_decomposition(cov_matrix)

        # Step 4: Sort descending
        sort_indices = np.argsort(eigenvals)[::-1]

        #  Step 5: Keep only n_components
        if n_components is not None:
            eigenvals_sorted = eigenvals_sorted[:n_components]
            eigenvecs_sorted = eigenvecs_sorted[:, :n_components]

        # Step 6: Explained variance
        total_variance = np.sum(eigenvals_sorted)
        explained_variance_ratio = eigenvals_sorted / total_variance


        return {
            'components': eigenvecs_sorted,           # Principal components
            'eigenvalues': eigenvals_sorted,          # Importance scores
            'explained_variance_ratio': explained_variance_ratio,  # Percentage explained
            'data_centered': data_centered            # For transformations
        }



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