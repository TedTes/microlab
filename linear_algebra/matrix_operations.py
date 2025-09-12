
import numpy as np

class MatrixOperations():

    def matrix_multiply(self, matrix_a:np.ndarray, matrix_b:np.ndarray) -> np.ndarray:
        """
        Multiply two matrices.
        
        Args:
            matrix_a: First matrix (m × n)
            matrix_b: Second matrix (n × p)
            
        Returns:
            Result matrix (m × p)
        """
        if matrix_a.shape[1] != matrix_b.shape[0]:
          raise ValueError(f"Cannot multiply {matrix_a.shape} × {matrix_b.shape}")
        
        result = np.matmul(matrix_a, matrix_b)

        return result

    def determinant(self, matrix:np.ndarray) -> float:
        """
        Calculate the determinant of a square matrix.

        Args:
        matrix: Square matrix (n × n)

        Returns:
        Determinant value
        """
        # validation 
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Determinant only defined for square matrices")
        det_value = np.linalg.det(matrix)
        return det_value
    def matrix_inverse(self, matrix:np.ndarray) -> np.ndarray :
        """
        Calculate the inverse of a square matrix.

        Args:
        matrix: Square matrix (n × n)

        Returns:
        Inverse matrix

        Raises:
        ValueError: If matrix is singular (det = 0)
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Inverse only defined for square matrices")
        
        det_val = self.determinant(matrix)
        if abs(det_val)  < 1e-10:
            raise ValueError("Matrix is singular (determinant ≈ 0), cannot invert")
        inverse_matrix = np.linalg.inv(matrix)

        return inverse_matrix
    def matrix_transpose(self, matrix:np.ndarray) -> np.ndarray:
        """
        Calculate the transpose of a matrix.

        Args:
        matrix: Input matrix (m × n)

        Returns:
        Transposed matrix (n × m)
        """
        transposed = matrix.T  # or  np.transpose(matrix)

        return transposed

    def matrix_rank(self,matrix:np.ndarray)->int:
        """
        Calculate the rank of a matrix.

        Args:
        matrix: Input matrix (m × n)

        Returns:
        Rank of the matrix (integer)
        """
        rank = np.linalg.matrix_rank(matrix)
        return rank



matrix_ops = MatrixOperations()

A = np.array([[1, 2], 
              [3, 4]])
B = np.array([[5, 6], 
              [7, 8]])

C = np.array([[1, 0, 2], 
              [0, 1, 3], 
              [0, 0, 1]])

result = matrix_ops.matrix_multiply(A,B)
print("matrix multiplication result")
print(result)

# Neural network example
inputs = np.array([[0.8, 0.2, 0.9]])  # 1×3 (one sample, 3 features)
weights = np.array([[0.1, 0.4], 
                    [0.3, 0.7], 
                    [0.2, 0.5]])       # 3×2 (3 inputs, 2 outputs)


output = matrix_ops.matrix_multiply(inputs, weights)
print(f"Neural network output: {output}")


# Test 1: Identity matrix (should give 1.0)
I = np.array([[1, 0], 
              [0, 1]])
print(f"Identity det: {matrix_ops.determinant(I)}")

# Test 2: Singular matrix (should give 0.0)
singular = np.array([[1, 2], 
                     [2, 4]])  # Second row = 2 × first row
print(f"Singular det: {matrix_ops.determinant(singular)}") 

# Test 3: Regular matrix
print(f"Regular det: {matrix_ops.determinant(A)}")

# Test 4: 3×3 matrix

print(f"3×3 det: {matrix_ops.determinant(C)}") 


print("matrix inverse test")

# Test 1: Simple 2×2 matrix

A_inv = matrix_ops.matrix_inverse(A)
print("A inverse:", A_inv)

# Test the inverse property: A @ A_inv should equal identity
identity_check = A @ A_inv
print("A @ A_inv:", identity_check)  # Should be [[1,0], [0,1]]

# Test 2: Identity matrix (inverse of identity is identity)
I = np.array([[1, 0], 
              [0, 1]])
I_inv = matrix_ops.matrix_inverse(I)
print("Identity inverse:", I_inv)  # Should be [[1,0], [0,1]]

# Test 3: Try singular matrix (should raise error)
try:
    singular = np.array([[1, 2], 
                         [2, 4]])
    matrix_ops.matrix_inverse(singular)
except ValueError as e:
    print("Expected error:", e)


print("transpose of A")
A_T = matrix_ops.matrix_transpose(A)
print("Original shape:", A.shape)
print("Transposed shape:", A_T.shape)
print("Transposed:")
print(A_T)


print("#############MATRIX RANK ################")
# Test 1: Full rank matrix
A = np.array([[1, 2],
              [3, 4]])
rank_A = matrix_ops.matrix_rank(A)
print(f"Full rank 2×2 matrix rank: {rank_A}")  # Should be 2

# Test 2: Rank deficient matrix (dependent rows)
B = np.array([[1, 2],
              [2, 4]])  # Second row = 2 × first row
rank_B = matrix_ops.matrix_rank(B)
print(f"Rank deficient matrix rank: {rank_B}")  # Should be 1

# Test 3: Zero matrix
C = np.zeros((3, 3))
rank_C = matrix_ops.matrix_rank(C)
print(f"Zero matrix rank: {rank_C}")  # Should be 0

# Test 4: Rectangular matrix
D = np.array([[1, 2, 3],
              [4, 5, 6]])
rank_D = matrix_ops.matrix_rank(D)
print(f"2×3 matrix rank: {rank_D}")  # Should be 2

# Test 5: Identity matrix
I = np.eye(3)
rank_I = matrix_ops.matrix_rank(I)
print(f"3×3 identity matrix rank: {rank_I}")  # Should be 3