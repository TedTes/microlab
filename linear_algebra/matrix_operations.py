
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


matrix_ops = MatrixOperations()

A = np.array([[1, 2], 
              [3, 4]])
B = np.array([[5, 6], 
              [7, 8]])

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
A = np.array([[1, 2], 
              [3, 4]])
print(f"Regular det: {matrix_ops.determinant(A)}")

# Test 4: 3×3 matrix
B = np.array([[1, 0, 2], 
              [0, 1, 3], 
              [0, 0, 1]])
print(f"3×3 det: {matrix_ops.determinant(B)}") 