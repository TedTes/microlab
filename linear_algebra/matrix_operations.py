
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