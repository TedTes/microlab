import numpy as np
# import matplotlib.pyplot as plt 
from typing import List, Dict , Tuple
import click


class VectorOperations():
    """
    Interactive vector operations for understanding similarity and relationships.
    
    Key Applications:
    - Recommendation systems (user similarity)
    - Search engines (document similarity) 
    - Image processing (feature similarity)
    - Natural language processing (word similarity)
    """

    def __init__(self):
        self.vector_history = []  # Store vectors for analysis
    def dot_product(self, vector_a:np.ndarray, vector_b: np.ndarray) -> float:
        """
        Calculate dot product of two vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector
        
    Returns:
        Dot product value
        """
        if len(vector_a) != len(vector_b):
          raise ValueError("Vectors must have same length")
        dot_product_result = np.dot(vector_a, vector_b)

        return dot_product_result



vector_ops = VectorOperations()
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = vector_ops.dot_product(a,b)
print("dot product result")
print(result)