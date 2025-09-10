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
    def magnitude(self, vector:np.ndarray) -> float:
        """
        Calculate the magnitude (length) of a vector.

        Args:
            vector: Input vector
            
        Returns:
            Magnitude of the vector
        """
        magnitude_result = np.linalg.norm(vector)

        return magnitude_result

    def cosine_similarity(self, vector_a: np.ndarray, vector_b:np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Formula: cos(θ) = (A · B) / (|A| × |B|)
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """

        dot_prod = self.dot_product(vector_a, vector_b)
        mag_a = self.magnitude(vector_a)
        mag_b = self.magnitude(vector_b)

        if mag_a == 0  or mag_b == 0:
            return 0.0

        cosine_sim = dot_prod / (mag_a * mag_b)

        return cosine_sim
    def angle_between_vectors(self, vector_a:np.ndarray, vector_b:np.ndarray, degrees:bool=True) -> float:

        """
        Calculate angle between two vectors.
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            degrees: Return angle in degrees (True) or radians (False)
            
        Returns:
            Angle between vectors
        """
        cos_sim = self.cosine_similarity(vector_a, vector_b)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angle_radians = np.arccos(cos_sim)

        if degrees:
            angle_result=np.degrees(angle_radians)
        else:
            angle_result = angle_radians
        return angle_result

vector_ops = VectorOperations()
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = vector_ops.dot_product(a,b)
print("dot product result")
print(result)

v = np.array([3, 4])
result = vector_ops.magnitude(v)
print("2d vector magnitude")
print(result)

# Test with 3D vector
v2 = np.array([1, 2, 2])  
result2 = vector_ops.magnitude(v2)
print("3d vector magnitude")
print(result2)


print("##### cosine similarity ####")
# Identical vectors (should give 1.0)
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
print(vector_ops.cosine_similarity(a, b))

# Opposite vectors (should give -1.0)
a = np.array([1, 2, 3])
b = np.array([-1, -2, -3])  
print(vector_ops.cosine_similarity(a, b)) 

# Perpendicular vectors (should give 0.0)
a = np.array([1, 0])
b = np.array([0, 1])
print(vector_ops.cosine_similarity(a, b))
print("#### angle test#####")
# Same direction (should give 0°)
a = np.array([1, 1])
b = np.array([2, 2])  # Same direction, different magnitude
print(vector_ops.angle_between_vectors(a, b)) 

# Perpendicular vectors (should give 90°)
a = np.array([1, 0])
b = np.array([0, 1])
print(vector_ops.angle_between_vectors(a, b))
# Opposite direction (should give 180°)
a = np.array([1, 0])
b = np.array([-1, 0])
print(vector_ops.angle_between_vectors(a, b))