import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer

# Example 1D vector
vector = np.array([1, 2, 3, 4, 5])

# Reshape the vector to a 2D array with one feature
vector_reshaped = vector.reshape(-1, 1)

# Initialize and apply RobustScaler
robust_scaler = RobustScaler()
robust_scaled_vector = robust_scaler.fit_transform(vector_reshaped).flatten()
print("RobustScaler:", type(robust_scaled_vector))

# Initialize and apply QuantileTransformer (normal output distribution)
quantile_transformer = QuantileTransformer(output_distribution='normal')
quantile_transformed_vector = quantile_transformer.fit_transform(vector_reshaped).flatten()
print("QuantileTransformer:", quantile_transformed_vector)
