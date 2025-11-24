class LeastSquares:
    def __init__(self, satellite_data, observations):
        self.satellite_data = satellite_data
        self.observations = observations
        self.position = None

    def fit(self):
        # Prepare the design matrix and observation vector
        A = self._create_design_matrix()
        b = self.observations
        
        # Solve the least squares problem
        pass

    def predict(self):
        if self.position is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.position

    def _create_design_matrix(self):
        # Create the design matrix based on satellite data
        # This is a placeholder implementation
        pass

    def _calculate_row(self, data):
        # Placeholder for calculating a row of the design matrix
        pass

import numpy as np
