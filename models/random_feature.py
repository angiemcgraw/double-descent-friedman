"""
Caolan Disini
Last updated: April 9th, 2026

Random feature regression model using Fourier features. This model 
is a linear regression model that uses random Fourier features to 
approximate a kernel function. The complexity parameter controls 
the number of random features used in the model. 

"""
import numpy as np
import rfflearn.cpu as rfflearn
from models.base_model import BaseModel

class RandomFeatureRegression(BaseModel):
    def __init__(self, input_dim, complexity=10):
        """
        input_dim: number of input features
        complexity: number of random features
        """
        super().__init__(complexity)
        self.input_dim = input_dim
        
        # Kernel regressor with random Fourier features.
        self.reg = rfflearn.RFFRegressor(dim_kernel=self.complexity, std_kernel=0.5)

        # Kernel regressor with orthogonal random features.
        # reg = rfflearn.ORFRegressor(dim_kernel=self.input_dim, std_kernel=5.0)

        # Kernel regressor with quasi-random Fourier features.
        # reg = rfflearn.QRFRegressor(dim_kernel=self.input_dim, std_kernel=5.0)
        #self.reg = rfflearn.RFFGPR(dim_kernel=self.complexity, std_kernel=5.0, std_error=1.0)


    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)