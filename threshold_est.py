import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from itertools import combinations

class ThresholdEstimator:
    def __init__(self, x, y, covariates=None, grid_points=100, num_thresholds=1):
        """Initialize the estimator with data, optional covariates, grid search resolution, and number of thresholds."""
        self.x = np.array(x)
        self.y = np.array(y)
        self.covariates = np.array(covariates) if covariates is not None else None
        self.grid_points = grid_points
        self.num_thresholds = num_thresholds
        self.thresholds = None
    
    def fit(self):
        """Find the thresholds that minimize mean squared error (MSE) using linear regression."""
        grid = np.linspace(self.x.min(), self.x.max(), self.grid_points)
        best_thresholds, best_mse = None, float('inf')
        
        for thresholds in combinations(grid, self.num_thresholds):
            segments = np.digitize(self.x, bins=thresholds)
            X = np.column_stack((self.x, segments))
            if self.covariates is not None:
                X = np.column_stack((X, self.covariates))
            
            model = LinearRegression().fit(X, self.y)
            mse = np.mean((self.y - model.predict(X)) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_thresholds = thresholds
        
        self.thresholds = best_thresholds
        return best_thresholds
    
    def bootstrap_std_error(self, n_bootstrap=1000):
        """Compute bootstrapped standard errors for the estimated thresholds."""
        if self.thresholds is None:
            raise ValueError("Model must be fitted before computing standard errors.")
        
        threshold_samples = []
        for _ in range(n_bootstrap):
            x_resampled, y_resampled, covariates_resampled = resample(self.x, self.y, self.covariates)
            est = ThresholdEstimator(x_resampled, y_resampled, covariates_resampled, self.grid_points, self.num_thresholds)
            threshold_samples.append(est.fit())
        
        return np.std(threshold_samples, axis=0)
    
    @staticmethod
    def monte_carlo_simulation(n_simulations=100, n_samples=500, true_breaks=None):
        """Test the estimator on synthetic data with and without structural breaks."""
        results = []
        for _ in range(n_simulations):
            x = np.random.uniform(0, 10, n_samples)
            if true_breaks is None:
                y = np.random.normal(0, 1, n_samples)
            else:
                segments = np.digitize(x, bins=true_breaks)
                y = np.random.normal(segments, 1, n_samples)
            
            estimator = ThresholdEstimator(x, y, num_thresholds=len(true_breaks) if true_breaks else 1)
            estimated_thresholds = estimator.fit()
            std_error = estimator.bootstrap_std_error()
            
            results.append((estimated_thresholds, std_error))
        
        return results
