import numpy as np
from sklearn.utils import resample

class ThresholdEstimator:
    def __init__(self, x, y, grid_points=100):
        """Initialize the estimator with data and grid search resolution."""
        self.x = np.array(x)
        self.y = np.array(y)
        self.grid_points = grid_points
        self.threshold = None
    
    def fit(self):
        """Find the threshold that minimizes mean squared error (MSE)."""
        grid = np.linspace(self.x.min(), self.x.max(), self.grid_points)
        best_threshold, best_mse = None, float('inf')
        
        for t in grid:
            below, above = self.x <= t, self.x > t
            if below.sum() == 0 or above.sum() == 0:
                continue  # Avoid degenerate cases
            
            y_pred = np.where(below, self.y[below].mean(), self.y[above].mean())
            mse = np.mean((self.y - y_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_threshold = t
        
        self.threshold = best_threshold
        return best_threshold
    
    def bootstrap_std_error(self, n_bootstrap=1000):
        """Compute bootstrapped standard errors for the estimated threshold."""
        if self.threshold is None:
            raise ValueError("Model must be fitted before computing standard errors.")
        
        thresholds = []
        for _ in range(n_bootstrap):
            x_resampled, y_resampled = resample(self.x, self.y)
            est = ThresholdEstimator(x_resampled, y_resampled, self.grid_points)
            thresholds.append(est.fit())
        
        return np.std(thresholds)
    
    @staticmethod
    def monte_carlo_simulation(n_simulations=100, n_samples=500, true_break=None):
        """Test the estimator on synthetic data with and without a structural break."""
        results = []
        for _ in range(n_simulations):
            x = np.random.uniform(0, 10, n_samples)
            if true_break is None:
                y = np.random.normal(0, 1, n_samples)
            else:
                y = np.where(x < true_break, np.random.normal(0, 1, n_samples), np.random.normal(2, 1, n_samples))
            
            estimator = ThresholdEstimator(x, y)
            estimated_threshold = estimator.fit()
            std_error = estimator.bootstrap_std_error()
            
            results.append((estimated_threshold, std_error))
        
        return results
