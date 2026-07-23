# utils/time_series.py
"""
Time series analysis for leaf health tracking
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """
    Analyze time series data for leaf health tracking.
    """
    
    def __init__(self):
        self.data = []
        self.timestamps = []
    
    def add_observation(self, timestamp: datetime, metrics: Dict[str, Any]):
        """
        Add a new observation to the time series.
        
        Args:
            timestamp: Observation timestamp
            metrics: Dictionary of metrics
        """
        self.timestamps.append(timestamp)
        self.data.append(metrics)
    
    def get_trend(self, metric_key: str) -> Dict[str, Any]:
        """
        Calculate trend for a specific metric.
        
        Args:
            metric_key: Key of the metric to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.data) < 2:
            return {'error': 'Not enough data points'}
        
        # Extract values
        values = []
        for entry in self.data:
            if metric_key in entry:
                values.append(entry[metric_key])
        
        if len(values) < 2:
            return {'error': f'No data for metric: {metric_key}'}
        
        # Calculate trend
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        trend = z[0]  # Slope
        
        # Calculate moving average
        window = min(3, len(values))
        if window > 1:
            moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
        else:
            moving_avg = values
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)
        latest = values[-1]
        previous = values[-2] if len(values) > 1 else latest
        
        # Determine direction
        if trend > 0.1:
            direction = 'increasing'
        elif trend < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'metric': metric_key,
            'trend_slope': float(trend),
            'direction': direction,
            'latest_value': float(latest),
            'previous_value': float(previous),
            'change': float(latest - previous),
            'change_percent': float((latest - previous) / (previous + 1e-6) * 100),
            'mean': float(mean),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val),
            'moving_average': moving_avg.tolist(),
            'num_points': len(values)
        }
    
    def detect_anomalies(self, metric_key: str, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a metric time series.
        
        Args:
            metric_key: Key of the metric to analyze
            threshold: Z-score threshold for anomaly detection
        
        Returns:
            List of detected anomalies
        """
        if len(self.data) < 3:
            return []
        
        # Extract values
        values = []
        timestamps = []
        for i, entry in enumerate(self.data):
            if metric_key in entry:
                values.append(entry[metric_key])
                timestamps.append(self.timestamps[i])
        
        if len(values) < 3:
            return []
        
        # Calculate z-scores
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        z_scores = [(v - mean) / std for v in values]
        
        # Find anomalies
        anomalies = []
        for i, z in enumerate(z_scores):
            if abs(z) > threshold:
                anomalies.append({
                    'timestamp': timestamps[i].isoformat(),
                    'value': values[i],
                    'z_score': float(z),
                    'index': i,
                    'severity': 'high' if abs(z) > 3 else 'medium'
                })
        
        return anomalies
    
    def predict_next(self, metric_key: str, days_ahead: int = 1) -> Dict[str, Any]:
        """
        Predict next value for a metric.
        
        Args:
            metric_key: Key of the metric to predict
            days_ahead: Number of days to predict ahead
        
        Returns:
            Dictionary with prediction
        """
        if len(self.data) < 2:
            return {'error': 'Not enough data for prediction'}
        
        # Extract values
        values = []
        for entry in self.data:
            if metric_key in entry:
                values.append(entry[metric_key])
        
        if len(values) < 2:
            return {'error': f'No data for metric: {metric_key}'}
        
        # Simple linear regression
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        
        # Predict next
        next_idx = len(values)
        prediction = z[0] * next_idx + z[1]
        
        # Calculate confidence interval (simplified)
        residuals = np.array(values) - (z[0] * x + z[1])
        std_residual = np.std(residuals)
        confidence_interval = 1.96 * std_residual  # 95% confidence
        
        return {
            'metric': metric_key,
            'prediction': float(prediction),
            'confidence_interval': float(confidence_interval),
            'lower_bound': float(prediction - confidence_interval),
            'upper_bound': float(prediction + confidence_interval),
            'trend_slope': float(z[0]),
            'last_value': float(values[-1]),
            'days_ahead': days_ahead
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.data:
            return {'error': 'No data available'}
        
        summary = {
            'num_observations': len(self.data),
            'time_range': {
                'start': self.timestamps[0].isoformat(),
                'end': self.timestamps[-1].isoformat()
            },
            'metrics': {}
        }
        
        # Collect all metric keys
        all_keys = set()
        for entry in self.data:
            all_keys.update(entry.keys())
        
        # Calculate statistics for each metric
        for key in all_keys:
            values = []
            for entry in self.data:
                if key in entry:
                    values.append(entry[key])
            
            if values:
                summary['metrics'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'latest': float(values[-1]),
                    'first': float(values[0]),
                    'change': float(values[-1] - values[0])
                }
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert time series to dictionary.
        
        Returns:
            Dictionary with all data
        """
        return {
            'timestamps': [t.isoformat() for t in self.timestamps],
            'data': self.data,
            'num_observations': len(self.data)
        }
    
    def from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Load time series from dictionary.
        
        Args:
            data: Dictionary with time series data
        
        Returns:
            True if loaded successfully
        """
        try:
            self.timestamps = [datetime.fromisoformat(t) for t in data['timestamps']]
            self.data = data['data']
            return True
        except Exception as e:
            logger.error(f"Failed to load time series: {e}")
            return False