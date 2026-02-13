"""
Configuration management for the ML Pipeline.
Handles all environment-based and file-based configurations with edge cases.
"""
import os
from pathlib import Path
from typing import Optional

class Config:
    """Configuration class with environment variable support and validation."""
    
    # Paths - with validation for existence and accessibility
    @staticmethod
    def get_models_dir() -> str:
        """Get models directory with fallback."""
        return os.getenv('MODELS_DIR', 'models')
    
    @staticmethod
    def get_data_dir() -> str:
        """Get data directory with fallback."""
        return os.getenv('DATA_DIR', 'data')
    
    @staticmethod
    def get_model_path() -> str:
        """Get full model path with validation."""
        models_dir = Config.get_models_dir()
        model_name = os.getenv('MODEL_NAME', 'model_v1.pkl')
        return os.path.join(models_dir, model_name)
    
    @staticmethod
    def get_features_path() -> str:
        """Get features path with validation."""
        models_dir = Config.get_models_dir()
        features_name = os.getenv('FEATURES_NAME', 'features.pkl')
        return os.path.join(models_dir, features_name)
    
    @staticmethod
    def get_raw_data_path() -> str:
        """Get raw data CSV path."""
        data_dir = Config.get_data_dir()
        data_file = os.getenv('RAW_DATA_FILE', 'raw.csv')
        return os.path.join(data_dir, data_file)
    
    @staticmethod
    def get_drift_log_path() -> str:
        """Get drift log CSV path."""
        data_dir = Config.get_data_dir()
        drift_file = os.getenv('DRIFT_LOG_FILE', 'drift_log.csv')
        return os.path.join(data_dir, drift_file)
    
    # Server Configuration
    @staticmethod
    def get_api_port() -> int:
        """Get API port with validation."""
        try:
            port = int(os.getenv('API_PORT', '8000'))
            if not (1 <= port <= 65535):
                raise ValueError(f"Port {port} out of valid range [1, 65535]")
            return port
        except ValueError as e:
            print(f"⚠️ Invalid API_PORT: {e}, using default 8000")
            return 8000
    
    @staticmethod
    def get_api_host() -> str:
        """Get API host with validation."""
        host = os.getenv('API_HOST', '0.0.0.0')
        if not isinstance(host, str) or not host.strip():
            return '0.0.0.0'
        return host
    
    @staticmethod
    def get_prometheus_port() -> int:
        """Get Prometheus port with validation."""
        try:
            port = int(os.getenv('PROMETHEUS_PORT', '9090'))
            if not (1 <= port <= 65535):
                raise ValueError(f"Port {port} out of valid range")
            return port
        except ValueError as e:
            print(f"⚠️ Invalid PROMETHEUS_PORT: {e}, using default 9090")
            return 9090
    
    @staticmethod
    def get_grafana_port() -> int:
        """Get Grafana port with validation."""
        try:
            port = int(os.getenv('GRAFANA_PORT', '3000'))
            if not (1 <= port <= 65535):
                raise ValueError(f"Port {port} out of valid range")
            return port
        except ValueError as e:
            print(f"⚠️ Invalid GRAFANA_PORT: {e}, using default 3000")
            return 3000
    
    # Model Configuration
    @staticmethod
    def get_contamination_rate() -> float:
        """Get contamination rate for Isolation Forest with validation."""
        try:
            rate = float(os.getenv('CONTAMINATION_RATE', '0.01'))
            if not (0.0 <= rate <= 1.0):
                raise ValueError(f"Contamination rate {rate} out of range [0.0, 1.0]")
            return rate
        except ValueError as e:
            print(f"⚠️ Invalid CONTAMINATION_RATE: {e}, using default 0.01")
            return 0.01
    
    @staticmethod
    def get_random_state() -> int:
        """Get random state seed for reproducibility."""
        try:
            return int(os.getenv('RANDOM_STATE', '42'))
        except ValueError:
            return 42
    
    @staticmethod
    def get_test_size() -> float:
        """Get test split ratio with validation."""
        try:
            size = float(os.getenv('TEST_SIZE', '0.2'))
            if not (0.0 < size < 1.0):
                raise ValueError(f"Test size {size} out of range (0.0, 1.0)")
            return size
        except ValueError as e:
            print(f"⚠️ Invalid TEST_SIZE: {e}, using default 0.2")
            return 0.2
    
    @staticmethod
    def get_n_estimators() -> int:
        """Get number of estimators for Random Forest."""
        try:
            n = int(os.getenv('N_ESTIMATORS', '50'))
            if n <= 0:
                raise ValueError(f"N_ESTIMATORS must be > 0, got {n}")
            return n
        except ValueError as e:
            print(f"⚠️ Invalid N_ESTIMATORS: {e}, using default 50")
            return 50
    
    # Feature Configuration
    @staticmethod
    def get_features() -> list:
        """Get feature list with fallback."""
        features_str = os.getenv('FEATURES', None)
        if features_str:
            return features_str.split(',')
        return [
            'discount_percent',
            'discounted_price',
            'price',
            'quantity_sold',
            'rating',
            'review_count'
        ]
    
    @staticmethod
    def get_target_column() -> str:
        """Get target column name."""
        return os.getenv('TARGET_COLUMN', 'total_revenue')
    
    # Utility methods
    @staticmethod
    def ensure_dirs_exist() -> None:
        """Create required directories if they don't exist."""
        for directory in [Config.get_models_dir(), Config.get_data_dir()]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def validate_model_files() -> bool:
        """Check if required model files exist."""
        model_path = Config.get_model_path()
        features_path = Config.get_features_path()
        return os.path.isfile(model_path) and os.path.isfile(features_path)
    
    @staticmethod
    def validate_raw_data() -> bool:
        """Check if raw data file exists."""
        return os.path.isfile(Config.get_raw_data_path())
