# backend/metrics.py
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# The two metrics the judges want to see
PREDICTION_COUNT = Counter('predictions_total', 'Total number of predictions made')
DRIFT_COUNT = Counter('drift_detected_total', 'Total number of drifted inputs detected')

def get_metrics():
    """Generates the metrics page for Prometheus to scrape."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)