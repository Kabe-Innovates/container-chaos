#!/bin/bash
# Entry point for Docker container to handle environment-based configuration

import os
import sys

# Generate prometheus.yml with correct API port
api_port = os.getenv('API_PORT', '8000')

prometheus_config = f"""global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:{api_port}']
"""

with open('/etc/prometheus/prometheus.yml', 'w') as f:
    f.write(prometheus_config)

print(f"✅ Generated prometheus.yml with API_PORT={api_port}")
