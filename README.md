cat << 'EOF' > README.md
# Amazon Revenue Predictor (Container Chaos)

A resilient, containerized machine learning API built to predict Amazon product revenue and dynamically detect data drift in production.

## The Endpoint (For Judges)

The inference API is live and ready for load testing:

**URL:** `https://journalistically-pleasureful-deneen.ngrok-free.dev/predict`
**Method:** `POST`

### Quick Test (cURL)
Copy and paste this to verify the API contract:

```bash
curl -X POST "[https://journalistically-pleasureful-deneen.ngrok-free.dev/predict](https://journalistically-pleasureful-deneen.ngrok-free.dev/predict)" \
     -H "Content-Type: application/json" \
     -d '{
           "discount_percent": 10.0,
           "discounted_price": 90.0,
           "price": 100.0,
           "quantity_sold": 2,
           "rating": 4.5,
           "review_count": 150
         }'