# Phase 5 — Week 10: Serverless ML & Infrastructure as Code

**Job relevance**: "Serverless technologies" appears explicitly in 85%+ of cloud-native AI engineer job descriptions. Terraform is the standard IaC tool.

## What This Week Covers

1. **AWS Lambda** — Package ML model as serverless function
2. **GCP Cloud Functions** — Event-driven ML inference
3. **AWS Fargate / GCP Cloud Run** — Container-based serverless (better for large models)
4. **Terraform** — Infrastructure as Code for ML infrastructure

## Why Serverless ML?

| Factor | Traditional Server | Serverless |
|--------|-------------------|------------|
| Cost | Pay 24/7 | Pay per request |
| Scaling | Manual / HPA | Automatic (0 → ∞) |
| Maintenance | OS patches, capacity planning | None |
| Cold start | None | 1-10s (mitigatable) |
| Model size | Any | ~250MB (Lambda); ~32MB (Cloud Fn) |
| Best for | High-traffic, large models | Sporadic traffic, small models |

## AWS Lambda ML Inference

```python
# lambda_function.py
import json
import pickle
import boto3

# Load model at cold start (cached between invocations)
s3 = boto3.client("s3")
s3.download_file("my-bucket", "model.pkl", "/tmp/model.pkl")
with open("/tmp/model.pkl", "rb") as f:
    MODEL = pickle.load(f)

def lambda_handler(event, context):
    features = json.loads(event["body"])["features"]
    prediction = MODEL.predict([features])[0]
    probability = MODEL.predict_proba([features])[0].tolist()
    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": int(prediction),
            "probability": probability,
        }),
    }
```

### Cold Start Optimization
```python
# Keep model in /tmp — persists between Lambda invocations
# Use Provisioned Concurrency for latency-critical endpoints
# Quantize model (joblib compress=3) to reduce package size
import joblib
joblib.dump(model, "model.pkl", compress=3)  # 70-80% size reduction
```

## Terraform for ML Infrastructure

```hcl
# main.tf — SageMaker + S3 + IAM with one command: terraform apply

resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "my-ml-artifacts-${var.environment}"
}

resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_sagemaker_endpoint_configuration" "ml_endpoint_config" {
  name = "ml-endpoint-config-${var.model_version}"
  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.ml_model.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
    initial_variant_weight = 1
  }
}

resource "aws_sagemaker_endpoint" "ml_endpoint" {
  name                 = "ml-production-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ml_endpoint_config.name
  tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

## GCP Cloud Run (Container Serverless)

```dockerfile
# Dockerfile for Cloud Run ML service
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Cloud Run sets PORT env variable
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
```

```bash
# Deploy to Cloud Run
gcloud run deploy ml-inference \
  --image gcr.io/PROJECT_ID/ml-inference:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 100 \
  --concurrency 80
```

## Decision Tree: Which Serverless to Use?

```
Model size < 250MB AND traffic < 10 req/s?
├── YES → AWS Lambda or GCP Cloud Functions
└── NO →
    Model needs GPU?
    ├── YES → AWS Fargate + GPU or SageMaker Async Inference
    └── NO → Cloud Run (GCP) or AWS Fargate (flexible container serverless)
```

## Files

- `lambda_inference/` — AWS Lambda function with sklearn model
- `cloud_run_service/` — GCP Cloud Run FastAPI inference service
- `terraform/` — IaC for SageMaker endpoint + S3 + IAM
- `cold_start_benchmark.py` — Measure and optimize cold start latency
