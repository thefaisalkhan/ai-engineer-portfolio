# Phase 5 — Week 9: Cloud ML Platforms

**Job relevance**: 88% of senior AI engineer roles require cloud-specific ML experience. SageMaker + Vertex AI are the two most demanded platforms.

## What This Week Covers

| Platform | Key Services Used |
|----------|------------------|
| AWS SageMaker | Training jobs, SageMaker Pipelines, Model Registry, real-time endpoints |
| GCP Vertex AI | Custom training, Vertex Pipelines, Model Registry, prediction endpoints |
| Azure ML | Compute clusters, AutoML, MLflow integration, managed endpoints |

## Architecture

```
Data → Feature Store → Training Job → Model Registry → Endpoint → Monitor
         │                                    │
    (versioned)                        (approval gate)
```

## AWS SageMaker Workflow

```python
# 1. Training job
estimator = sagemaker.sklearn.SKLearn(
    entry_point="train.py",
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    role=role,
    hyperparameters={"n-estimators": 100, "max-depth": 6},
)
estimator.fit({"train": s3_train_path})

# 2. Register to Model Registry
model_package = estimator.register(
    content_types=["application/json"],
    response_types=["application/json"],
    model_package_group_name="MyMLModels",
    approval_status="PendingManualApproval",
)

# 3. Deploy endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="my-model-endpoint",
)
```

## GCP Vertex AI Workflow

```python
# 1. Custom training job
job = aiplatform.CustomTrainingJob(
    display_name="my-training-job",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
    requirements=["scikit-learn==1.2.0"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)
model = job.run(
    dataset=dataset,
    target_column="label",
    machine_type="n1-standard-4",
)

# 2. Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3,  # autoscaling
)
```

## Key Concepts

### Managed Feature Stores
- **SageMaker Feature Store**: online (low-latency) + offline (S3) feature serving
- **Vertex Feature Store**: entity-based feature serving with point-in-time correctness

### Cloud Model Monitoring
Both platforms provide built-in monitoring for:
- Data quality drift (input distribution changes)
- Model quality drift (prediction distribution changes)
- Feature attribution drift (SHAP-based)

### Cost Comparison (approximate)
| Task | SageMaker | Vertex AI |
|------|-----------|-----------|
| ml.m5.xlarge / n1-standard-4 (training) | $0.23/hr | $0.19/hr |
| Real-time endpoint (ml.t2.medium) | $0.065/hr | ~$0.05/hr |
| Spot/preemptible training | 70% discount | 60-91% discount |

## Files

- `sagemaker_pipeline.py` — SageMaker Pipelines for end-to-end ML workflow
- `vertex_ai_training.py` — Vertex AI custom training and deployment
- `azure_ml_experiment.py` — Azure ML with MLflow tracking integration
- `feature_store_demo.py` — Managed feature store patterns

## When to Use Each Platform

| Scenario | Recommended |
|----------|-------------|
| Already on AWS | SageMaker |
| Already on GCP | Vertex AI |
| Need AutoML quickly | Azure AutoML or Vertex AutoML |
| Multi-cloud / open source | Use MLflow + Kubernetes (any cloud) |
| Regulated industry (finance/health) | Azure (compliance certifications) |
