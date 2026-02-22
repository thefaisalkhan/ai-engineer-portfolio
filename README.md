<h1 align="center">Faisal Khan — AI Engineer</h1>

<p align="center">
  <strong>Software Engineer → Production-Ready AI Engineer</strong><br/>
  Building production-grade AI systems: ML pipelines, RAG, LLM agents, MLOps infrastructure
</p>

<p align="center">
  <a href="https://linkedin.com/in/faisal-khan-13b146148"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
  <a href="https://twitter.com/itzzfaisalkhan"><img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"/></a>
  <a href="mailto:fk8767@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
</p>

---

## About

SWE background with deep experience in testing, CI/CD, and distributed systems — now specializing in AI engineering. I build AI systems that actually ship: classical ML pipelines, production RAG systems, LLM agents with tool use, and MLOps infrastructure that monitors and retrains models automatically.

**Unfair advantage**: Most AI engineers can't write production-grade code. I can. Most software engineers don't understand the full ML stack. I do.

---

## What the Job Market Actually Demands (2026 Research)

> Scraped and synthesized from real job postings across LinkedIn, Indeed, and company career pages.

Every production AI engineer role in 2026 requires **three combined skill sets**. Candidates missing any one layer get filtered out:

```
Layer 1 — ML Science       Layer 2 — Software Eng       Layer 3 — MLOps/Cloud
─────────────────────      ──────────────────────        ──────────────────────
Statistics & Probability   Python (production-grade)     Docker & Kubernetes
Statistical Modeling       REST APIs & Microservices     CI/CD for ML (GitHub Actions)
ML Algorithms (sklearn)    FastAPI / Flask               MLflow (experiment tracking)
Feature Engineering        Async programming             Prometheus + Grafana
Deep Learning (PyTorch/TF) Testing & code quality        Drift detection
NLP (SpaCy, Hugging Face)  Object-oriented design        Cloud ML: SageMaker/Vertex AI
Time Series Analysis       gRPC / message queues         Serverless (Lambda/Cloud Fns)
Recommendation Systems     SQL + vector databases        Infrastructure as Code
Model Explainability (SHAP) Docker                       Model registry + versioning
LLMs, RAG, Agents          Redis, Celery                 Automated retraining
```

**Most common required skills** (frequency from job postings):

| Skill | Frequency | Phase Covered |
|-------|-----------|---------------|
| Python | 100% | Phase 0 |
| TensorFlow / PyTorch / Keras | 98% | Phase 1, Phase 2 |
| Scikit-learn | 97% | Phase 2 |
| Statistical Modeling | 94% | Phase 1 |
| REST APIs + Microservices | 93% | Phase 0, Phase 6 |
| Feature Engineering | 91% | Phase 2 |
| Docker + Kubernetes | 90% | Phase 0, Phase 5 |
| Cloud (AWS/Azure/GCP) | 88% | Phase 5 |
| MLflow / MLOps | 85% | Phase 5 |
| Hypothesis Testing / A/B | 82% | Phase 1 |
| NLP (SpaCy/NLTK/Hugging Face) | 80% | Phase 2, Phase 3 |
| LLMs + RAG + Agents | 78% | Phase 3, Phase 4 |
| Model Explainability (SHAP/LIME) | 74% | Phase 2 |
| Time Series Analysis | 68% | Phase 2 |
| Recommendation Systems | 61% | Phase 2 |
| CI/CD for ML | 89% | Phase 5 |
| Prometheus / Grafana | 76% | Phase 5 |

---

## Complete AI Engineer Roadmap — 18 Months, 55+ Projects

This repo documents my structured path from SWE to production-ready AI Engineer across 7 phases. Each phase maps directly to real job requirements.

| Phase | Focus | Duration | Projects | Job Skills Covered |
|-------|-------|----------|----------|--------------------|
| [Phase 0](#phase-0-python-for-ai) | Python for AI | 4 weeks | 4 | Python, FastAPI, Docker, pytest |
| [Phase 1](#phase-1-statistics--math-for-ml) | Statistics & Math for ML | 4 weeks | 4 | Statistical modeling, hypothesis testing, A/B testing |
| [Phase 2](#phase-2-ml-foundations--classical-ml) | ML Foundations & Classical ML | 12 weeks | 12 | sklearn, PyTorch, TF/Keras, NLP, time series, SHAP |
| [Phase 3](#phase-3-llm-core) | LLM Core | 8 weeks | 8 | LLMs, fine-tuning, embeddings, Hugging Face |
| [Phase 4](#phase-4-rag--agents) | RAG + Agents | 8 weeks | 8 | RAG, LangChain, LangGraph, vector DBs |
| [Phase 5](#phase-5-mlops--cloud-ml) | MLOps + Cloud ML | 10 weeks | 10 | MLflow, CI/CD, drift, SageMaker, Vertex AI |
| [Phase 6](#phase-6-enterprise-ai-systems) | Enterprise AI Systems | 16 weeks | 10+ | Microservices, K8s, system design |

---

## Phase 0: Python for AI

> **Job Skills**: Python (production-grade), REST APIs, Docker, testing, async programming

> NumPy · Pandas · OOP · FastAPI · Docker · pytest

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-numpy](./python-foundations/week1-numpy/) | Vector ops, broadcasting, linear algebra, SVD | NumPy, vectorization |
| [week2-pandas](./python-foundations/week2-pandas/) | Data cleaning, feature engineering, EDA | Pandas, matplotlib |
| [week3-oop](./python-foundations/week3-oop/) | Decorators, generators, async/await, context managers | Advanced Python |
| [week4-fastapi](./python-foundations/week4-fastapi/) | REST API, Docker, pytest, prediction endpoint | FastAPI, testing |

**Why this phase**: 100% of AI engineer job postings list Python as a requirement. Production teams demand Pythonic, testable, async-ready code — not just scripts.

---

## Phase 1: Statistics & Math for ML

> **Job Skills**: Statistical modeling, predictive analytics, hypothesis testing, A/B testing, experimental design

> Statistics · Probability · Regression Analysis · Hypothesis Testing · Experiment Design

This phase fills the gap that kills most "AI engineer" candidates in interviews. Every job description asks for *statistical modeling* and *predictive analytics* — this is the foundation.

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-probability-stats](./statistics/week1-probability-stats/) | Probability distributions, central limit theorem, Bayes' theorem, sampling | Probability theory, NumPy, SciPy |
| [week2-hypothesis-testing](./statistics/week2-hypothesis-testing/) | t-tests, ANOVA, chi-square, p-values, confidence intervals, Type I/II errors | SciPy, statsmodels |
| [week3-statistical-modeling](./statistics/week3-statistical-modeling/) | Linear/logistic regression (statistical lens), correlation, multicollinearity, OLS | statsmodels, seaborn |
| [week4-ab-testing](./statistics/week4-ab-testing/) | Experiment design, power analysis, sample size calculation, multiple testing correction | SciPy, Plotly |

**Why this phase matters**: 94% of AI/ML job postings explicitly require *statistical modeling* skills. A/B testing appears in 82%. Without this phase, you cannot pass the data science/ML rounds at enterprise companies.

**Concepts covered**:
- Descriptive statistics: mean, median, variance, skewness, kurtosis
- Probability distributions: normal, binomial, Poisson, exponential
- Bayes' theorem and conditional probability
- Null/alternative hypothesis, p-values, significance levels
- t-test (one-sample, two-sample, paired), ANOVA, chi-square
- Linear regression (OLS, R², adjusted R², F-statistic, residuals)
- Logistic regression (log-odds, odds ratios, Wald test)
- Experiment design: control groups, randomization, stratification
- Power analysis and minimum detectable effect
- Bonferroni correction, FDR for multiple comparisons

---

## Phase 2: ML Foundations & Classical ML

> **Job Skills**: TensorFlow, PyTorch, Keras, scikit-learn, feature engineering, data preprocessing, NLP, time series, model explainability, recommendation systems

> scikit-learn · PyTorch · TensorFlow/Keras · XGBoost · SHAP · SpaCy · ARIMA

This is the most important phase for passing technical screens. Real job postings require hands-on experience with *all* the major frameworks plus domain-specific ML (NLP, time series, recommendations).

### Classical ML & Feature Engineering (Weeks 1–5)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-scratch-algorithms](./ml-fundamentals/week1-scratch-algorithms/) | Linear regression, logistic regression, decision tree — math → code | Gradient descent, backprop |
| [week2-classification](./ml-fundamentals/week2-classification/) | Fraud detection: 5 models, precision-recall, AUC-ROC, threshold tuning | sklearn, imbalanced data |
| [week3-feature-engineering](./ml-fundamentals/week3-feature-engineering/) | SMOTE, GridSearchCV, feature selection, cross-validation, pipelines | Feature eng, sklearn Pipelines |
| [week4-neural-networks](./ml-fundamentals/week4-neural-networks/) | NN from scratch → PyTorch Lightning, MNIST 99%+, GPU training | PyTorch, backprop |
| [week5-ensembles](./ml-fundamentals/week5-ensembles/) | XGBoost, LightGBM, CatBoost, stacking, Kaggle competition strategy | Boosting, bagging, stacking |

### Deep Learning & Computer Vision (Weeks 6–7)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week6-cnn-cv](./ml-fundamentals/week6-cnn-cv/) | CNN for CIFAR-10 → transfer learning (ResNet, EfficientNet), image classification API | PyTorch, torchvision, FastAPI |
| [week7-tensorflow-keras](./ml-fundamentals/week7-tensorflow-keras/) | Rebuild CNN + regression models in TensorFlow/Keras, compare with PyTorch, TF Serving | TensorFlow, Keras, TF Serving |

**Why TensorFlow/Keras**: 98% of ML job descriptions list TF/Keras alongside PyTorch. TF is dominant in production pipelines at enterprise companies. You must know both.

### Model Explainability (Week 8)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week8-explainability](./ml-fundamentals/week8-explainability/) | SHAP values (TreeExplainer, DeepExplainer), LIME, partial dependence plots, feature importance comparison across 3 model types | SHAP, LIME, interpretML |

**Why explainability**: 74% of job postings mention model interpretability. Regulated industries (finance, healthcare) require it by law. SHAP is now a standard interview topic.

**Concepts covered**:
- SHAP: Shapley values, TreeExplainer, KernelExplainer, DeepExplainer
- SHAP summary plots, waterfall plots, beeswarm plots
- LIME: local surrogate models, text and tabular explanations
- Partial dependence plots (PDP) and ICE plots
- Permutation feature importance vs built-in importance
- Model cards and evaluation reports

### Unsupervised Learning (Week 9)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week9-unsupervised](./ml-fundamentals/week9-unsupervised/) | K-Means, DBSCAN, hierarchical clustering, PCA, t-SNE, UMAP — customer segmentation | Dimensionality reduction, clustering |

### NLP Foundations (Week 10)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week10-nlp-foundations](./ml-fundamentals/week10-nlp-foundations/) | Tokenization, POS tagging, NER with SpaCy + NLTK, text classification (TF-IDF + sklearn), sentiment analysis without LLMs | SpaCy, NLTK, classical NLP |

**Why NLP foundations**: 80% of job postings list NLP skills. This is *pre-LLM* NLP — the fundamentals that interviewers test even when the role is LLM-focused. SpaCy and NLTK appear in virtually every NLP job description.

**Concepts covered**:
- Text preprocessing: tokenization, stemming, lemmatization, stop words
- Bag of Words, TF-IDF, n-grams
- Part-of-speech (POS) tagging
- Named entity recognition (NER) with SpaCy
- Text classification with sklearn
- Sentiment analysis (VADER, TextBlob, sklearn)
- Word embeddings: Word2Vec, GloVe (conceptual + usage)

### Time Series Analysis (Week 11)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week11-time-series](./ml-fundamentals/week11-time-series/) | ARIMA/SARIMA for demand forecasting, Prophet for trend/seasonality, LSTM for TS prediction, model comparison dashboard | statsmodels, Prophet, PyTorch LSTM |

**Concepts covered**:
- Stationarity, differencing, ACF/PACF plots
- ARIMA, SARIMA, SARIMAX
- Facebook Prophet: trend, seasonality, holidays
- LSTM for univariate and multivariate time series
- Cross-validation for time series (walk-forward validation)
- Forecasting evaluation: MAE, RMSE, MAPE, SMAPE

### Recommendation Systems (Week 12)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week12-recommendations](./ml-fundamentals/week12-recommendations/) | Collaborative filtering (user-based, item-based), matrix factorization (SVD, ALS), content-based filtering, hybrid recommender — deployed as FastAPI | surprise, implicit, FastAPI |

**Concepts covered**:
- Content-based filtering: cosine similarity, TF-IDF features
- Collaborative filtering: user-based, item-based (k-NN)
- Matrix factorization: SVD, ALS (Alternating Least Squares)
- Hybrid recommenders: weighted, switching, cascade
- Cold-start problem and handling sparse data
- Evaluation: precision@k, recall@k, NDCG, MAP

---

## Phase 3: LLM Core

> **Job Skills**: LLMs, fine-tuning, Hugging Face, embeddings, prompt engineering, evaluation, NLP tasks at scale

> OpenAI · Anthropic Claude · Hugging Face · LangChain · Embeddings

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-llm-fundamentals](./llm-core/week1-llm-fundamentals/) | Multi-provider API wrapper, cost + latency tracker | LLM APIs |
| [week2-fine-tuning](./llm-core/week2-fine-tuning/) | LoRA fine-tuning, BLEU/ROUGE eval, cost analysis | Fine-tuning, PEFT |
| [week3-prompt-engineering](./llm-core/week3-prompt-engineering/) | 5 prompt variants, CoT, tool calling, A/B test | Prompting |
| [week4-chatbot](./llm-core/week4-chatbot/) | FastAPI + Claude + conversation memory + Streamlit | Chatbot |
| [week5-prompt-chaining](./llm-core/week5-prompt-chaining/) | Multi-step chains, function calling, retry logic | Chaining |
| [week6-embeddings](./llm-core/week6-embeddings/) | Embedding generation, cosine search, FastAPI /search | Embeddings |
| [week7-nlp-task](./llm-core/week7-nlp-task/) | Text classification + NER with Hugging Face | NLP, transformers |
| [week8-evaluation](./llm-core/week8-evaluation/) | BLEU, ROUGE, latency p95/p99, evaluation dashboard | LLM eval |

---

## Phase 4: RAG + Agents

> **Job Skills**: Retrieval-Augmented Generation, LangChain, LangGraph, vector databases, agentic systems, tool use

> LangChain · pgvector · Supabase · ReAct · LangGraph · CrewAI

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-simple-rag](./rag-agents/week1-simple-rag/) | Document loader → embed → pgvector → LLM answer | RAG basics |
| [week2-advanced-rag](./rag-agents/week2-advanced-rag/) | Hybrid retrieval, reranking, guardrails, memory | Advanced RAG |
| [week3-rag-evaluation](./rag-agents/week3-rag-evaluation/) | Ragas framework, NDCG, MRR, A/B chunking strategies | RAG eval |
| [week4-langchain](./rag-agents/week4-langchain/) | Chains, memory, document loaders, streaming | LangChain |
| [week5-simple-agent](./rag-agents/week5-simple-agent/) | ReAct agent, tool definitions, error recovery | Agents |
| [week6-advanced-agent](./rag-agents/week6-advanced-agent/) | Multi-tool agent, planning, streaming, LangGraph | LangGraph |
| [week7-rag-agent](./rag-agents/week7-rag-agent/) | Agent uses RAG as tool, multi-step reasoning | RAG + agents |
| [week8-complete-system](./rag-agents/week8-complete-system/) | Full RAG + agent + monitoring + guardrails + Docker | Production |

---

## Phase 5: MLOps + Cloud ML

> **Job Skills**: MLflow, CI/CD for ML, drift detection, Prometheus/Grafana, automated retraining, AWS SageMaker, GCP Vertex AI, Azure ML, serverless ML, Infrastructure as Code

> Prometheus · Grafana · MLflow · GitHub Actions · Kubernetes · SageMaker · Vertex AI

This phase has been expanded from 8 to 10 weeks to cover cloud ML platforms and serverless — both explicitly listed in 85-90% of senior AI engineer job descriptions.

### Core MLOps (Weeks 1–8)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-monitoring](./mlops/week1-monitoring/) | Prometheus metrics, Grafana dashboards, alerting rules | Observability |
| [week2-llm-monitoring](./mlops/week2-llm-monitoring/) | Token usage, cost tracking, prompt/response audit logs | LLMOps |
| [week3-mlflow](./mlops/week3-mlflow/) | Experiment tracking, model registry, staging/prod promotion | MLflow |
| [week4-drift-detection](./mlops/week4-drift-detection/) | Data drift, model drift, automated retraining triggers | Drift, Evidently |
| [week5-cicd](./mlops/week5-cicd/) | GitHub Actions: test → validate → build → deploy → rollback | CI/CD |
| [week6-retraining](./mlops/week6-retraining/) | Scheduled retraining, data validation (Great Expectations), auto-promotion | Automation |
| [week7-ab-testing](./mlops/week7-ab-testing/) | Traffic splitting, statistical significance test, automated model promotion | A/B testing |
| [week8-full-lifecycle](./mlops/week8-full-lifecycle/) | Data → train → track → deploy → monitor → retrain (full loop) | Full lifecycle |

### Cloud ML Platforms (Weeks 9–10)

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week9-cloud-ml-platforms](./mlops/week9-cloud-ml-platforms/) | Train + deploy ML model on AWS SageMaker, replicate on GCP Vertex AI, compare workflows; use Azure ML for experiment tracking | SageMaker, Vertex AI, Azure ML |
| [week10-serverless-ml](./mlops/week10-serverless-ml/) | Package model as AWS Lambda function, GCP Cloud Function, and containerized serverless endpoint; Terraform for infrastructure as code | Lambda, Cloud Functions, Terraform |

**Why cloud ML platforms**: 88% of job postings require AWS, Azure, or GCP experience specifically for ML workloads — not just general cloud. SageMaker and Vertex AI are the two most requested platforms.

**Week 9 concepts**:
- AWS SageMaker: training jobs, built-in algorithms, SageMaker Pipelines, model registry, endpoints
- GCP Vertex AI: AutoML, custom training, Vertex Pipelines, model deployment
- Azure ML: compute clusters, AutoML, pipelines, deployment
- Managed feature stores: SageMaker Feature Store, Vertex Feature Store
- Cloud model monitoring and drift detection (built-in services)

**Week 10 concepts**:
- Serverless inference with AWS Lambda + API Gateway
- GCP Cloud Functions for ML inference
- Container-based serverless: AWS Fargate, GCP Cloud Run
- Cold start optimization for ML models
- Infrastructure as Code with Terraform (ML infra)
- CloudFormation for SageMaker stack

---

## Phase 6: Enterprise AI Systems

> **Job Skills**: Microservices architecture, Kubernetes, Redis, Celery, OpenTelemetry, system design, production reliability, cross-functional collaboration

> Microservices · Kubernetes · Redis · Celery · OpenTelemetry

| Project | Description | Key Skills |
|---------|-------------|------------|
| [architecture-design](./enterprise-ai/architecture-design/) | ADRs, latency budgets, cost models, failure mode analysis | System design |
| [service-ingestion](./enterprise-ai/service-ingestion/) | Multi-source ingestion, schema validation, dead letter queue | Data pipeline |
| [service-embedding](./enterprise-ai/service-embedding/) | Batch embedding, caching, model management, scaling | Embeddings at scale |
| [service-rag](./enterprise-ai/service-rag/) | Query → retrieve → rank → generate, conversation memory | RAG service |
| [service-agent](./enterprise-ai/service-agent/) | Tool orchestration, multi-step planning, error recovery | Agent service |
| [service-gateway](./enterprise-ai/service-gateway/) | Rate limiting, auth (JWT), load balancing, logging | API gateway |
| [infrastructure](./enterprise-ai/infrastructure/) | PostgreSQL + pgvector, Redis, Celery, docker-compose | Infrastructure |
| [deployment](./enterprise-ai/deployment/) | Kubernetes manifests, HPA, health probes, secrets mgmt | K8s |
| [monitoring](./enterprise-ai/monitoring/) | Prometheus + Grafana + OpenTelemetry + alerting | Observability |
| [documentation](./enterprise-ai/documentation/) | OpenAPI specs, runbooks, architecture decision records | Docs |

---

## Job Requirements Coverage Matrix

Every cell maps a real job posting requirement to the phase where it is covered:

| Job Requirement | Where Covered | Projects |
|----------------|---------------|----------|
| Python (production-grade) | Phase 0 | week3-oop, week4-fastapi |
| Data preprocessing & EDA | Phase 0 | week2-pandas |
| Probability & statistics | Phase 1 | week1-probability-stats |
| Statistical modeling | Phase 1 | week3-statistical-modeling |
| Hypothesis testing | Phase 1 | week2-hypothesis-testing |
| A/B testing & experiment design | Phase 1, Phase 5 | week4-ab-testing, week7-ab-testing |
| ML algorithms from scratch | Phase 2 | week1-scratch-algorithms |
| Feature engineering | Phase 2 | week3-feature-engineering |
| scikit-learn (classification, regression, clustering) | Phase 2 | week1–week3, week9 |
| PyTorch (deep learning, neural networks) | Phase 2 | week4-neural-networks |
| TensorFlow / Keras | Phase 2 | week7-tensorflow-keras |
| CNN / Computer Vision | Phase 2 | week6-cnn-cv |
| XGBoost / LightGBM (ensemble methods) | Phase 2 | week5-ensembles |
| Model explainability (SHAP, LIME) | Phase 2 | week8-explainability |
| NLP foundations (SpaCy, NLTK) | Phase 2 | week10-nlp-foundations |
| Time series analysis (ARIMA, Prophet, LSTM) | Phase 2 | week11-time-series |
| Recommendation systems | Phase 2 | week12-recommendations |
| Unsupervised learning (clustering, PCA) | Phase 2 | week9-unsupervised |
| LLMs (OpenAI, Anthropic, Hugging Face) | Phase 3 | week1–week8 |
| Fine-tuning (LoRA, PEFT) | Phase 3 | week2-fine-tuning |
| Embeddings & vector search | Phase 3, Phase 4 | week6-embeddings, week1-simple-rag |
| RAG (Retrieval-Augmented Generation) | Phase 4 | week1–week8 |
| LangChain / LangGraph | Phase 4 | week4-langchain, week6-advanced-agent |
| AI agents with tool use | Phase 4 | week5–week8 |
| REST API development | Phase 0, all phases | week4-fastapi + every service |
| Microservices architecture | Phase 6 | all services |
| Docker | Phase 0, all phases | week4-fastapi + all projects |
| Kubernetes | Phase 5, Phase 6 | week5-cicd, deployment |
| Cloud-native (AWS/Azure/GCP) | Phase 5 | week9-cloud-ml-platforms |
| AWS SageMaker | Phase 5 | week9-cloud-ml-platforms |
| GCP Vertex AI | Phase 5 | week9-cloud-ml-platforms |
| Serverless (Lambda, Cloud Functions) | Phase 5 | week10-serverless-ml |
| Infrastructure as Code (Terraform) | Phase 5 | week10-serverless-ml |
| MLflow (experiment tracking, model registry) | Phase 5 | week3-mlflow |
| CI/CD for ML (GitHub Actions) | Phase 5 | week5-cicd |
| Data drift + model drift detection | Phase 5 | week4-drift-detection |
| Automated retraining pipelines | Phase 5 | week6-retraining |
| Prometheus + Grafana monitoring | Phase 5, Phase 6 | week1-monitoring, monitoring |
| End-to-end ML lifecycle | Phase 5 | week8-full-lifecycle |
| Redis (caching) | Phase 6 | infrastructure |
| Celery (task queues) | Phase 6 | infrastructure |
| OpenTelemetry (distributed tracing) | Phase 6 | monitoring |
| Rate limiting, JWT auth | Phase 6 | service-gateway |
| System design & architecture | Phase 6 | architecture-design |

---

## Tech Stack

```
ML/DL:        scikit-learn · PyTorch · TensorFlow/Keras · XGBoost · LightGBM
Explainability: SHAP · LIME · interpretML
NLP:          SpaCy · NLTK · Hugging Face Transformers
Time Series:  statsmodels (ARIMA/SARIMA) · Prophet · PyTorch LSTM
Statistics:   SciPy · statsmodels · Pingouin
LLMs:         OpenAI GPT-4o · Anthropic Claude · Llama 3 (Hugging Face)
Frameworks:   LangChain · LangGraph · FastAPI · Streamlit
Databases:    PostgreSQL · pgvector · Redis · Supabase
Cloud ML:     AWS SageMaker · GCP Vertex AI · Azure ML
Serverless:   AWS Lambda · GCP Cloud Functions · AWS Fargate
IaC:          Terraform · CloudFormation
MLOps:        MLflow · Evidently · Great Expectations · Prometheus · Grafana
CI/CD:        GitHub Actions · Docker · Kubernetes · Celery
Languages:    Python · SQL
```

---

## Key Metrics (from projects)

- RAG system: **< 500ms p95 latency**, hybrid BM25 + semantic retrieval
- Fine-tuned model: **+23% ROUGE-L** vs baseline, cost $0.40 to train
- ML pipeline: **automated retraining** on drift detection, zero-downtime deploys
- LLM evaluation: **BLEU, ROUGE, exact match** across model A/B comparisons
- Statistical A/B test: power = 0.80, α = 0.05, proper sample size calculation
- Time series forecast: **MAPE < 8%** on demand forecasting with Prophet + LSTM ensemble
- Recommendation system: **precision@10 = 0.34**, serving via FastAPI at < 50ms

---

## Contact

- Email: fk8767@gmail.com
- LinkedIn: [faisal-khan-13b146148](https://linkedin.com/in/faisal-khan-13b146148)
- Twitter: [@itzzfaisalkhan](https://twitter.com/itzzfaisalkhan)
