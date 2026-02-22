<h1 align="center">Faisal Khan — AI Engineer</h1>

<p align="center">
  <strong>Software Engineer → AI Engineer</strong><br/>
  Building production-grade AI systems: RAG pipelines, LLM agents, MLOps infrastructure
</p>

<p align="center">
  <a href="https://linkedin.com/in/faisal-khan-13b146148"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
  <a href="https://twitter.com/itzzfaisalkhan"><img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"/></a>
  <a href="mailto:fk8767@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
</p>

---

## About

SWE background with deep experience in testing, CI/CD, and distributed systems — now specializing in AI engineering. I build AI systems that actually ship: production RAG pipelines, LLM agents with tool use, and MLOps infrastructure that monitors and retrains models automatically.

**Unfair advantage**: Most AI engineers can't write production-grade code. I can.

---

## AI Engineering Roadmap — 12 Months, 36+ Projects

This repo documents my structured path from SWE to AI Engineer across 5 phases:

| Phase | Focus | Duration | Projects |
|-------|-------|----------|---------|
| [Phase 0](#phase-0-python-for-ai) | Python for AI | 4 weeks | 4 repos |
| [Phase 1](#phase-1-ml-foundations) | ML Foundations | 8 weeks | 8 repos |
| [Phase 2](#phase-2-llm-core) | LLM Core | 8 weeks | 8 repos |
| [Phase 3](#phase-3-rag--agents) | RAG + Agents | 8 weeks | 8 repos |
| [Phase 4](#phase-4-mlops--llmops) | MLOps/LLMOps | 8 weeks | 8 repos |
| [Phase 5](#phase-5-enterprise-ai-systems) | Enterprise AI | 16 weeks | 10+ repos |

---

## Phase 0: Python for AI

> NumPy · Pandas · OOP · FastAPI · Docker · pytest

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-numpy](./python-foundations/week1-numpy/) | Vector ops, broadcasting, linear algebra, SVD | NumPy, vectorization |
| [week2-pandas](./python-foundations/week2-pandas/) | Data cleaning, feature engineering, EDA | Pandas, matplotlib |
| [week3-oop](./python-foundations/week3-oop/) | Decorators, generators, async/await, context managers | Advanced Python |
| [week4-fastapi](./python-foundations/week4-fastapi/) | REST API, Docker, pytest, prediction endpoint | FastAPI, testing |

---

## Phase 1: ML Foundations

> scikit-learn · PyTorch · XGBoost · sklearn Pipelines

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-scratch-algorithms](./ml-fundamentals/week1-scratch-algorithms/) | Linear regression, logistic regression, decision tree from scratch | Math, backprop |
| [week2-classification](./ml-fundamentals/week2-classification/) | Fraud detection: 5 models, AUC-ROC comparison | sklearn, evaluation |
| [week3-feature-engineering](./ml-fundamentals/week3-feature-engineering/) | SMOTE, GridSearch, feature selection, cross-validation | Feature eng |
| [week4-neural-networks](./ml-fundamentals/week4-neural-networks/) | NN from scratch → PyTorch, MNIST 99%+ | PyTorch, backprop |
| [week5-cnn](./ml-fundamentals/week5-cnn/) | CNN for CIFAR-10, filter visualization | PyTorch, CNN |
| [week6-unsupervised](./ml-fundamentals/week6-unsupervised/) | K-Means, hierarchical, PCA, customer segmentation | Unsupervised ML |
| [week7-ensembles](./ml-fundamentals/week7-ensembles/) | XGBoost, LightGBM, stacking, Kaggle competition | Ensemble methods |
| [week8-production-pipeline](./ml-fundamentals/week8-production-pipeline/) | sklearn Pipeline → FastAPI → Docker | Production ML |

---

## Phase 2: LLM Core

> OpenAI · Anthropic Claude · Hugging Face · LangChain · Embeddings

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-llm-fundamentals](./llm-core/week1-llm-fundamentals/) | Multi-provider API wrapper, cost + latency tracker | LLM APIs |
| [week2-fine-tuning](./llm-core/week2-fine-tuning/) | LoRA fine-tuning, BLEU/ROUGE eval, cost analysis | Fine-tuning |
| [week3-prompt-engineering](./llm-core/week3-prompt-engineering/) | 5 prompt variants, CoT, tool calling, A/B test | Prompting |
| [week4-chatbot](./llm-core/week4-chatbot/) | FastAPI + Claude + conversation memory + Streamlit | Chatbot |
| [week5-prompt-chaining](./llm-core/week5-prompt-chaining/) | Multi-step chains, function calling, retry logic | Chaining |
| [week6-embeddings](./llm-core/week6-embeddings/) | Embedding generation, cosine search, FastAPI /search | Embeddings |
| [week7-nlp-task](./llm-core/week7-nlp-task/) | Text classification + NER with Hugging Face | NLP |
| [week8-evaluation](./llm-core/week8-evaluation/) | BLEU, ROUGE, latency p95/p99, evaluation dashboard | Evaluation |

---

## Phase 3: RAG + Agents

> LangChain · pgvector · Supabase · ReAct · LangGraph

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-simple-rag](./rag-agents/week1-simple-rag/) | Document loader → embed → pgvector → LLM answer | RAG basics |
| [week2-advanced-rag](./rag-agents/week2-advanced-rag/) | Hybrid retrieval, reranking, guardrails, memory | Advanced RAG |
| [week3-rag-evaluation](./rag-agents/week3-rag-evaluation/) | Ragas framework, NDCG, MRR, A/B chunking strategies | RAG eval |
| [week4-langchain](./rag-agents/week4-langchain/) | Chains, memory, document loaders, streaming | LangChain |
| [week5-simple-agent](./rag-agents/week5-simple-agent/) | ReAct agent, tool definitions, error recovery | Agents |
| [week6-advanced-agent](./rag-agents/week6-advanced-agent/) | Multi-tool agent, planning, streaming, LangGraph | Advanced agents |
| [week7-rag-agent](./rag-agents/week7-rag-agent/) | Agent uses RAG as tool, multi-step reasoning | RAG + agents |
| [week8-complete-system](./rag-agents/week8-complete-system/) | Full RAG + agent + monitoring + guardrails + Docker | Production |

---

## Phase 4: MLOps / LLMOps

> Prometheus · Grafana · MLflow · GitHub Actions · Kubernetes

| Project | Description | Key Skills |
|---------|-------------|------------|
| [week1-monitoring](./mlops/week1-monitoring/) | Prometheus metrics, Grafana dashboards, alerting | Observability |
| [week2-llm-monitoring](./mlops/week2-llm-monitoring/) | Token usage, cost tracking, prompt/response audit logs | LLM ops |
| [week3-mlflow](./mlops/week3-mlflow/) | Experiment tracking, model registry, staging/prod | MLflow |
| [week4-drift-detection](./mlops/week4-drift-detection/) | Data drift, model drift, automated retraining triggers | Drift |
| [week5-cicd](./mlops/week5-cicd/) | GitHub Actions: test → validate → build → deploy → rollback | CI/CD |
| [week6-retraining](./mlops/week6-retraining/) | Scheduled retraining, data validation, auto-promotion | Retraining |
| [week7-ab-testing](./mlops/week7-ab-testing/) | Traffic splitting, t-test, automated model promotion | A/B testing |
| [week8-full-lifecycle](./mlops/week8-full-lifecycle/) | Data → train → track → deploy → monitor → retrain | Full lifecycle |

---

## Phase 5: Enterprise AI Systems

> Microservices · Kubernetes · Redis · Celery · OpenTelemetry

| Project | Description | Key Skills |
|---------|-------------|------------|
| [architecture-design](./enterprise-ai/architecture-design/) | ADRs, latency budgets, cost models, failure mode analysis | System design |
| [service-ingestion](./enterprise-ai/service-ingestion/) | Multi-source ingestion, schema validation, dead letter queue | Data pipeline |
| [service-embedding](./enterprise-ai/service-embedding/) | Batch embedding, caching, model management, scaling | Embeddings |
| [service-rag](./enterprise-ai/service-rag/) | Query → retrieve → rank → generate, conversation memory | RAG service |
| [service-agent](./enterprise-ai/service-agent/) | Tool orchestration, multi-step planning, error recovery | Agent service |
| [service-gateway](./enterprise-ai/service-gateway/) | Rate limiting, auth (JWT), load balancing, logging | API gateway |
| [infrastructure](./enterprise-ai/infrastructure/) | PostgreSQL + pgvector, Redis, Celery, docker-compose | Infrastructure |
| [deployment](./enterprise-ai/deployment/) | Kubernetes manifests, HPA, health probes, secrets mgmt | K8s |
| [monitoring](./enterprise-ai/monitoring/) | Prometheus + Grafana + OpenTelemetry + alerting | Observability |
| [documentation](./enterprise-ai/documentation/) | OpenAPI specs, runbooks, architecture decision records | Docs |

---

## Tech Stack

```
LLMs:         OpenAI GPT-4o · Anthropic Claude · Llama 3 (Hugging Face)
Frameworks:   LangChain · LangGraph · FastAPI · Streamlit
Databases:    PostgreSQL · pgvector · Supabase · Redis
MLOps:        MLflow · Prometheus · Grafana · GitHub Actions
Infra:        Docker · Kubernetes · Celery · AWS
Languages:    Python · SQL
```

---

## Key Metrics (from projects)

- RAG system: **< 500ms p95 latency**, hybrid BM25 + semantic retrieval
- Fine-tuned model: **+23% ROUGE-L** vs baseline, cost $0.40 to train
- ML pipeline: **automated retraining** on drift detection, zero-downtime deploys
- LLM evaluation: **BLEU, ROUGE, exact match** across model A/B comparisons

---

## Contact

- Email: fk8767@gmail.com
- LinkedIn: [faisal-khan-13b146148](https://linkedin.com/in/faisal-khan-13b146148)
- Twitter: [@itzzfaisalkhan](https://twitter.com/itzzfaisalkhan)
