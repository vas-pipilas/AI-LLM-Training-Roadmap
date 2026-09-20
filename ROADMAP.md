# Roadmap

This is a direction, not a rigid calendar. I would rather spend another week understanding something properly than mark a phase complete because a course progress bar reached 100%.

## Phase 1 — Python foundation ✅

**Status:** Complete (September 2026)

Completed Jose Portilla's *Complete Python Bootcamp: From Zero to Hero in Python*.

The purpose was to become comfortable enough with Python that AI code is not a black box: functions, data structures, loops and conditionals, modules, files, exceptions, OOP, decorators, testing, and general program structure.

**Decision:** do not take another generic Python course. Python development continues through practice and later ML/LLM work.

---

## Phase 2 — Machine Learning foundations 🔄

**Status:** Current

Primary learning path: **Andrew Ng / DeepLearning.AI Machine Learning Specialization**.

What I want from this phase is not just knowing how to call scikit-learn. I want the intuition behind what a model is doing and why.

Core areas include:

- supervised vs. unsupervised learning;
- regression and classification;
- loss/cost functions and gradient descent;
- model training and evaluation;
- overfitting, underfitting, bias and variance;
- feature engineering and preprocessing;
- decision trees and ensemble methods;
- clustering and anomaly detection;
- recommender-system fundamentals;
- practical NumPy, pandas, scikit-learn and notebook work.

Python practice runs in parallel. Whenever possible, implement small pieces myself before hiding them behind libraries.

**Exit condition:** I can take a dataset, frame an ML problem, prepare the data, choose a reasonable baseline, train/evaluate it, diagnose obvious problems, and explain the result.

---

## Phase 3 — Deep Learning and PyTorch

Learn neural networks deeply enough that transformers later make sense instead of appearing as magic.

Topics:

- tensors and PyTorch;
- forward propagation and backpropagation;
- activation and loss functions;
- optimization;
- training loops, validation and regularization;
- embeddings;
- sequence-model background where useful;
- attention;
- transformer foundations.

The practical framework target is **PyTorch**, even if individual courses expose TensorFlow as well.

**Exit condition:** build and train neural networks in PyTorch and understand the mechanics of their training well enough to debug them.

---

## Phase 4 — LLM Engineering

This is where the eventual specialization becomes explicit.

Topics will include:

- transformer architecture in more depth;
- tokenization and context windows;
- embeddings and semantic search;
- Hugging Face ecosystem;
- model APIs and local/open models where useful;
- prompting as an engineering tool rather than a substitute for architecture;
- structured output and tool/function calling;
- vector databases;
- RAG architecture;
- reranking and retrieval quality;
- agents and multi-step tool use;
- evaluation;
- fine-tuning, LoRA and PEFT;
- inference, quantization and serving fundamentals;
- safety, security, privacy and guardrails.

A key rule here is to understand the layers underneath frameworks. LangChain/LlamaIndex/etc. may be useful tools, but I do not want to become dependent on abstractions I cannot explain.

---

## Phase 5 — Production AI / MLOps / LLMOps

Knowing how to make a notebook work is different from engineering a system.

Build competence in:

- FastAPI and service/API design;
- PostgreSQL and appropriate data stores;
- Docker;
- Linux;
- Git/GitHub and good engineering workflow;
- testing;
- CI/CD;
- cloud fundamentals;
- model and application deployment;
- observability and monitoring;
- model/data drift concepts;
- evaluation pipelines;
- secrets and configuration;
- reliability, cost and performance;
- MLOps/LLMOps practices.

**Exit condition:** deploy an AI service that another person could actually use and operate, with tests, logs, configuration, monitoring and a reproducible deployment.

---

## Phase 6 — Portfolio and domain specialization

My Voice/Telecom background should become a differentiator.

Possible serious portfolio directions:

- telecom engineering RAG over RFC/3GPP/vendor documentation with citations;
- SIP call-flow reconstruction and RCA assistant;
- PCAP/log → structured events → analysis → LLM explanation pipeline;
- anomaly/incident analysis over telecom KPI data;
- an agent that gathers evidence from multiple engineering sources before proposing an RCA;
- AI functionality around the kind of observability/analytics systems I already understand operationally.

Projects must avoid real customer-sensitive data. Use synthetic, sanitized, public, or safely generated datasets.

The aim is to demonstrate both sides of the profile: **AI/LLM engineering + deep telecom domain knowledge**.

---

## Certification philosophy

Certifications are checkpoints, not the curriculum.

The order should normally be:

**learn → practice → build → gain cloud/tool experience → certify**

Current certification candidates are tracked in [CERTIFICATIONS.md](CERTIFICATIONS.md). We should re-check the market and exam versions when I approach each certification instead of blindly following a list written months earlier.

---

## Long-term destination

The target is competence for **AI/LLM Engineering roles**, particularly roles involving production systems, RAG/agents, AI platforms, MLOps/LLMOps, or technically demanding domains.

I should eventually be able to say, and demonstrate with code:

> I can understand the problem, choose an appropriate AI approach, build the system, evaluate it, debug it, deploy it, and operate it.
