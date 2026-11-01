# Roadmap

This is a direction, not a rigid calendar. I would rather spend time understanding a model properly than advance because a course progress bar reached 100%.

**Career target:** ML / LLM engineer capable of designing, building, evaluating, debugging and operating real AI systems, with a differentiated Telecom / Voice / SIP / IMS / observability specialization.

**Adaptive training is mandatory:** See [Adaptive Training Contract](knowledge/ADAPTIVE_ML_LLM_TRAINING_CONTRACT.md). Continuously evaluate genuine student progress, actual questions, repeated friction, code and testing results, and interests. Adjust proposed course order with evidence and student approval. Preserve the [Golden Guided Coding Rule](knowledge/GUIDED_CODING_GOLDEN_RULE.md).

## Phase 1 — Python foundation ✅

Completed Jose Portilla's Complete Python Bootcamp in September 2026. Core Python fundamentals are established, but fluency with numerical APIs and tensor operations continues through practical ML exercises. **Do not prescribe another general Python bootcamp by default.**

## Phase 2 — Machine Learning foundations ✅ (documented study milestone)

Andrew Ng / DeepLearning.AI Machine Learning Specialization, Courses 1–3 studied through regression/classification, neural networks, trees/ensembles, K-means, anomaly detection, collaborative/content-based recommendation, PCA and Lunar Lander DQN. The [Complete Specialization Booklet](knowledge/MACHINE_LEARNING_SPECIALIZATION_COMPLETE_BOOKLET.md) reconciles these topics. Recorded public tests and DQN training are documented in the individual notes. The repository does **not independently verify all hidden graders or external certificate-award status**.

### What we learned about the learner

Strengths: conceptual understanding improves quickly after concrete dataflow examples; student-authored graded code and passing public tests; strong production systems reasoning and telecom experience.

Targeted gaps: input/target/reward provenance, tensor dimensions/axes and tuples, translating formulas to vectorized code, understanding the complete forward/loss/backprop/optimizer chain, and distinction between training/inference and database/model state.

Respond with focused exercises, shape tracing and numerical examples, **not a restart of Python fundamentals**. See [Adaptive Training Contract](knowledge/ADAPTIVE_ML_LLM_TRAINING_CONTRACT.md).

## Phase 3A — Deep Learning Specialization (NEXT)

**Agreed 2026-11-01:** Take Andrew Ng's [Deep Learning Specialization](https://www.deeplearning.ai/specializations/deep-learning) before the PyTorch certificate. This refines the previous roadmap instead of replacing the destination.

Study all five courses in their sequence, prioritizing the difficult connections: neural-network dataflow, forward/back propagation, gradient checking, optimization, regularization, error analysis, sequence models, attention and Transformers. Courses 1, 2 and 5 have particularly high relevance; do not omit the other courses without new evidence.

**Hands-on competency target:** Independently build a tiny model, trace actual feature and label origins, explain batch/layer shapes, calculate and explain a forward prediction and loss, describe gradients and Adam updates, evaluate with proper splits, then save/reload and predict on new input. Look up API syntax as needed. This is a readiness target, not an externally imposed gate or graded-solution request.

Use targeted mathematics refreshers only when actual bottlenecks occur, rather than mandatory repetition of an entire prior math specialization.

## Phase 3B — PyTorch for Deep Learning Professional Certificate

After deep-learning concepts are better internalized, work through [PyTorch for Deep Learning](https://www.deeplearning.ai/specializations/pytorch-for-deep-learning-professional-certificate). Focus on tensors and autograd, Dataset/DataLoader, custom training loops, pretrained models, evaluation, inference, saving and deployment.

Suggested proof: rebuild one familiar previous model in PyTorch, compare results and evaluation criteria, explain tensor shapes and parameter updates, and package a small reproducible application.

## Phase 4 — Transformers, NLP and LLM Engineering

Study tokenization, embeddings, self-attention, Transformers, language model inference, retrieval/RAG, fine-tuning, tool calling, agents and LLM-specific evaluation. Use [Natural Language Processing Specialization](https://www.deeplearning.ai/specializations/natural-language-processing/) modules selectively if they serve current gaps; do not automatically add overlapping certificates.

Build a sourced document-question-answering system with measured retrieval and answer quality rather than merely a UI wrapping an LLM API.

## Phase 5 — Production AI / MLOps / LLMOps

Focus on APIs, data pipelines, model and data versioning, reproducibility, monitoring, evaluation, drift, cost, latency, fallback behavior, testing and safety. Add focused production training only when relevant to the project and current gaps.

**Portfolio capstone:** A privacy-safe Telecom Incident Intelligence system using synthetic or properly sanitized data: KPI anomaly detection/classification, retrieval from approved troubleshooting documents and read-only analytical tools, with measured false positives, grounded answer quality, latency/cost and operational failure containment. No production HOMER access or customer data without explicit approval.

## Phase 6 — Portfolio and domain specialization

Use ML and LLM engineering together with Voice / IMS / SIP / SS7 / Diameter and observability skills. Publish reproducible projects, honest evaluations, architecture tradeoffs and test evidence rather than collecting certificates alone.

## Certification philosophy

Certificates document learning progress. Job readiness depends on independently explaining dataflow, building practical software, selecting sound metrics, diagnosing mistakes and evaluating real performance. Course order is **adaptive**, not fixed: revisit at major checkpoints or repeated difficulties. Recommendations must distinguish observed competency from inference and user-selected interests.

## Current next action

**Start Deep Learning Specialization, Course 1: Neural Networks and Deep Learning.** Keep existing student-first guided coding procedure. When a lab passes, conduct a post-test numerical/data-shape walkthrough on request and update the canonical cheat sheet and knowledge records selectively.

## Long-term destination

Production-capable ML/LLM engineer with distinctive expertise in Voice/Telecom systems. Depth, correctness and engineering judgment take priority over speed or the number of credentials.
