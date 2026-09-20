# Training Projects

Projects should grow with the roadmap. The purpose is not to fill GitHub with tutorial clones; it is to force knowledge to become practical.

## Project ladder

### Level 1 — Python / data exercises

Small programs written largely from scratch. Focus on clean functions, data structures, files, exceptions, tests and readable code.

### Level 2 — Classical ML

Take a real or safely public dataset through the complete workflow:

problem definition → exploration → preprocessing → baseline → training → evaluation → error analysis → short explanation of results.

At least one project should be written cleanly enough to run outside a notebook.

### Level 3 — Deep Learning

Train a small PyTorch model with an explicit training/evaluation loop. Be able to explain tensors, loss, gradients, optimization and validation rather than simply running somebody else's notebook.

### Level 4 — LLM systems

Build increasingly serious systems:

- embeddings/semantic-search experiment;
- RAG from raw documents through chunking, retrieval and cited answers;
- retrieval evaluation rather than judging quality by feel;
- structured-output/tool-calling application;
- agent with bounded tools and observable steps;
- fine-tuning/LoRA experiment when it solves a real problem.

### Level 5 — Production

Turn one strong AI project into a service:

API + persistence + Docker + tests + configuration + logging + evaluation/monitoring + deployment documentation.

## Telecom + AI portfolio direction

This is where the portfolio can become distinctive.

Ideas currently worth preserving:

**Telecom standards RAG** — search RFC/3GPP/vendor documentation and produce engineering answers with traceable citations.

**SIP RCA assistant** — parse or ingest sanitized SIP traces, reconstruct call flows, identify evidence, and have an LLM explain likely failure points without inventing missing evidence.

**Telecom incident investigator** — combine KPI/anomaly information, logs/traces and engineering knowledge to assemble an evidence-backed incident summary.

**Protocol-data pipeline** — PCAP/log → Python parsing → structured representation/database → deterministic analysis → LLM explanation. Keep deterministic protocol facts separate from probabilistic LLM interpretation.

These are ideas, not commitments. We will choose projects when the prerequisite knowledge exists.

## Project quality rules

A portfolio project should eventually answer:

- What problem does it solve?
- Why was this architecture chosen?
- What are the failure modes?
- How is quality measured?
- What did I personally implement and understand?
- How is it tested?
- How would it behave with bad/unknown input?
- How is it deployed and monitored?
- What would I change at larger scale?

And for telecom work: **never publish customer-sensitive traces, identifiers, credentials, proprietary datasets, or unsanitized production information.**
