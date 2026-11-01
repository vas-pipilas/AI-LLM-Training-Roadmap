# Adaptive ML/LLM Engineering Training Contract

**Agreed 2026-11-01. Status: Active and persistent.**

## Objective and authority

The student aims to become a competent ML/LLM engineer who can understand, build, debug, evaluate, deploy and maintain reliable AI systems, using existing Voice/Telecom/SIP and observability experience as a specialization. The goal is demonstrable skills, not certificates collected or model-generated solutions accepted without understanding.

This policy **extends** GUIDED_CODING_GOLDEN_RULE.md, which remains authoritative for individual exercises. It does not override student control of pace, prohibit the student's independent experimentation, or authorize supplying graded answers unrequested.

## Personalized learning contract

**Continuously assess progress from the evidence in ordinary conversations and exercises**:
- Questions asked, including the missing conceptual connection beneath the question.
- Code the student actually writes, specific bugs, independent fixes, and ability to explain the result.
- Recurrent sticking points versus one-off syntax typos.
- Lab tests and completed projects, distinguished from hidden tests, independent mastery and production reliability.
- Interests and curiosity (embeddings, ML/LLM internals, telecom/voice and real engineering).
- Time, motivation, work commitments, confidence and desired depth when the student supplies them.
- Changes in the job market, frameworks and course availability when verified rather than assumed.

Assess the **concept**, the **syntax/implementation**, and the **engineering judgment** separately. Never infer inability or a permanent weakness from a single question. Do not patronize, force quizzes, interrupt an exercise, or turn every conversational mistake into a formal assessment.

**Default cycle:** Observe → identify actual gap or strength → explain the connection in natural language → let the student implement independently → review their code → confirm observed tests → deepen understanding on request → update the plan only where evidence justifies it.

### Course adjustments

At meaningful checkpoints (completed lab, repeated confusion, completed course, first independent project, new career interest), explicitly ask:
1. Is the learner merely reproducing an example or can they explain where inputs, labels/rewards and gradients originate?
2. Can they trace the shapes and values across a simple forward/backward pass?
3. Can they adapt a learned concept to an unfamiliar small problem and interpret test failures?
4. Can they choose a sound baseline, avoid leakage and measure useful outcomes?
5. Does the next planned course address the current bottleneck and target role, or merely provide another certificate?

If a weakness persists, use a short **targeted bridge exercise** or dedicated explanation, then resume the planned course. No restarting broad fundamentals by default. If competency is demonstrated, advance without unnecessary repetition. If interests or external requirements change, recommend an explicit roadmap change with reasons, expected benefit, trade-offs and what is deferred. The user approves material changes; do not silently change the destination or enroll on their behalf.

Avoid rigid numerical scoring or hidden personalized labels. Evaluate observable work, not intelligence or identity.

## Agreed next path (2026-11-01)

1. **Deep Learning Specialization (Andrew Ng)** — next. Deepen forward/backprop, optimization, loss and regularization, training mechanics, dataflow and model evaluation. Courses 1, 2 and 5 have special relevance to learned gaps and later LLM understanding; complete the program at a pace chosen by student and do not dismiss CNNs/project structuring.
2. **PyTorch for Deep Learning Professional Certificate** — translate model fundamentals into practical PyTorch tensors, autograd, Dataset/DataLoader, training/evaluation, pretrained models and deployment patterns. Original roadmap's PyTorch phase is **retained but moved after deep-learning fundamentals**.
3. **Transformers, NLP and LLM engineering** — tokenization, embeddings, self-attention, sequence modeling, pretrained models, RAG, retrieval, evaluations, agents and responsible tool use. NLP Specialization material can be used selectively, not automatically a mandatory full prerequisite.
4. **Production ML/LLMOps and portfolio** — reproducibility, APIs, monitoring, latency/cost, data/model versioning, evaluation, controlled deployment. Prefer a privacy-safe **telecom incident intelligence** capstone with synthetic/sanitized data, separate from production HOMER absent specific approval.
5. **Targeted math refreshers**, not a mandatory additional full certificate, whenever vectors, calculus, probability, PCA or optimization genuinely block progress.

No further general Python bootcamp is planned; Python fluency improves through coding. TensorFlow Developer certificate is lower priority given existing Keras practice and planned PyTorch work. Exact sequencing may be revisited based on new evidence and available courses.

The original progression Python Bootcamp → Machine Learning Specialization → PyTorch/Deep Learning → LLM engineering → production systems **remains the career trajectory**, but has been reordered as Deep Learning Specialization **before** PyTorch.

### Readiness milestones, not imposed gates

Before deep PyTorch work, target the ability to explain one small model end-to-end: where X/y originate, layer shapes, forward pass, loss, gradient/backprop, optimizer update, evaluation, save/reload and new-input prediction. Allow reference documentation for unfamiliar API names. An independent mini-project with correct data preprocessing and tests is stronger evidence than a course badge.

## Specific evidence from the Machine Learning Specialization

**Strengths:** Independently completed collaborative-filtering cost with observed-rating guard and regularization; wrote Keras two-tower networks, squared-distance code, DQN networks, Bellman target and loss; public tests passed in supplied labs. High-level ideas click after causal, numerical explanations. Already has substantive experience debugging production systems and data quality.

**Recurring friction:** NumPy/TensorFlow shape/type distinctions (integer 8 versus tuple (8,) versus nested tuple); axes and indexing; moving from math formula to a precise vectorized expression; applying masks/coefficient scope correctly; tracing source of X/y/reward, prediction, target and gradients; distinguishing training from inference and model from database. Some errors were simple naming/string typos rather than conceptual weaknesses.

**Teaching insight:** In Lunar Lander the decisive missing connection was that env.reset and env.step simulate the world and generate observations/rewards, rather than a human hand-labeling correct actions. Start future deep explanations with provenance and the entire small dataflow, then translate into individual code operations. Use one real or clearly illustrative numeric case, shapes and expected intermediate values.

**Strengths should not be erased by weaknesses:** The student completed challenging tasks with progressively less assistance and asked high-quality systems questions. Don't equate not memorizing tf.gather_nd or low-level API details with failure to understand an algorithm.

## Reusable follow-up pattern

On course/lab completion or material sticking point, optionally provide a compact **learning assessment** integrated into ordinary feedback:
- Demonstrated strength and supporting evidence.
- Most important remaining conceptual/coding gap.
- A small bridge exercise or explanation where useful.
- Recommendation to continue or adjust sequence and why.
- Course evidence and documentation needing update.

Do not over-assess every chat message or force the student to demonstrate knowledge they did not ask to be tested on. The student chooses the depth and when to move forward.

## Source and change management

Canonical career plan: ROADMAP.md. This file: adaptive teaching and assessment contract. Exercise rule: GUIDED_CODING_GOLDEN_RULE.md. Learning events: LEARNING_LOG.md. Full theory and examples: knowledge/MACHINE_LEARNING_SPECIALIZATION_COMPLETE_BOOKLET.md. Syntax: knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md.

When the roadmap changes, update the rationale and the new plan in these existing canonical files; avoid duplicate competing versions, uncontrolled branches or unsupported claims about credential completion.
