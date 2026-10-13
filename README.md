
## Motivation

Modern conversational systems frequently encounter queries such as:

**User:** Which customers escalated due to billing issues?

**Assistant:** Returns relevant cases.

**User:** What were the common reasons?

The follow-up query contains no explicit reference to billing issues or escalated customers. Humans naturally recover this context from conversation history, whereas traditional retrieval systems often treat each query independently.

The goal of this project is to reconstruct the missing context and generate a self-contained query that can be executed reliably by downstream retrieval and analytics systems.

---

## Problem Statement

Given:

* A multi-turn conversation history
* A context-dependent follow-up query

Generate:

* A standalone reformulated query that preserves the original intent while incorporating all necessary contextual information from previous turns.

Example:

Input:

```text
User: Which customers escalated due to billing issues?

Assistant: Returns relevant cases.

User: What were the common reasons?
```

Generated Query:

```text
What were the common reasons behind customer escalations caused by billing issues identified in the previous result set?
```

---

# System Architecture

The architecture is built around a hierarchical conversational memory representation.

Rather than storing the entire dialogue history, the system constructs a compressed semantic representation that balances recent conversational context with long-term dialogue information.

---

## Conversational Memory Representation

The conversational state is modeled using two complementary memory components.

### Short-Term Memory

Captures recent conversational dynamics.

For every query:

* Extract embeddings of the previous 5 conversation turns
* Compute an aggregated representation
* Preserve recently introduced entities, constraints, and references

This component emphasizes local conversational continuity.

---

### Long-Term Memory

Captures historical context.

A rolling summary of the entire conversation is maintained throughout the interaction.

The summary:

* Compresses long dialogue histories
* Preserves global context
* Retains information beyond the recent interaction window

The summary is converted into a semantic embedding representation.

---

### Context Fusion

Three embeddings participate in query understanding:

```text
E_recent   = Average(Embeddings of Last 5 Turns)

E_summary  = Embedding(Global Conversation Summary)

E_query    = Embedding(Current User Query)
```

The conversational representation is formed by combining:

```text
Conversation State

= Fusion(
    E_recent,
    E_summary,
    E_query
  )
```

This design intentionally biases the representation toward recent interactions while retaining awareness of long-range conversational context.

---

## Query Rewriting Pipeline

The generated conversational state is supplied to a query rewriting model.

Responsibilities:

* Pronoun resolution
* Reference recovery
* Missing attribute completion
* Context reconstruction
* Intent preservation

The output is a context-complete analytical query suitable for downstream retrieval systems.

---

# Synthetic Benchmark Generation

A major challenge in conversational query rewriting research is the lack of publicly available datasets containing diverse conversational dependencies.

To address this limitation, a synthetic benchmark generation framework was developed.

---

## Dependency Taxonomy

Each generated conversation belongs to one of four structural categories.

### Direct Dependency

References the immediately preceding turn.

Example:

```text
User: Which customers churned?

User: What were their complaints?
```

---

### Long-Range Dependency

References information from earlier portions of the conversation.

Example:

```text
User: Show escalated customers.

...

User: Which of those had billing issues?
```

---

### Multi-Dependency

Requires information from multiple previous turns simultaneously.

Example:

```text
Compare the customers discussed earlier with the escalations identified later.
```

---

### Independent Query

Introduces a new topic unrelated to previous context.

Example:

```text
What are the available subscription plans?
```

---

## Intent Taxonomy

Each generated follow-up query is assigned a conversational intent.

Supported intents include:

* Clarification
* Verification
* Exploration
* Condition Addition
* Condition Removal
* Response Confirmation
* Response Critique
* Format Transformation
* Opinion Request
* Information Expansion

This dual-axis taxonomy enables diverse and realistic conversational behavior.

---

## Dataset Generation Pipeline

The benchmark generation process follows four stages.

### 1. Dependency Sampling

Selects the structural dependency type.

### 2. Parent Selection

Identifies historical conversation turns that should influence the next query.

Parent selection considers:

* Recency
* Semantic Similarity
* Utility Scores

### 3. Intent Assignment

Determines the user's conversational objective.

### 4. Query Synthesis

Generates a context-dependent follow-up query consistent with the chosen dependency structure and intent.

This process produces realistic multi-turn conversational interactions for evaluating contextual reasoning systems.

---

# Evaluation Methodology

The generated rewritten query is compared against a ground-truth reformulation.

The objective is to measure how effectively the system reconstructs missing conversational context.

---

## Metrics

### F1 Score

Measures token-level overlap between generated and reference queries.

---

### Semantic Overlap

Measures semantic equivalence between generated and reference queries.

Computed using:

* BERTScore
* Entailment-based similarity

---

# Results

| Metric           | Score |
| ---------------- | ----- |
| F1 Score         | 0.86  |
| Semantic Overlap | 0.78  |

The results indicate strong context reconstruction performance across diverse conversational dependency structures.

---

# Key Contributions

* Hierarchical conversational memory architecture
* Recency-aware context encoding
* Query rewriting for conversational follow-ups
* Synthetic benchmark generation for contextual reasoning
* Dependency-aware dialogue simulation
* Intent-conditioned query synthesis
* Automated evaluation framework
* Context reconstruction using semantic representations

---

# Technology Stack

### Language Models

* GPT-4o-mini
* GPT-4o

### NLP Components

* Semantic Embeddings
* Query Rewriting
* Context Reconstruction
* Intent Modeling

### Evaluation

* F1 Score
* BERTScore
* Semantic Overlap
* Entailment-Based Similarity

---

# Conversational-Query-Rewriting
