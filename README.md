# Large Language Models (LLMs)

## 📖 Introduction

This repository comprises a collection of **hands-on Jupyter notebooks** showcasing practical applications of **Large Language Models (LLMs)** and modern **transformer architectures** across core Natural Language Processing (NLP) tasks.

The goal is to provide **clear, code-first examples** of how LLMs and pre-trained transformer models can be used for:
- Text generation  
- Text classification  
- Question answering  
- Emotion detection  

Large Language Models are neural networks trained on massive text corpora that excel at language understanding and generation. Transformers, the architectural backbone of LLMs, rely on **self-attention mechanisms** to model contextual relationships effectively across diverse NLP applications.

---

## 📁 Repository Structure

| File | Description |
|-----|------------|
| **GPT Models.ipynb** | Prompt-based text generation using GPT-style LLMs |
| **HuggingFace Transformers.ipynb** | Hugging Face Transformers framework|
| **Question and Answering Models with BERT.ipynb** | Extractive question answering using BERT |
| **Text Classification with XLNet.ipynb** | Text classification using XLNet |
---

## 🚀 Running the Notebooks

### 1. Clone the Repository
```bash
git clone https://github.com/iamkarandeepsingh/LLMs.git
cd LLMs
```

### 2. Set Up a Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
```

On Windows:
```bash
.venv\Scripts\activate
```
## 🧠 Notebook Highlights

### 📌 GPT Models

This notebook provides a systematic exploration of **Large Language Models (LLMs)** through GPT-style architectures, with an emphasis on **prompt-conditioned generation** and inference-time behavior rather than task-specific fine-tuning.

Key technical focus:
- Prompt formulation as an implicit control mechanism for model behavior
- Sensitivity of generated outputs to lexical and structural prompt variations
- Analysis of coherence, contextual grounding, and response diversity
- Practical interaction with LLM APIs under real-world usage constraints

The notebook develops a strong understanding of **prompt engineering as a first-class design choice** in modern LLM-driven systems and highlights the trade-offs inherent in generative modeling.

---

### 📌 HuggingFace Transformers

This notebook examines the **core abstractions of the Hugging Face Transformers framework**, providing a model-agnostic view of transformer-based NLP pipelines.

Key technical focus:
- Tokenization strategies and their impact on downstream representations
- Construction of attention masks and model inputs
- Extraction and interpretation of hidden states and embeddings
- Unified inference pipelines across heterogeneous NLP tasks

This notebook establishes a **foundational understanding of transformer internals**, enabling principled model selection and reuse across multiple NLP problem settings.

---

### 📌 Question Answering with BERT

This notebook investigates **extractive question answering** using BERT, framing the task as a **token-level span prediction problem** under a bidirectional attention mechanism.

Key technical focus:
- Encoding of question–context pairs for joint representation learning
- Interpretation of start and end logits as probabilistic span boundaries
- Answer span extraction under overlapping and ambiguous contexts
- Failure modes of extractive QA under long or noisy inputs

The notebook provides insight into **fine-grained token-level reasoning** and demonstrates how bidirectional transformers localize semantic information within text.

---

### 📌 Text Classification with XLNet

This notebook explores **sequence-level text classification** using XLNet, an autoregressive transformer that departs from standard bidirectional pretraining objectives.

Key technical focus:
- Permutation-based language modeling and its implications
- Input formatting and positional encoding in autoregressive transformers
- Adaptation of generative representations for discriminative tasks
- Comparative behavior relative to bidirectional architectures such as BERT

This notebook highlights architectural trade-offs between **autoregressive and bidirectional transformers** in classification-oriented NLP tasks.

---

## 🎓 Learning Outcomes

By completing this repository, the reader develops a **rigorous, systems-level understanding** of transformer-based NLP models, extending beyond surface-level usage.

Specifically, this work enables:

- A principled understanding of **Large Language Model (LLM) inference**, including prompt-conditioned behavior, response variability, and limitations inherent to generative models.
- Deep familiarity with **transformer preprocessing pipelines**, including tokenization, attention masking, and embedding construction.
- The ability to interpret **model outputs at different granularities**, ranging from token-level logits (e.g., extractive QA) to sequence-level predictions (e.g., classification).
- Insight into how architectural choices (bidirectional vs. autoregressive transformers) influence downstream task performance and failure modes.
- Practical experience designing and executing **end-to-end NLP inference workflows** suitable for academic, experimental, and applied settings.

Overall, the repository builds **conceptual fluency and implementation-level competence** with modern transformer models.

---

## ✅ Key Takeaways

- Transformer architectures provide a **unified modeling paradigm** capable of addressing diverse NLP tasks, but their effectiveness is highly dependent on task formulation and input representation.
- Prompt design functions as an implicit control mechanism for LLMs, significantly influencing generation quality, factual consistency, and response diversity.
- Bidirectional models (e.g., BERT) excel at **localized semantic reasoning**, making them well-suited for token-level tasks such as extractive question answering.
- Autoregressive models (e.g., XLNet, GPT-style models) exhibit strengths in **context modeling and generation**, but require careful adaptation for discriminative tasks.
- Pretrained transformer models offer strong inference-time performance; however, they expose **systematic limitations** in handling ambiguity, emotional nuance, and domain-specific semantics.
- Effective use of LLMs requires not only model access, but also **interpretation of outputs, awareness of failure modes, and informed architectural choices**.

These findings reinforce that modern NLP systems are not purely model-driven, but are shaped by **design decisions at the data, prompt, and inference levels**.
