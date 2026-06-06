# BITS Pilani M.Tech (AI & ML) - Theory to Production Mastery

Welcome to the ultimate practical implementation playground. The goal of this repository is to bridge the gap between academic theory and production-grade ML engineering. 

This repository is structured as a curated engineering portfolio. It transitions systematically from writing foundational algorithms using pure math up to building multi-agent, production-ready AI systems.

---

## 🏗️ Portfolio Architecture & Progress Tracker

| Module / Project Name | Difficulty | Core Tech Stack | Status |
| :--- | :---: | :--- | :---: |
| **[01] Deep Learning Foundations** | | | |
| ├── Neural Networks From Scratch | Intermediate | NumPy, Python | ✅ Completed |
| ├── CNN Classification Pipeline | Intermediate | PyTorch, Albumentations, W&B | 🔄 Planning |
| ├── Time Series Forecasting | Intermediate | PyTorch, TCN, Optuna | 🔄 Planning |
| └── Transformer From Scratch | Advanced | PyTorch, RoPE, KV-Cache | 🔄 Planning |
| **[02] Reinforcement Learning Zoo** | | | |
| ├── Tabular Planning Matrices | Beginner | NumPy, Gymnasium | 🔄 Planning |
| ├── Deep Q-Network Variants | Intermediate | PyTorch, Gymnasium, W&B | 🔄 Planning |
| ├── PPO Game Playing Agent | Advanced | PyTorch, Vectorized Gym, GAE | 🔄 Planning |
| └── Imitation Learning Sandbox | Intermediate | PyTorch, SB3 Trajectories | 🔄 Planning |
| **[03] Natural Language Processing** | | | |
| ├── Word2Vec Embeddings | Beginner | PyTorch, UMAP, Tokenizers | 🔄 Planning |
| ├── Core Parsing & Tagging | Intermediate | PyTorch, BiLSTM-CRF, Spacy | 🔄 Planning |
| └── BERT Multitask Fine-Tuning | Advanced | PyTorch, HF Accelerate | 🔄 Planning |
| **[04] Flagship Systems** | | | |
| ├── Distributed ML Sandbox | Advanced | PyTorch, FedProx, MAML | 🔄 Planning |
| └── Enterprise GraphRAG System | Expert | Neo4j, FAISS, LlamaIndex, Docker | 🔄 Planning |

*Status Indicators: 🔄 Planning | 🏗️ In Progress | ✅ Completed*

---

## 🛠️ Code Quality & Engineering Pillars

To maximize the resume and research impact of this repository, every implementation folder must strictly follow these engineering guidelines:

1. **No Magic Numbers:** All hyperparameters, paths, and configurations must live inside explicit `.yaml` configs.
2. **Deterministic Outputs:** Global random seeds must be locked across all frameworks to guarantee reproducibility.
3. **Automated Tracking:** No print loops for metrics. Use Weights & Biases (`wandb`) for real-time tracking and logging.
4. **Unit Testing:** Write `pytest` checks to verify tensor dimensions, loss calculation behavior, and dataset tokenization alignment.

---

## 🤝 Collaboration & Contribution Guidelines (For Batchmates)

Let's learn together. Bug fixes, optimizations, documentation enhancements, and alternative architectural variants are highly encouraged via Pull Requests!

### How to Raise a PR:
1. **Fork/Clone the Repo:** Ensure you are working on your own feature branch: `git checkout -b feature/your-feature-name`.
2. **Follow the Structure:** Keep your additions cleanly isolated within the appropriate sub-folders. Do not litter the root folder.
3. **Ensure Formatting Passes:** Run linting tools or execute validation test files before committing your work.
4. **Submit Your PR:** Fill out the provided PR template details clearly (what you changed, why, and a link to experimental proof or logs if applicable).

### Code Review Expectations:
* Every PR will be thoroughly reviewed.
* Code style, tensor alignment checking, and performance validation tests must pass before merging.
* Be ready for collaborative, constructive discussions in the comments!

---

## 📜 License
This repository is open-sourced under the MIT License. Feel free to use, adapt, and build upon this code for your own academic and research pursuits.