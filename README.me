# MolPredictor_Alpha 🧬

A comprehensive machine learning pipeline for Tox21 ADMET toxicity prediction (NR-AR pathway), exploring the Pareto frontier of traditional ML models on highly imbalanced chemical datasets.
一个用于 Tox21 ADMET 毒性预测（NR-AR 通路）的综合机器学习流水线，探索传统机器学习模型在极度不平衡化学数据集上的帕累托边界。

## 📖 Project Overview (项目简介)

Predicting molecular toxicity is a critical step in AI-driven Drug Discovery (AIDD). This project serves as the baseline (Alpha phase) for the MolPredictor initiative. It evaluates the performance of traditional machine learning algorithms—including K-Nearest Neighbors (KNN), Logistic Regression (with L1/L2 regularization), Random Forest, and XGBoost—using 1024-bit Morgan Fingerprints. 

预测分子毒性是人工智能驱动的药物发现（AIDD）中的关键步骤。本项目是 MolPredictor 计划的基线（Alpha 阶段）。它使用 1024 位 Morgan 指纹评估了传统机器学习算法的性能，包括 K 近邻（KNN）、逻辑回归（含 L1/L2 正则化）、随机森林和 XGBoost。

The core challenge addressed in this repository is the extreme class imbalance (approx. 1:22 for toxic vs. non-toxic compounds) and the inherent sparsity of the structural features.

本仓库解决的核心挑战是极度的类别不平衡（毒性与无毒化合物比例约为 1:22）以及结构特征固有的稀疏性。

Tech Stack & Environment (技术栈与环境)
- Core (核心): Python 3.12, Pandas, NumPy
- Chemoinformatics (化学信息学): RDKit (SMILES to Morgan Fingerprints)
- Machine Learning (机器学习): Scikit-learn, XGBoost
- Hardware/OS (硬件与系统): WSL2 (Ubuntu 24.04) on Windows 11, CUDA-enabled (NVIDIA GeForce RTX 5060 Laptop GPU)

Key Results & Strategy Evolution (核心战果与策略演进)

Traditional models like Random Forest hit a "recall ceiling" of 0.45 due to the extreme sparsity of the data. To break this bottleneck, we implemented a dual-strategy using XGBoost:

由于数据的极端稀疏性，像随机森林这样的传统模型触及了 0.45 的“召回率天花板”。为了打破这一瓶颈，我们使用 XGBoost 实施了双重策略：

1. Training Phase (Mathematical Pruning): Applied a rational `scale_pos_weight` with `gamma` and L2 regularization (`reg_lambda`) to prevent the model from memorizing noise (overfitting).
1. 训练阶段（数学剪枝）： 应用理性的 `scale_pos_weight` 配合 `gamma` 和 L2 正则化 (`reg_lambda`)，防止模型死记硬背噪声（过拟合）。

2. Inference Phase (Threshold Moving): Lowered the classification threshold to `0.3` to prioritize sensitivity, aligning with the industrial requirement of early virtual screening where false negatives are highly costly.
2. 推理阶段（阈值移动）： 将分类阈值降低至 `0.3` 以优先保证灵敏度，这符合早期虚拟筛选的工业需求，因为在这一阶段假阴性（漏报）的代价极其高昂。

**Performance Comparison (性能对比):**

| Model / Strategy                         | Test AUC   | Gap (Overfitting) | Recall (Toxic) | Precision |
| Random Forest (Baseline)                 | 0.7797     | 0.0941            | 0.45           | 0.42      |
| XGBoost (Default Threshold 0.5)          | 0.7930     | 0.1095            | 0.44           | 0.69      |
| XGBoost (Extreme Pos Weight)             | 0.7547     | 0.2066            | 0.55           | 0.11      |
| XGBoost (Rational Weight + Threshold 0.3)| **0.7630** | **0.2001**        | **0.63**       | **0.08**  |

Result: Successfully intercepted 39 out of 62 toxic molecules, creating a highly sensitive early-warning radar for toxicity.
结果：成功拦截了 62 个毒性分子中的 39 个，打造了一个高灵敏度的早期毒性预警雷达。

Future Work: MolPredictor Phase 1 (未来工作：MolPredictor P1 阶段)
Feature attribution analysis (Gain scores) revealed that traditional ML models heavily rely on a few specific fingerprint bits (e.g., Bit 519). "Flattening" a 3D molecular structure with complex electron cloud interactions into a 1024-bit vector causes severe information loss.

特征归因分析（Gain scores）表明，传统机器学习模型高度依赖极少数的特定指纹位点（如 Bit 519）。将具有复杂电子云相互作用的 3D 分子结构“压扁”成 1024 维向量会导致严重的信息折损。

The next phase of this project will transition to Deep Representation Learning using Graph Neural Networks (GNNs). By representing compounds directly as topological graphs (atoms as nodes, bonds as edges) and utilizing Message Passing, we aim to overcome the limitations of handcrafted features and achieve truly data-driven ADMET predictions.

本项目的下一阶段将过渡到使用图神经网络（GNN）的深度表征学习。通过将化合物直接表示为拓扑图（原子为节点，化学键为边）并利用消息传递机制（Message Passing），我们旨在克服人工特征的局限性，实现真正数据驱动的 ADMET 预测。