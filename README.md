# Attention-Enhanced Spiking Graph Neural Network for Facial Deepfake Detection (DC-SGAT)

**A Technical Writing Report and Research Proposal**  
Submitted in partial fulfillment of the requirements for the award of Bachelor of Technology  
**Department of Computer Science, International Institute of Information Technology Bhubaneswar**  
*November 2026*

**Authors:**
- **Amit Kumar Hota** (B123013)
- **Sibani Sudan** (b423056)

*Supervisor: [Guide/Supervisor Name]*  
*Document Status: Research proposal and experimental protocol — model results not yet measured.*

---

## Abstract

Facial deepfake detection systems must distinguish manipulation evidence from identity, background, and processing artifacts while remaining reliable on unfamiliar datasets. Existing multi-view detectors combine facial geometry with edge and residual-noise information, but their graph reasoning is conventionally implemented using continuous-valued artificial neurons. Spiking neural networks (SNNs) offer an alternative evidence-accumulation mechanism; however, replacing a network component or adding generic attention does not by itself establish originality or improved performance.

This project proposes a **Discrepancy-Conditioned Spiking Graph Attention (DC-SGAT)** model for facial deepfake detection. Starting from the multi-information fusion strategy of MIF-Net, the proposed system extracts landmark-centered descriptors from edge-enhanced (Sobel) and residual-noise (constrained Bayar-style) views using a shared conventional image encoder. Corresponding regional descriptors define a two-view facial graph (136 nodes: 68 per view). Leaky integrate-and-fire (LIF) graph layers process node representations over a bounded simulation interval ($T=8$), while an attention mechanism conditioned on cross-view feature discrepancy regulates information exchange.

---

## Controlled Model & Ablation Matrix

To isolate the actual source of any empirical performance gain, experiments are separated across a controlled matrix:

| ID | Model | Question Answered |
| :--- | :--- | :--- |
| **B0** | Image-only encoder and classifier | Is graph modeling helpful? |
| **B1** | Adapted two-view ANN-GCN | Conventional continuous graph reference baseline |
| **B2** | Fixed-step SNN-GCN, no attention | Direct effect of spiking substitution |
| **B3** | SNN + ordinary graph attention | Effect of generic attention |
| **P** | **SNN + discrepancy-conditioned attention (DC-SGAT)** | Increment beyond ordinary attention |
| **A1** | Model P with discrepancy vector zeroed | Parameter-matched discrepancy control |
| **A2** | Model P edge-only / residual-only | Are both views useful? |
| **A3** | Model P at $T \in \{4, 8, 16\}$ | Accuracy/step sensitivity |
| **Opt**| Model P + SEENN-I confidence early exit | Adaptive inference trade-off |

---

## Key Findings & Literature Analysis

### 1. MIF-Net (Xu et al., IEEE TMM 2026)
- **Reference**: DeepFake Detection With Multi-View Fusion and Graph Convolutional Network (`10.1109/TMM.2025.3618565`).
- Combines facial landmarks, graph convolution, a Sobel branch, a constrained residual-filter branch, attention fusion, and a sigmoid classifier.
- Serves as the principal structural baseline (Model B1).

### 2. MAST (Jang et al., arXiv:2605.05895v2)
- Uses learnable LIF dynamics with Spike-Driven Self-Attention for AI-generated video detection.
- Because MAST already uses spike-driven attention and MIF-Net already uses attention fusion, **generic attention alone is not a defensible originality claim**. The proposed contribution is the specific *discrepancy-conditioned* attention mechanism and its controlled evaluation.

### 3. SEENN (Li et al., NeurIPS 2023)
- SEENN-I introduces confidence-based early exit for SNNs ($\max(p_t, 1 - p_t) \ge \eta$).
- Included as an optional inference-efficiency extension, not as a deepfake detector or new invention.

---

## Model Architecture (DC-SGAT)

```
Aligned Face (224×224) + 68 Facial Landmarks
               │
       ┌───────┴───────┐
       ▼               ▼
Sobel Edge View   Constrained Residual View
       │               │
       └───────┬───────┘
               ▼
Shared Lightweight Encoder + Landmark Region Pooling
               │
               ▼
Two-View Graph (136 nodes: 68 edge + 68 residual, matching cross-view links)
               │
               ▼
Two LIF Graph Layers with Discrepancy-Conditioned Cross-View Attention
               │
               ▼
Time-Averaged Membrane Readout & Graph Mean Pooling
               │
               ▼
Continuous Fake Probability (Validation-Selected Threshold)
```

### Mathematical Formulation
1. **Residual Constraint**: Center coefficient $w(0,0) = -1$, sum of off-center coefficients $\sum w = +1$.
2. **Discrepancy Vector**: $d_i = |h_i^e - h_i^r|$ between corresponding edge and residual view descriptors.
3. **Cross-View Attention**:
   $$e_i^{er} = b^T \text{LeakyReLU}(W_q h_i^e + W_k h_i^r + W_d d_i)$$
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
4. **LIF Dynamics**:
   $$\tilde{U}_t^{(l)} = \beta U_{t-1}^{(l)} + J_t^{(l)}, \quad S_t^{(l)} = \Theta(\tilde{U}_t^{(l)} - \theta), \quad U_t^{(l)} = \tilde{U}_t^{(l)} - \theta S_t^{(l)}$$
5. **Loss & Activity Penalty**:
   $$\mathcal{L} = \text{BCEWithLogits}(\text{logit}, y) + \lambda_s R_{spike}$$

---

## Evaluation Protocol & Safeguards

- **No Data Leakage**: Video identifiers are strictly split before frame extraction. All frames, crops, and derivatives from a source video remain in the same split. Frame-random splitting is prohibited.
- **Datasets**: FaceForensics++ (initial training/internal testing) and Celeb-DF (external test).
- **Mandatory Metrics**: Accuracy, Precision, Recall/Sensitivity, F1-score (positive class fake default), Specificity, ROC curve, ROC-AUC, Average Precision (AP), and confusion matrix (`[[TN, FP], [FN, TP]]`).
- **Validation-Selected Threshold**: Decision thresholds are tuned strictly on validation and frozen before testing.

---

## Repository Structure

```
dc-sgat/
├── configs/
│   └── config.yaml              # Experimental configuration & hyperparameter manifest
├── docs/
│   └── Deepfake_SNN_Technical_Writing_Draft.pdf  # Full 22-page research proposal PDF
├── src/
│   ├── models/
│   │   ├── filters.py           # Sobel & Constrained Bayar Conv2d
│   │   ├── lif_neuron.py        # LIF spiking dynamics with surrogate gradients
│   │   ├── attention.py         # Discrepancy-conditioned attention module
│   │   ├── graph_utils.py       # 68-landmark facial topology & adjacency
│   │   ├── dc_sgat.py           # Full DC-SGAT model implementation
│   │   └── baselines.py         # B0, B1, B2, B3 baseline models
│   ├── data/
│   │   ├── dataset.py           # Leakage-safe video partitioned dataset
│   │   └── transforms.py        # Coherent geometric image/landmark transforms
│   ├── evaluation/
│   │   └── metrics.py           # Metric calculation & video-level aggregation
│   ├── train.py                 # Training script (Algorithm 3.1)
│   └── evaluate.py              # Evaluation script (Algorithm 4.1)
├── tests/
│   └── test_dc_sgat.py          # Unit tests verifying forward/backward passes
├── requirements.txt
├── LICENSE                      # MIT License
└── README.md
```

---

## Scientific Integrity Statement

This repository contains the technical writing research proposal, architectural formulation, and initial implementation framework. **No newly trained model checkpoints or empirical detector scores are reported yet.** Published numbers from cited papers are clearly separated and attributed. Genuine experimental metrics will be populated during semester 8 evaluation.
