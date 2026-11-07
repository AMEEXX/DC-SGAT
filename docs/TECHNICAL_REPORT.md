# Attention-Enhanced Spiking Graph Neural Network for Facial Deepfake Detection

**A Technical Writing Report Submitted in partial fulfillment of the requirements for the award of the Bachelor of Technology**

**by**  
- **Amit Kumar Hota (B123013)**  
- **Sibani Sudan (b423056)**  

**Under the supervision of**  
`[Guide/Supervisor Name]`

Department of Computer Science  
International Institute of Information Technology Bhubaneswar  
*November 2026*  

*Research proposal and experimental protocol — model results not yet measured*

---

## APPROVAL OF THE VIVA-VOCE BOARD

*November [Date], 2026*  
**UNSIGNED TEMPLATE — to be completed only after the viva-voce examination.**

The report entitled **“Attention-Enhanced Spiking Graph Neural Network for Facial Deepfake Detection”**, submitted by Amit Kumar Hota (B123013) and Sibani Sudan (b423056), is presented for consideration under the Bachelor of Technology programme of the International Institute of Information Technology Bhubaneswar. Acceptance, examination date and examiner signatures are to be entered by the authorized board; this draft does not certify that an examination has occurred.

`[Supervisor signature]` &emsp;&emsp;&emsp;&emsp; `[Panel members and signatures]`

### Document Status
This document is a proposed technical-writing report for semester 7 and an implementation plan for semester 8. It is not a completed experimental paper. No new model has been trained or evaluated for this report. Proposed architecture, training configurations and outcomes are explicitly distinguished from published observations. Supervisor details, institution-specific approvals and final experimental evidence remain to be supplied.

---

## ACKNOWLEDGEMENTS

We thank our project supervisor and the faculty of the Department of Computer Science for guidance in defining the problem of facial deepfake detection using spiking neural networks. We acknowledge the authors of the cited studies and the maintainers of research datasets and software resources on which the proposed investigation builds. This draft acknowledgement should be reviewed by both authors before submission and supplemented with the supervisor’s confirmed name.

- **Amit Kumar Hota (B123013)**
- **Sibani Sudan (b423056)**

---

## ABSTRACT

Facial deepfake detection systems must distinguish manipulation evidence from identity, background, and processing artifacts while remaining reliable on unfamiliar datasets. Existing multi-view detectors combine facial geometry with edge and residual-noise information, but their graph reasoning is conventionally implemented using continuous-valued artificial neurons. Spiking neural networks offer an alternative evidence-accumulation mechanism; however, replacing a network component or adding generic attention does not by itself establish originality or improved performance. 

This technical-writing report proposes a discrepancy-conditioned spiking graph attention model for facial deepfake detection. Starting from the multi-information fusion strategy of MIF-Net, the proposed system will extract landmark-centered descriptors from edge-enhanced and residual-noise views using a shared conventional image encoder. Corresponding regional descriptors will define a two-view facial graph. Leaky integrate-and-fire graph layers will process the node representations over a bounded simulation interval, while an attention mechanism conditioned on cross-view feature discrepancy will regulate information exchange. 

The research will compare a conventional graph baseline, a fixed-step spiking replacement, ordinary graph attention, and discrepancy-conditioned attention under matched preprocessing and training conditions. Evaluation will include accuracy, precision, recall, F1-score, specificity, receiver operating characteristic curves, ROC–AUC, average precision, and confusion matrices. Cross-dataset and compression experiments will test generalization, and measured latency, spike activity, and average simulation steps will characterize computational behavior. A confidence-based early-exit experiment inspired by SEENN is optional and will not be treated as a new invention. All proposed improvements remain hypotheses: the present document reports no newly trained checkpoint, measured detector score, or guaranteed novelty. The planned implementation will determine whether the specific attention mechanism improves robustness beyond established alternatives.

**Keywords**: facial deepfake detection; spiking neural network; graph attention; multi-view fusion; surrogate gradients; robustness.

---

## TABLE OF CONTENTS

- ABSTRACT
- LIST OF TABLES
- LIST OF FIGURES
- LIST OF ALGORITHMS
- LIST OF ABBREVIATIONS
- CHAPTER 1 — INTRODUCTION
  - 1.1 Background and motivation
  - 1.2 Problem statement
  - 1.3 Objectives and research questions
  - 1.4 Scope and exclusions
  - 1.5 Expected contributions and scientific integrity
- CHAPTER 2 — RELATED WORK AND SOURCE ANALYSIS
  - 2.1 MIF-Net: the principal reference
  - 2.2 MAST: what the supplied SNN paper already contains
  - 2.3 SEENN: optional inference-efficiency reference
  - 2.4 Published scores: attribution and limits
  - 2.5 Prior art and bounded research gap
- CHAPTER 3 — PROPOSED METHODOLOGY
  - 3.1 Model definition and design rationale
  - 3.2 Preprocessing and forensic views
  - 3.3 Node features and graph construction
  - 3.4 Discrepancy-conditioned attention
  - 3.5 LIF graph dynamics and gradients
  - 3.6 Readout, loss and inference
  - 3.7 Optional confidence-based early exit
- CHAPTER 4 — EXPERIMENTAL DESIGN AND PENDING RESULTS
  - 4.1 Current experimental status
  - 4.2 Dataset protocol and leakage prevention
  - 4.3 Initial configuration and compute assumptions
  - 4.4 Controlled model matrix
  - 4.5 Mandatory metrics and score conventions
  - 4.6 Robustness, plots and training records
  - 4.7 Pending results — not yet measured
  - 4.8 Reproducibility, risks and milestones
- CHAPTER 5 — CONCLUSION AND FUTURE SCOPE
  - 5.1 Conclusion
  - 5.2 Future scope
  - 5.3 Submission and implementation readiness
- REFERENCES

---

## LIST OF TABLES

- 2.1 Comparison of the three supplied research papers
- 2.2 Published results and comparison boundaries
- 3.1 Proposed tensor dimensions and components
- 4.1 Initial experimental configuration
- 4.2 Controlled baseline and ablation matrix
- 4.3 Required metric definitions
- 4.4 Pending detector results
- 4.5 Pending robustness and computation results
- 4.6 Research implementation milestones

## LIST OF FIGURES
- Figure 3.1. Proposed architecture block diagram (conceptual, not an experimental result).

## LIST OF ALGORITHMS
- Algorithm 3.1. Fixed-step training and inference.
- Algorithm 4.1. Leakage-safe evaluation and metric generation.

## LIST OF ABBREVIATIONS

| Abbreviation | Meaning |
| :--- | :--- |
| **ANN** | Artificial neural network |
| **SNN** | Spiking neural network |
| **GCN / GAT** | Graph convolutional / graph attention network |
| **LIF** | Leaky integrate-and-fire |
| **BCE** | Binary cross-entropy |
| **ROC–AUC** | Area under the receiver operating characteristic curve |
| **AP** | Average precision |
| **TP / FP** | True positive / false positive |
| **TN / FN** | True negative / false negative |
| **MVP** | Minimum viable implementation |

---

## CHAPTER 1 — INTRODUCTION

### 1.1 Background and motivation
Facial manipulation detection is a binary classification problem in which authentic and altered face images must be separated using evidence that remains useful outside the training distribution. Identity, compression, acquisition conditions and preprocessing can all become shortcuts. Therefore, a strong result on one dataset is insufficient evidence of reliable deployment. MIF-Net combines facial landmarks, Sobel-enhanced image information and a constrained residual view to address complementary evidence sources [1].

The teacher’s requested modification is interpreted as a research programme: replace conventional graph processing with spiking graph processing, then investigate a specifically defined attention mechanism. The objective is to measure whether these changes help, rather than assume that an SNN necessarily improves accuracy. Attention is already present in both MIF-Net and MAST; the contribution must concern how the proposed attention is constructed and evaluated [1,2].

### 1.2 Problem statement
Given an aligned facial image $I$, predict a continuous fake probability $p \in [0,1]$ and a binary class after applying a validation-selected threshold. Investigate whether discrepancy-conditioned attention on a multi-view facial spiking graph improves discrimination and robustness compared with matched conventional and spiking alternatives.

### 1.3 Objectives and research questions
- **RQ1**: What changes when the conventional graph operator is replaced by a fixed-step LIF graph operator?
- **RQ2**: Does ordinary graph attention already explain any improvement?
- **RQ3**: Does conditioning attention on corresponding edge/residual feature discrepancy provide an additional benefit?
- **RQ4**: Are observed changes preserved under compression and external-dataset testing?
- **RQ5**: What computational overhead accompanies the proposed attention?

The primary comparison is discrepancy-conditioned versus ordinary attention using the same spiking backbone. The ANN-versus-SNN comparison is a separate research question. Accuracy, precision, recall, F1, ROC–AUC and a confusion matrix are mandatory outcomes, not assumed advantages.

### 1.4 Scope and exclusions
The initial implementation is a hybrid ANN–SNN frame-level facial manipulation detector. A conventional encoder supplies node features; graph layers use spiking dynamics; the final classifier is continuous-valued. A video score may be obtained by aggregating frame probabilities. This does not constitute learned inter-frame temporal modeling. Audio, fully generated scene videos, face recognition and neuromorphic hardware deployment are outside the MVP.

### 1.5 Expected contributions and scientific integrity
The candidate contribution is a testable discrepancy-conditioned cross-view attention mechanism implemented within a facial spiking graph, supported by controlled ablations and reproducible evaluation. No “first-ever” claim is made. A model substitution can be an acceptable undergraduate experiment even when it is not a globally novel method, provided attribution is clear and the evidence is genuine. Negative or mixed findings must be reported.

---

## CHAPTER 2 — RELATED WORK AND SOURCE ANALYSIS

### 2.1 MIF-Net: the principal reference
Xu et al. propose *DeepFake Detection With Multi-View Fusion and Graph Convolutional Network*, published online in October 2025 and in IEEE Transactions on Multimedia, volume 28, 2026, pp. 167–180, DOI 10.1109/TMM.2025.3618565 [1]. The architecture uses facial landmarks, graph convolution, a Sobel branch, a constrained residual-filter branch, attention fusion, and a sigmoid classifier. The paper alternates between “Bayer” and “Bayar”; implementation should specify the actual filter constraints instead of perpetuating the naming ambiguity.

The reported training uses 512×512 inputs, batch size 24, 50 epochs, Adam and learning rate 0.002, with a frozen backbone for 20 epochs followed by unfreezing for 30. It evaluates within FaceForensics++ and across DFDC, Celeb-DF, DeeperForensics-1.0 and DeepfakeTIMIT. The proposed project will use a smaller initial backbone and input resolution if required; such a model must be called an adapted reference baseline, not an exact reproduction [1, Sections III–V].

### 2.2 MAST: what the supplied SNN paper already contains
Jang et al., *Detecting AI-Generated Videos with Spiking Neural Networks*, arXiv:2605.05895v2, 31 July 2026, propose MAST [2]. The model combines four pixel-level temporal residual channels and two semantic displacement/curvature channels, learnable per-channel LIF dynamics, an Efficient Spike-Driven Self-Attention block, a frozen X-CLIP encoder and a classifier. Section 4.2 explicitly contains attention. Consequently, “add attention to MAST” is not an adequate novelty description.

MAST concerns cross-generator detection of AI-generated videos, rather than exclusively facial manipulation. Its frozen X-CLIP encoder uses full-clip temporal context. Simply stopping a later spiking branch early would not eliminate the already incurred encoder cost or establish causal streaming inference. The present proposal therefore treats MAST as SNN design evidence rather than a directly comparable facial-dataset accuracy baseline.

Appendix I reports ten training epochs, AdamW, weight decay 0.01, a cosine schedule, SDTB learning rate $3 \times 10^{-4}$, batch size 16 per GPU over four RTX 3090 GPUs, and three seeds. Equation 12 lists main BCE, auxiliary BCE and supervised contrastive loss; supplementary descriptions additionally discuss anomaly and rate objectives. The text also states a learning rate for an encoder described as frozen. These details require implementation-level clarification before claiming exact reproduction [2, Section 4 and Appendices F–I].

### 2.3 SEENN: optional inference-efficiency reference
Li, Geller, Kim and Panda introduce SEENN at NeurIPS 2023 [3]. SEENN-I uses confidence thresholds to stop SNN simulation, whereas SEENN-II uses a learned reinforcement-learning policy. Its experiments concern CIFAR-10, CIFAR-100, ImageNet and CIFAR10-DVS, not deepfake detection. The official repository provides training and SEENN-I evaluation code; its SEENN-II evaluation section is marked “To be updated” in the inspected version [5].

SEENN is not the replacement detector proposed here. A confidence-only SEENN-I-inspired controller may be tested after the fixed-step model works. Such a controller is attributed existing methodology, not a new contribution. Simulation steps spent on one face are distinct from consecutive video frames.

#### Table 2.1. Comparison of supplied papers.

| Source | Task / mechanism | Role in this project |
| :--- | :--- | :--- |
| **MIF-Net [1]** | Facial multi-view CNN + GCN; attention fusion | Principal model reference |
| **MAST v2 [2]** | Generated-video residual SNN; spike self-attention; X-CLIP | SNN training and attention prior art |
| **SEENN [3]** | Sample-adaptive SNN steps for recognition | Optional inference controller |

### 2.4 Published scores: attribution and limits
Table 2.2 contains author-reported reference numbers only. None is a score measured for this project, and values across different datasets are not a leaderboard comparison.

#### Table 2.2. Published results and comparison boundaries.

| Source and protocol | Published number | Interpretation |
| :--- | :--- | :--- |
| **MIF-Net Table III, FF++ frame level** | AUC 99.24%; ACC 95.47%; F1 98.96% | Definitions and population need clarification |
| **MIF-Net Table IV, FF++ frame level** | AUC 99.34%; ACC 95.79% | Differs from Table III; do not merge |
| **MAST Table 2, GenVideo unseen generators** | mACC 93.14%; mAUC 94.95% | Arithmetic mean of per-generator evaluations |
| **MAST Table 5, matched comparison** | ANN 92.78/94.92; SNN 93.14/94.95 | mACC/mAUC on its own protocol |
| **SEENN abstract, CIFAR-10** | About 96.1% accuracy; 1.08 average steps | Recognition result, not deepfake accuracy |

MAST Table 2 does not establish best AUC across all listed methods: NSG-VD is reported at 96.14% mean AUC. Its reported SNN module latency is 33.88 ms versus 54.19 ms for its ANN counterpart on RTX 3090 under the specified configuration; this is not a latency measurement of the proposed model. MIF-Net Table IX gives 30.81 FPS while the prose gives 31.81 FPS. Aggregate AUC or accuracy cannot be used to reconstruct missing precision, recall or a unique confusion matrix [1–3].

### 2.5 Prior art and bounded research gap
Graph attention is already used in deepfake image detection; the 2025 multi-graph attention study is one example [7]. Spiking networks, attention and adaptive simulation budgets also exist in the supplied papers. The scoped literature assessment therefore supports a candidate experimental gap, not proof of absence: compare cross-view discrepancy-conditioned attention with ordinary attention within an otherwise matched facial spiking graph. Further title/abstract/full-text checking of spiking graph, uncertainty-aware multi-view and facial graph methods is required before publication-level novelty claims.

---

## CHAPTER 3 — PROPOSED METHODOLOGY

### 3.1 Model definition and design rationale
The proposed working name is **DC-SGAT: Discrepancy-Conditioned Spiking Graph Attention**. “Discrepancy” denotes differences between corresponding latent descriptors from two forensic views; it is not a ground-truth forgery label. Compression, occlusion and lighting can also cause disagreement. The model must learn whether such differences are useful, and an ablation must test this hypothesis.

#### Figure 3.1. Conceptual processing path:
```
Aligned face and 68 facial landmarks
                ↓
Sobel edge view + constrained residual view
                ↓
Shared conventional encoder and landmark-region pooling
                ↓
Two-view graph: 68 nodes per view, corresponding cross-view links
                ↓
Two LIF graph layers with discrepancy-conditioned cross-view attention
                ↓
Time-averaged membrane readout and binary classifier
                ↓
Continuous fake probability → validation-selected threshold
```
*This block diagram describes intended computation, not an observed experimental result. No copied experimental plots are used.*

### 3.2 Preprocessing and forensic views
Detect and align the face once. Apply the same spatial crop to both derived views. The MVP uses 224×224 images. Construct Sobel edge magnitude and a three-channel constrained prediction-error residual. For a learned Bayar-style filter, constrain the center coefficient to −1 and the sum of off-center coefficients to +1; document normalization and numerical handling. Do not call this operation random-noise augmentation.

View normalization statistics must be estimated from training data only. Apply geometric augmentation coherently to both views and landmarks. Prefer transformations that do not erase the forensic signal. Landmark failures must be logged by class and dataset; they must not silently remove a disproportionate fraction of hard fake examples.

### 3.3 Node features and graph construction
Let landmarks be $\ell_i = (x_i, y_i)$, $i=1,\dots,68$, normalized to the aligned crop. A shared pretrained lightweight encoder produces a feature map for each view. Sample or locally pool a descriptor around each landmark, project it to $d=64$, and concatenate normalized coordinates and optionally landmark confidence before a final node projection. The same landmark positions define both view graphs. The exact landmark detector and weight version must be frozen in the experiment manifest.

$$I \in \mathbb{R}^{B \times 3 \times 224 \times 224}; \quad X^e, X^r \in \mathbb{R}^{B \times 68 \times 64}; \quad X = [X^e; X^r] \in \mathbb{R}^{B \times 136 \times 64}$$

Within-view edges follow a fixed documented facial topology plus self-links. Cross-view edges connect matching landmarks. Normalize spatial distances by face scale. Maintain identical topology in ANN and SNN controls. Do not use class labels or test-set statistics to construct adjacency.

#### Table 3.1. Proposed dimensions and operators.

| Component | Initial specification |
| :--- | :--- |
| **Views** | Edge and residual, aligned spatially |
| **Graph** | 136 nodes; fixed facial + matching-view links |
| **Node width** | 64 features after projection |
| **Graph depth** | Two layers |
| **Spiking state** | Membrane $U$: $B \times 136 \times 64$; reset per independent input |
| **Simulation steps** | $T=8$ initially; sensitivity at $T=4$ and $T=16$ |
| **Readout** | Mean final-layer membrane over steps; graph mean pooling |
| **Classifier** | Continuous MLP logit and sigmoid fake probability |

### 3.4 Discrepancy-conditioned attention
Compute projected view descriptors $h_i^e$ and $h_i^r$ using learned adapters into a shared feature space. Their absolute difference $d_i$ is included in the score for the corresponding cross-view edge. Ordinary attention controls use the same scoring network dimensions but omit discrepancy information; an additional parameter-matched control can feed zeroed discrepancy vectors.

$$h_i^v = P_v x_i^v; \quad d_i = |h_i^e - h_i^r|$$
$$e_{ij}^v = a^T \text{LeakyReLU}(W_q h_i^v + W_k h_j^v + W_r r_{ij})$$
$$e_i^{er} = b^T \text{LeakyReLU}(W_q h_i^e + W_k h_i^r + W_d d_i)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

Here $r_{ij}$ represents normalized geometric edge information; separate parameters may distinguish within-view and cross-view relations. Compute attention once from the input descriptors for each sample and reuse it across SNN steps in the MVP. A conventional softmax produces continuous weights. Therefore this design is a hybrid attention mechanism operating on spiking message sources, not a fully spike-only attention engine. Adding two heads is optional after a one-head control trains reliably.

### 3.5 LIF graph dynamics and gradients
Let $M$ be the weighted aggregation operator using $\alpha$. At layer 1, the static descriptors are injected as continuous currents at every simulation step. At deeper layers, spikes supply messages. With $0 < \beta < 1$ and threshold $\theta > 0$, use:

$$J_t^{(1)} = M X W^{(1)}; \quad J_t^{(l)} = M S_t^{(l-1)} W^{(l)}, \quad l > 1$$
$$\tilde{U}_t^{(l)} = \beta U_{t-1}^{(l)} + J_t^{(l)}; \quad S_t^{(l)} = \mathbf{1}[\tilde{U}_t^{(l)} \ge \theta]$$
$$U_t^{(l)} = \tilde{U}_t^{(l)} - \theta S_t^{(l)}$$

Initialize $\beta=0.9$ and $\theta=1.0$ as starting assumptions, not optimal settings. Use binary spikes with a documented surrogate gradient; the multispike construction in MAST is not assumed identical. Train by backpropagation through the unrolled simulation using a fast-sigmoid or ATan surrogate [6]. Reset membrane states between independent samples or minibatches during training and evaluation. If later learning $\beta$ or $\theta$, constrain their valid ranges and ablate the change separately.

Repeated SNN steps on one frame implement neural evidence accumulation, not video-frame dynamics. An optional temporal-video extension would require a separate axis for frame time and additional experiments. It is deliberately excluded from the main contribution.

### 3.6 Readout, loss and inference
$$\bar{u} = \frac{1}{T} \sum_t U_t^{(2)}; \quad z = \text{MeanPool\_nodes}(\bar{u}); \quad p = \text{sigmoid}(\text{MLP}(z))$$
$$\mathcal{L} = \text{BCEWithLogits}(\text{logit}, y) + \lambda_s R_{spike}; \quad y=1 \text{ for fake.}$$

For the MVP, begin with $\lambda_s=0$ to establish learnability. Then test a small activity penalty such as $\lambda_s \in \{10^{-4}, 10^{-3}\}$, selected on validation only. $R_{spike}$ is emitted spikes divided by available neuron-step positions. The default classification loss is unweighted if balanced sampling is used; do not automatically combine aggressive oversampling with inverse-frequency weighting. Record the chosen imbalance strategy.

#### Algorithm 3.1. Fixed-step training and inference.
1. Load a training batch and its normalized landmarks; create both forensic views.
2. Encode views, pool landmark descriptors, construct the fixed topology and attention weights.
3. Reset all LIF states. Unroll $T$ simulation steps and collect spikes and membrane states.
4. Pool the membrane readout, compute logits, BCE and any declared activity penalty.
5. Backpropagate using the surrogate; clip gradients; update trainable parameters.
6. Evaluate a separate validation partition; select the checkpoint using a predeclared criterion.
7. Freeze preprocessing, checkpoint and classification threshold before opening test results.

### 3.7 Optional confidence-based early exit
After the full-step model is stable, add a classifier on prefix-averaged membrane states and evaluate the SEENN-I confidence criterion $\max(p_t, 1-p_t) \ge \eta$. A prefix-head loss or documented prefix calibration is needed; a head trained only on the final state may behave poorly at earlier steps. Tune $\eta$ and any consecutive-step stability requirement on validation. Compare the accuracy/step trade-off against fixed $T$. This optional reuse is attributed to SEENN and is not the main novelty claim.

---

## CHAPTER 4 — EXPERIMENTAL DESIGN AND PENDING RESULTS

### 4.1 Current experimental status
No dataset, trained checkpoint or per-sample predictions were supplied. At preparation time the processing environment had no detected GPU and no installed PyTorch. Consequently, this report contains no completed training run, no measured project accuracy and no experimental ROC or confusion-matrix image. The files are a research proposal and evaluation specification, not a substitute for implementation. Existing paper scores are separated in Chapter 2.

### 4.2 Dataset protocol and leakage prevention
Use FaceForensics++ for initial training and internal evaluation, following its official release and split files [4]. Dataset access requires approval and acceptance of terms. The initial external test is Celeb-DF with a documented version and official access; access has not been obtained for this project. DeeperForensics or DFDC is an optional later test, not a required simultaneous download.

Split video identifiers before frame extraction. Keep all frames, crops and augmented derivatives from a source video together, and audit related authentic/manipulated pairs, source/target identities and duplicates across partitions. Official benchmark splits and a stricter identity-disjoint audit answer different questions; preserve benchmark comparability and report residual identity overlap rather than assuming it is absent.

For the MVP, uniformly sample a fixed documented number of frames per video. Preserve per-video sampling balance and report both the frame count and video count per class. A frame-random train/test split is prohibited. Training-only sampling can balance classes; validation and test preserve their documented populations. Save crop coordinates, failure counts and video-to-frame mappings.

### 4.3 Initial configuration and compute assumptions

#### Table 4.1 contains proposed starting settings; none is reported as tuned or executed.

| Item | Initial setting / decision |
| :--- | :--- |
| **Input and encoder** | 224×224; lightweight pretrained shared encoder |
| **Optimization** | AdamW; LR $10^{-3}$ for new layers, $10^{-4}$ for unfrozen encoder |
| **Schedule** | Up to 40 epochs; cosine decay; validation checkpointing |
| **Warm-up** | Freeze encoder initially; unfreeze only if needed |
| **Batch and precision** | Start batch 8; use mixed precision when supported |
| **SNN** | 2 layers; width 64; $T=8$; $\beta=0.9$; $\theta=1.0$ |
| **Loss** | BCE; initially no activity penalty |
| **Seeds** | 2026, 2027, 2028 where feasible |
| **Hardware** | Plan for CUDA GPU; profile memory before selecting final batch |
| **Record** | Device, software versions, runtime, seeds, manifests, weights |

A GPU with roughly 8–12 GB VRAM is a planning target for the lightweight MVP, not a verified minimum. Required memory depends on graph autograd, unrolled steps, encoder and batch. CPU smoke tests can validate shapes and gradients but are not evidence of research-scale training. MAST’s four-GPU configuration must not be implied available here. Cache node features only for frozen-encoder experiments and disclose that augmentation and fine-tuning then differ.

### 4.4 Controlled model matrix
Table 4.2 separates the effects of spiking and attention. Compare models under identical data and preprocessing; report parameter counts and any unavoidable capacity difference.

#### Table 4.2. Controlled baseline and ablation matrix.

| ID | Model | Question answered |
| :--- | :--- | :--- |
| **B0** | Image-only encoder and classifier | Is graph modeling helpful? |
| **B1** | Adapted two-view ANN-GCN | Conventional graph reference |
| **B2** | Fixed-step SNN-GCN, no attention | Effect of spiking substitution |
| **B3** | SNN + ordinary graph attention | Effect of generic attention |
| **P** | **SNN + discrepancy-conditioned attention** | Increment beyond ordinary attention |
| **A1** | P with discrepancy vector zeroed | Parameter-matched discrepancy control |
| **A2** | P edge-only / residual-only | Are both views useful? |
| **A3** | P at $T=4,8,16$ | Accuracy/step sensitivity |
| **Optional** | P + attributed confidence early exit | Adaptive inference trade-off |

The primary endpoint is video-level ROC–AUC on the declared external test, if available, with internal validation used for all tuning. Also report internal results. Do not select the final architecture using external test scores. If external access fails, clearly restrict conclusions to within-dataset and held-out-manipulation evidence.

### 4.5 Mandatory metrics and score conventions
Fake is the positive class. A false positive is an authentic sample called fake; a false negative is a fake sample missed. Report raw counts with a clear class order. Use the same frozen threshold for all threshold-dependent test metrics for a given trained model. State whether F1 is positive-class binary, macro or weighted; binary fake-class F1 is the mandatory default.

#### Table 4.3. Required metric definitions.

| Metric | Definition / data needed |
| :--- | :--- |
| **Accuracy** | $(TP+TN)/(TP+TN+FP+FN)$ |
| **Precision** | $TP/(TP+FP)$ |
| **Recall / sensitivity** | $TP/(TP+FN)$ |
| **F1** | $2TP/(2TP+FP+FN)$ |
| **Specificity** | $TN/(TN+FP)$ |
| **ROC** | TPR vs FPR as score threshold varies |
| **ROC–AUC** | Area under ROC; requires continuous scores and both classes |
| **Average precision** | $\sum_n (R_n - R_{n-1}) P_n$; not trapezoidal PR–AUC |
| **Confusion matrix** | Rows actual [real,fake], columns predicted [real,fake] |

ROC and AUC must be computed from fake scores, not thresholded labels. A one-class evaluation subset has undefined ROC–AUC. Define zero-division handling for precision/F1 and disclose it. Classification thresholds may be selected to maximize validation F1 or satisfy a predeclared false-positive constraint; never select the best threshold on test labels [8].

$$\text{Confusion matrix} = \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$

For video-level scoring, average a fixed number of uniformly sampled frame probabilities per video and evaluate each video once. For frame-level reporting, keep video identifiers and avoid treating frames as statistically independent. Bootstrap videos, retaining all frames of each selected video for frame metrics. Use paired resampling when comparing model predictions on identical videos. Report seeds separately, mean and variability where practical; do not claim significance from a tiny difference without uncertainty.

### 4.6 Robustness, plots and training records
Evaluate compression and resizing using documented codec/quality settings and equivalent processing for both classes. Report the evaluated FF++ compression release by its exact identifier if used. Add blur only as a controlled condition. Avoid test-time preprocessing changes that improve only the proposed model. Held-out-manipulation tests should exclude the selected manipulation from training and validation, and clearly state how related identities are handled.

Final measured outputs must include: per-epoch training/validation loss; validation AUC or the checkpoint criterion; ROC with AUC; precision–recall curve and AP; raw and optionally normalized confusion matrices; per-class metrics; dataset/class support; threshold; per-video scores; and computational measurements. Attention heatmaps are exploratory visualizations, not proof of causal explanation. Save separate model checkpoints, optimizer state, configuration and epoch history.

#### Algorithm 4.1. Evaluation and metric generation.
1. Run the selected checkpoint on validation; save sample ID, video ID, label and fake probability.
2. Select the threshold on validation only and save the decision rule.
3. Generate test probabilities with evaluation mode, state resets and no augmentation leakage.
4. Verify labels, duplicate IDs, per-video label consistency and dataset split provenance.
5. Aggregate video scores under the predeclared rule; compute all mandatory metrics.
6. Export ROC/PR points, plots, confusion matrix, prediction CSV and run metadata.
7. Repeat declared seeds and video-group bootstrap; report uncertainty and failure cases.

### 4.7 Pending results — not yet measured

#### Table 4.4. The following cells are intentionally unfilled by evidence. They are not expected results or target claims.

| Model | Accuracy | Precision | Recall | F1 | ROC–AUC | AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 ANN-GCN** | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| **B2 SNN-GCN** | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| **B3 ordinary attention** | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| **P proposed attention** | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

Confusion matrix counts TP, TN, FP and FN: not measured. ROC curve: not available because no test probabilities exist. Training curves and checkpoint epoch: not available. Published accuracy, AUC or F1 cannot determine these missing quantities.

#### Table 4.5. Robustness and computation evidence still required.

| Condition / quantity | Status |
| :--- | :--- |
| **External-dataset ROC–AUC and F1** | Not measured |
| **Compression/resizing robustness** | Not measured |
| **Latency including preprocessing** | Not measured |
| **Trainable and total parameters** | Not measured |
| **Peak memory / spike activity** | Not measured |
| **Average steps under optional exit** | Not measured |
| **Hardware energy in joules** | Not measured; not inferred from spike count |

Do not add a new composite “accuracy score” to make the work look novel. Standard metrics plus error analysis are more defensible. For optional early exit, report exit coverage and error among exited samples. SNN operations and firing rates are computational proxies; conventional softmax attention, image encoding and graph construction still incur cost. Energy must be directly measured or explicitly labeled as a model-based estimate with assumptions, never claimed as hardware-measured savings.

### 4.8 Reproducibility, risks and milestones
Store a manifest containing dataset version, access terms, source-video splits, frame indices, detector weights, crop settings, normalization, encoder weights, graph topology, step count, neuron equations, surrogate, loss, optimizer, schedule, seeds, threshold, checkpoint hash and evaluation script version. Store raw predictions so every reported metric can be regenerated. Document failed runs and all deviations from the planned protocol.

Risks include weak forensic signal at landmarks, instability through time, ordinary attention explaining all gains, attention emphasizing compression, high conventional-encoder cost and unavailable dataset access. Mitigations are an image-only baseline, no-attention SNN control, fixed encoder warm-up, gradient clipping, limited T, explicit corruption tests and reduced-scope experiments. A converged matched baseline is preferable to stacking additional modules without attribution.

#### Table 4.6. Research implementation milestones.

| Stage | Deliverable and acceptance condition |
| :--- | :--- |
| **Semester 7** | Literature synthesis, approved method and TW; no fabricated results |
| **Build weeks 1–2** | Access approved; split audit; preprocessing and class counts |
| **Weeks 3–4** | Image baseline and ANN-GCN converge; saved predictions |
| **Weeks 5–6** | Fixed-step SNN learns; gradient and state-reset tests |
| **Weeks 7–8** | Ordinary and discrepancy attention controls completed |
| **Weeks 9–10** | External/compression tests; repeated seeds if feasible |
| **Weeks 11–12** | Metrics, uncertainty, ablations, demo and final report |

---

## CHAPTER 5 — CONCLUSION AND FUTURE SCOPE

### 5.1 Conclusion
This report defines a feasible modification of a multi-view facial deepfake detector using a spiking graph module and a precisely specified cross-view attention mechanism. It avoids the unsupported claim that generic attention or an SNN is new. The research question is whether discrepancy conditioning improves performance beyond a matched ordinary-attention SNN, and whether any benefit survives external or degraded evaluation. The complete set of requested detection metrics is specified, but actual values remain pending controlled training and testing.

### 5.2 Future scope
After the frame-level MVP is evaluated, future work may add temporal face graphs, learnable neuron dynamics, carefully calibrated early exit, or a lighter encoder. Each extension needs an isolated control. A full-video extension must distinguish video time from SNN simulation time and avoid future-frame leakage in any streaming claim. Neuromorphic deployment requires a hardware-compatible attention implementation and measured end-to-end energy, not only spike-rate estimates.

### 5.3 Submission and implementation readiness
Before semester-7 submission, obtain the supervisor’s name, confirm the official ID capitalization and institutional formatting requirements, review the unsigned approval page and update the contents fields. Before experimental submission, obtain authorized datasets, confirm GPU access, run baseline reproductions or label adaptations honestly, and replace pending results only with values regenerated from saved predictions. The final title may be retained, but the abstract and conclusion must be revised to match measured outcomes.

---

## REFERENCES

- **[1]** X. Xu, J. Chen, Y. Zhang, C. Li, A. K. Singh and Z. Lyu, “DeepFake Detection With Multi-View Fusion and Graph Convolutional Network,” *IEEE Transactions on Multimedia*, vol. 28, pp. 167–180, 2026. Online publication: 6 October 2025. DOI: [10.1109/TMM.2025.3618565](https://doi.org/10.1109/TMM.2025.3618565). Supplied full text: [VUahcgfK](https://www.genspark.ai/api/files/s/VUahcgfK).
- **[2]** M. Jang, Y. Yang, H.-S. Kim, M. Son, Y. Kim and C. Kim, “Detecting AI-Generated Videos with Spiking Neural Networks,” *arXiv:2605.05895v2*, 31 July 2026 (preprint). DOI/Link: [arXiv:2605.05895v2](https://arxiv.org/abs/2605.05895v2). Supplied PDF: [RZ7K5hOu](https://www.genspark.ai/api/files/s/RZ7K5hOu).
- **[3]** Y. Li, T. Geller, Y. Kim and P. Panda, “SEENN: Towards Temporal Spiking Early-Exit Neural Networks,” *Advances in Neural Information Processing Systems*, 2023. Supplied paper: [Edl80U4G](https://www.genspark.ai/api/files/s/Edl80U4G). Official code: [Intelligent-Computing-Lab-Yale/SEENN](https://github.com/Intelligent-Computing-Lab-Yale/SEENN).
- **[4]** A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies and M. Nießner, “FaceForensics++: Learning to Detect Manipulated Facial Images,” *ICCV*, 2019. Official dataset and access: [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics).
- **[5]** Intelligent Computing Lab, Yale, SEENN official implementation and evaluation instructions. [Intelligent-Computing-Lab-Yale/SEENN](https://github.com/Intelligent-Computing-Lab-Yale/SEENN). Inspected 7 November 2026.
- **[6]** snnTorch documentation, “Tutorial 6 — Surrogate Gradient Descent in a Convolutional SNN.” [snntorch.readthedocs.io](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html). Inspected 7 November 2026.
- **[7]** “A Deepfake Image Detection Method Based on a Multi-Graph Attention Network,” *Electronics*, vol. 14, no. 3, article 482, 2025. DOI: [10.3390/electronics14030482](https://www.mdpi.com/2079-9292/14/3/482). Cited for existing graph-attention prior art, not as a matched facial benchmark.
- **[8]** scikit-learn documentation, “Model evaluation: quantifying the quality of predictions.” [scikit-learn.org](https://scikit-learn.org/stable/modules/model_evaluation.html). Definitions distinguish average precision from trapezoidal precision–recall area.

### Source and version notes
The supplied MIF-Net PDF contains 14 pages; MAST v2 contains 35 pages including supplementary material; SEENN contains 16 pages. Architecture/training evidence was checked in MIF-Net Sections III–V, MAST Sections 4–5 and Appendices F–I, and SEENN Sections 3–4. The template structure and institutional image derive from the user-supplied `SampleReport.docx`. No claim is made of exhaustive independent verification of every paper result or of the global novelty of DC-SGAT. This report preserves original attribution and does not redistribute full source papers.
