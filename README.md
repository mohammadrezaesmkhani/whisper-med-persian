# 🔪 Surgical-Whisper: Progressive Multi-Stage QLoRA for Low-Resource ASR

[![Model: Whisper-Medium](https://img.shields.io/badge/Model-Whisper--Medium%20(763M)-blue.svg)](https://huggingface.co/openai/whisper-medium)
[![Method: Surgical QLoRA](https://img.shields.io/badge/PEFT-Surgical%20QLoRA%20(NF4)-orange.svg)](https://github.com/huggingface/peft)
[![Language: Persian](https://img.shields.io/badge/Language-Persian%20(fa)-green.svg)](#)
[![Best WER: 10.0%](https://img.shields.io/badge/Best%20WER-10.0%25%20(CV)%20%7C%208.0%25%20(Mixed)-brightgreen.svg)](#experimental-benchmarks--ablation-results)

**Surgical-Whisper** introduces a progressive, layer-selective Parameter-Efficient Fine-Tuning (PEFT) framework for adapting **`openai/whisper-medium`** to low-resource and domain-specific **Persian Automatic Speech Recognition (ASR)**.

Instead of applying uniform Low-Rank Adaptation (LoRA) across all transformer blocks in a single training pass, we formulate a **Multi-Stage Surgical Curriculum (Freeze $\rightarrow$ Surgical Adapt $\rightarrow$ Merge $\rightarrow$ Full-Layer Re-LoRA $\rightarrow$ Anti-Forgetting Domain Replay)**, reducing validation Word Error Rate (WER) from **34.0% in baseline runs down to 25.0% in Stage 1, and 10.0% in Stage 2** on Mozilla Common Voice (and **8.0%–9.0%** under cross-domain data replay).

## 🔬 Theoretical & Mathematical Formulation

### 1. Acoustic Front-End Preservation (Skipping Layers $l \in [0, 8]$ in Stage 1)

Given an 80-channel log-Mel spectrogram $\mathbf{X} \in \mathbb{R}^{80 \times T}$, the Whisper encoder passes representations through a convolutional stem ($\text{Conv1d}_1 \rightarrow \text{GELU} \rightarrow \text{Conv1d}_2$) followed by $L = 24$ transformer blocks. 

In pre-trained speech transformers, shallow layers ($l \in [0, l_{\text{acoustic}}]$, where $l_{\text{acoustic}} = 8$) act as universal acoustic and phonetic feature extractors (capturing formants, pitch contours, and local spectro-temporal transitions), whereas deeper layers ($l \in [9, 23]$) encode language-specific morpho-syntactic and lexical semantics.

During initial cross-lingual transfer to Persian (`fa`), the randomly initialized LoRA adapters and unaligned target token distributions produce high initial cross-entropy loss $\mathcal{L}_{\text{CE}}$, generating high-variance backward gradients:

$$\left\lVert \frac{\partial \mathcal{L}_{\text{CE}}}{\partial \mathbf{h}^{(l)}} \right\rVert_2 \gg 0 \quad \text{at } t \to 0$$

If shallow layers $l \in [0, 8]$ are updated simultaneously with unaligned top layers, these noisy gradients distort the pre-trained acoustic manifold—a phenomenon known as **Acoustic Representation Collapse**. To prevent this, **Stage 1 (`Sergical-01`)** enforces a hard surgical mask over the layer index set $\mathcal{S}_1 = \lbrace 9, 10, \dots, 23 \rbrace$:

$$W^{(l)}_{\text{Stage 1}} = \begin{cases} W_0^{(l)} & \text{if } l \in \lbrace 0, 1, \dots, 8 \rbrace \quad (\text{Frozen Acoustic Stem: } \nabla W^{(l)} = \mathbf{0}) \\ W_0^{(l)} + \frac{\alpha_1}{r_1} B_1^{(l)} A_1^{(l)} & \text{if } l \in \lbrace 9, 10, \dots, 23 \rbrace \quad (\text{Surgical Semantic Adaptation}) \end{cases}$$

where $A_1^{(l)} \in \mathbb{R}^{r_1 \times d_{\text{in}}}$, $B_1^{(l)} \in \mathbb{R}^{d_{\text{out}} \times r_1}$, and $r_1 = 32, \alpha_1 = 64$. Once the deeper semantic layers align with Persian token representations (reaching **25.0% WER**), backpropagated gradients become well-conditioned for full-network adaptation.

### 2. Implicit Rank Expansion via Iterative Merge-and-Reinit (Stage 2 "Positive Shock")

In standard single-stage LoRA with rank $r$, the weight update $\Delta W = \frac{\alpha}{r} B A$ is strictly confined to a low-dimensional subspace:

$$\text{rank}(\Delta W) \le r$$

Increasing $r$ directly (e.g., from $32$ to $64$) doubles activation memory and adapter parameter overhead. Instead, **Stage 2 (`sergical-02`)** permanently fuses the converged Stage 1 adapters into the base weights, re-quantizes the merged backbone into 4-bit NormalFloat (`NF4`), and initializes a fresh LoRA set $(B_2^{(l)}, A_2^{(l)})$ across **all** layers $\mathcal{S}_2 = \lbrace 0, 1, \dots, 23 \rbrace$:

$$W_{\text{merged}}^{(l)} = \text{Quant}_{\text{NF4}}\left(W_0^{(l)} + \frac{\alpha_1}{r_1} B_1^{(l)} A_1^{(l)} \cdot \mathbb{I}(l \ge 9)\right)$$

$$W_{\text{Stage 2}}^{(l)} = W_{\text{merged}}^{(l)} + \frac{\alpha_2}{r_2} B_2^{(l)} A_2^{(l)}, \quad \forall l \in \lbrace 0, \dots, 23 \rbrace$$

By the subadditivity of matrix rank, the cumulative weight perturbation $\Delta W_{\text{total}}^{(l)} = \Delta W_1^{(l)} + \Delta W_2^{(l)}$ across layers $l \in [9, 23]$ spans an expanded optimization subspace:

$$\text{rank}\left(\Delta W_1^{(l)} + \Delta W_2^{(l)}\right) \le \text{rank}\left(\Delta W_1^{(l)}\right) + \text{rank}\left(\Delta W_2^{(l)}\right) = r_1 + r_2 = 64$$

**Why this creates a "Positive Shock" in optimization:**
* **Subspace Escape:** Re-initializing $B_2^{(l)} = \mathbf{0}$ and $A_2^{(l)} \sim \mathcal{N}(0, \sigma^2)$ changes the principal gradient directions, allowing the optimizer to escape the rank-32 local minimum of Stage 1.
* **Warm-Restart Dynamics:** Combining fresh adapters with a Cosine Annealing schedule ($\eta_{\max} = 5 \times 10^{-4}$) acts as a controlled warm restart while unlocked shallow layers ($l \in [0, 8]$) fine-tune Persian-specific co-articulation and phoneme boundaries, driving validation WER down from **25.0% to 10.0%**.

### 3. Exact Parameter Accounting per Whisper-Medium Layer

For `openai/whisper-medium` ($d_{\text{model}} = 1024$, $d_{\text{ff}} = 4096$, $L = 24$), applying LoRA of rank $r$ to the 6 linear projection modules $\mathcal{M} =$ {`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`} yields:

* **Per Encoder Layer** (1 Self-Attention block + FFN):
  $$P_{\text{enc}}(r) = 4 \times r(2 d_{\text{model}}) + 2 \times r(d_{\text{model}} + d_{\text{ff}}) = 8192r + 10240r = 18,432r$$
* **Per Decoder Layer** (Self-Attention + Cross-Attention + FFN):
  $$P_{\text{dec}}(r) = 8 \times r(2 d_{\text{model}}) + 2 \times r(d_{\text{model}} + d_{\text{ff}}) = 16384r + 10240r = 26,624r$$
* **Total per Layer Index $l$ (Encoder + Decoder pair):**
  $$P_{\text{pair}}(r) = P_{\text{enc}}(r) + P_{\text{dec}}(r) = 45,056r$$

At $r = 32$, each layer pair adds exactly $P_{\text{pair}}(32) = \mathbf{1,441,792}$ trainable parameters (and at $r = 16$, $P_{\text{pair}}(16) = \mathbf{720,896}$):
* **Stage 1 (`Exp 101`, 15 layers `9..23`, $r=32$):** $15 \times 1,441,792 = \mathbf{21,626,880}$ trainable parameters.
* **Stage 2 (`Exp 102`, 24 layers `0..23`, $r=32$):** $24 \times 1,441,792 = \mathbf{34,603,008}$ trainable parameters.
* **Anti-Forgetting (`Exp 203`, 6 layers `18..23`, $r=16$):** $6 \times 720,896 = \mathbf{4,325,376}$ trainable parameters.

### 4. Stability–Plasticity Control in Continual Domain Adaptation

When adapting a general Persian model ($D_{\text{Old}} = \text{Common Voice}$) to a specialized target domain ($D_{\text{New}} = \text{Mana}$), unconstrained updates cause **Catastrophic Forgetting** ($\mathcal{L}(W; D_{\text{Old}})$ diverges). We bound parameter drift $\lVert W - W_{\text{Stage 2}} \rVert_F$ via two mechanisms:
1. **Spectral & Depth Regularization (`Exp 203`):** Restricting updates to the top 6 layers ($l \in [18, 23]$), halving rank ($r=16, \alpha=16$), and reducing the learning rate ($\eta = 5 \times 10^{-5}$) limits the Frobenius norm of weight perturbations, preserving general speech features while achieving **10.0% WER** on the target domain.
2. **Experience Replay Mixing (`Exp Mx01`):** Minimizing the composite empirical risk over a mixed distribution $D_{\text{Mix}} = \lambda D_{\text{New}} \cup (1-\lambda) D_{\text{Old}}$:

$$\min_{\Theta_{\text{LoRA}}} \; \lambda \, \mathbb{E}_{(x,y) \sim D_{\text{New}}}[\mathcal{L}(x,y)] + (1 - \lambda) \, \mathbb{E}_{(x,y) \sim D_{\text{Old}}}[\mathcal{L}(x,y)]$$

induces positive backward transfer, reducing WER to **8.0% on Common Voice** and **9.0% on Mana**.

## 📊 Experimental Benchmarks & Ablation Results

### 1. Core Surgical Progression & Domain Adaptation (`whisperQlora` Benchmarks)

All experiments were conducted on `openai/whisper-medium` using **NVIDIA RTX 4090** GPUs.

| Exp ID | Strategy & Base Model | Dataset & Target Scope | Trainable Params | Train / Val WER |
| :--- | :--- | :--- | :---: | :---: |
| **1** | `Whisper_onFly` (`whisper-medium`) | `Creative_guy` — Full Dual ($r=32, \alpha=64$, LR `5e-4`, 12 ep) | 69,206,016 | 12.0% / 34.0% |
| **2** | `WhisperSchrodinger` (`whisper-medium`) | `Creative_guy` — All `0..23` ($r=16, \alpha=32$, LR `5e-5`, 15 ep) | 17,301,504 | 30.0% / 44.0% |
| **101** | **`Sergical-01` (`whisper-medium`)** | **`CV-01` — Skip `0..8`, Train `9..23` ($r=32, \alpha=64$, LR `1e-4`, 5 ep)** | **21,626,880** | **20.0% / 25.0%** |
| **102** | **`sergical-02` (`Merged 101`)** | **`CV-01` — All `0..23` Re-LoRA ($r=32, \alpha=64$, LR `5e-4`, 7 ep)** | **34,603,008** | **9.0% / 10.0%** |
| **201** | `sergical-03` (`Merged 101`) | `Mana` — Layers `15..23` ($r=32, \alpha=64$, LR `1e-4`, 8 ep) | 12,976,128 | 18.0% / 20.0% |
| **202** | `sergiacal-3_1` (`Merged 102`) | `Mana` — Layers `13..23` ($r=32, \alpha=64$, LR `1e-4`, 8 ep) | 15,859,712 | 10.0% / 11.3% |
| **203** | **`sergiacal-3_2` (`Merged 102`)** | **`Mana` — Top 6 Layers `18..23` ($r=16, \alpha=16$, LR `5e-5`, 7 ep)** | **4,325,376** | **11.38% / 10.0%** |
| **Mx01** | **`whisperMx01` (`Merged 102`)** | **`Full Mana + 50k CV` — All `0..23` ($r=16, \alpha=16$, LR `5e-5`, 7 ep)** | **17,301,504** | **9.0% / CV: 8.0%, Mana: 9.0%** |

* **Key Takeaways from Table 1:**
  * **Stage 1 (`Exp 101`):** Skipping layers `0..8` prevents acoustic representation collapse and drops validation WER from 34.0% to **25.0%** (16h on RTX 4090).
  * **Stage 2 (`Exp 102`):** Merging Stage 1 weights and applying full-depth QLoRA (`0..23`) triggers a massive optimization jump, cutting validation WER from 25.0% down to **10.0%** (22h on RTX 4090).
  * **Anti-Forgetting (`Exp 201–203` & `Mx01`):** While `Exp 201` and `202` adapt well to `Mana`, they exhibit forgetting on general `Common Voice` speech. `Exp 203` resolves this via top-6 layer surgical dampening (`r=16`, `LR=5e-5`), and `Exp Mx01` achieves **8.0% CV WER / 9.0% Mana WER** via replay mixing.

### 2. Multi-Phase Continual Learning & Data-Mix Ablation Matrix (`rezzProAsr`)

To systematically evaluate the trade-off between **Plasticity** (learning specialized `D_New` speech) and **Stability** (retaining general `D_Old` speech), we designed a 3-phase factorial ablation protocol:

#### Dataset Replay Ratios (`D_New` vs. `D_Old`)
| Data Mix ID | Specialized Data (`D_New`) | General Data (`D_Old`) | Total Samples | Replay Ratio (`New : Old`) |
| :--- | :---: | :---: | :---: | :---: |
| **`Data-Mix-1`** | $125 \times 25 = 3,000$ | $3,000$ | $6,000$ (`6k`) | **1 : 1** |
| **`Data-Mix-2`** | $125 \times 25 = 3,000$ | $10,000$ | $13,000$ (`13k`) | **1 : 3.3** |
| **`Data-Mix-3`** | $125 \times 25 = 3,000$ | $15,000$ | $18,000$ (`18k`) | **1 : 5** |
| **`Data-Mix-4`** | $125 \times 10 = 1,250$ | $1,250$ | $2,500$ (`2.5k`) | **1 : 1** |
| **`Data-Mix-5`** | $125 \times 15 = 1,875$ | $1,875$ | $3,750$ (`3.7k`) | **1 : 1** |
| **`Data-Mix-6`** | $125 \times 15 = 1,875$ | $3,750$ | $5,625$ (`5.6k`) | **1 : 3** |

#### 3-Phase Surgical Optimization Grid
| Phase & Exp IDs | Base Candidates & Data Mixes | Architectural Scope | Hyperparameters | Phase Winner |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** (`A-1..A-6`, `B-1..B-6`) | `Sergical203\|102` vs `Sergical801` on `Mix 1, 2, 3` | `Decoder_Only` vs `Encoder_Decoder` (`20..24`) | $r=16$, LR `5e-5` | **`B-4`**: `Mix-2` (1:3.3) + `Enc_Dec` |
| **Phase 2** (`C-1..C-8`) | `Best_Model_P1` (`B-4`) on `Data-Mix-2` | `Encoder_Decoder` (`20..24` vs `Only_24`) | $r \in \lbrace 4, 8, 16, 32 \rbrace$, LR `5e-5` | **`C-7`**: Rank $r=32$ on `20..24` |
| **Phase 3** (`D-1..D-3`) | `Best_Model_P2` (`C-7`) on `Data-Mix-2` | `Encoder_Decoder` (`20..24`) | $r=32$, LR $\in \lbrace \text{1e-5, 5e-5, 1e-4} \rbrace$ | **`D-3` (`FINAL_MODEL`)**: LR `1e-4` |

### 3. Baseline Selective Layer-Freezing Progression (Direct Unfreezing)

Before transitioning to QLoRA, we also benchmarked direct partial layer unfreezing (unfreezing Encoder layers `21..23` and Decoder layers `20..23` excluding `self_attn` — **88,174,592** trainable parameters / **11.54%** of total model capacity):

| Global Step | Training Loss | Validation Loss |
| :---: | :---: | :---: |
| **20** | 1.1629 | 0.8875 |
| **80** | 0.6849 | 0.6781 |
| **160** | 0.6088 | 0.6090 |
| **240** | 0.5601 | 0.5758 |
| **320** | **0.4613** | **0.5556** |

## 🗂️ Repository Structure

```text
Surgical-Whisper/
├── notebooks/
│   ├── 00_Selective_Layer_Freezing_Baseline.ipynb   # Direct top-layer unfreezing (11.54% trainable)
│   └── 02_Whisper_Surgical_Stage2_QLoRA.ipynb       # Stage 2 Continued QLoRA on merged Stage 1 model
├── src/
│   ├── surgical_stage1_and_merge.py                 # Stage 1: Skip layers 0..8 (r=32) + Merge weights
│   └── surgical_stage2_train.py                     # Stage 2: 4-bit NF4 load + Full-layer LoRA (r=32)
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

### 1. Installation
```bash
pip install torch torchvision torchaudio transformers datasets peft bitsandbytes accelerate evaluate jiwer librosa
```

### 2. Run Stage 1 (Surgical LoRA on Layers 9..23 + Weight Merging)
```bash
python src/surgical_stage1_and_merge.py
```

### 3. Run Stage 2 (4-Bit NF4 Continued QLoRA on All Layers 0..23)
```bash
python src/surgical_stage2_train.py
```

## 📜 Citation & Author

Developed by **Mohammad Reza Esmkhani** — Senior AI Engineer & Researcher specializing in Deep Learning, Vision-Language/Speech Models, and Parameter-Efficient Fine-Tuning (PEFT).
