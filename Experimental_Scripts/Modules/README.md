### Module 1 — IMU fall detection with a tiny 1D‑CNN (foundations)
Intuition
- Start with a single, always-available, low-power modality: the IMU (accelerometer/gyroscope). You’ll learn to window raw signals and train a lightweight 1D‑CNN for binary classification (fall vs non‑fall).

Key equations
- Windowing: given continuous IMU stream a(t), form windows X ∈ R^{T×C} with length T and channels C (e.g., ax, ay, az, gx, gy, gz).
- Model fθ: logits z = fθ(X); probabilities p = σ(z) for binary classification.
- Loss (binary cross‑entropy): L = −[y log p + (1−y) log(1−p)].
- Convolution (1D): y[n] = ∑_{k=0}^{K−1} w[k] x[n−k] + b.

Mini‑example
- Use 2 s windows at 100 Hz (T=200), C=6 channels. Two Conv1D blocks (kernel size 5), global average pooling, and a sigmoid head.

Small assignment
- Preprocess: z‑score each channel per subject; build subject‑wise train/test split.
- Train the 1D‑CNN; report accuracy, F1, sensitivity, specificity. Plot a confusion matrix.
- Stretch: Replace global average pooling with temporal attention and compare metrics.

---

### Module 2 — Federated learning (FedAvg) on non‑IID clients
Intuition
- With FL, each device trains locally and shares only model updates. Non‑IID data (different daily routines, mobility aids) makes naive averaging unstable, but FedAvg is the baseline protocol to understand first.

Key equations
- Local empirical risk at client k: F_k(θ) = (1/|D_k|) ∑_{(x,y)∈D_k} ℓ(fθ(x), y).
- FedAvg local update: θ_k^{t+1} = θ^t − η ∑_{e=1}^{E} ∇θ F_k(θ) (E local epochs).
- Server aggregation: θ^{t+1} = ∑_{k∈S_t} (|D_k|/∑_{j∈S_t}|D_j|) θ_k^{t+1}.

Mini‑example
- Simulate 50 clients; give each a different fall:non‑fall ratio (label‑skew). Train 5 local epochs per round; sample 10 clients per round.

Small assignment
- Implement FedAvg with subject‑wise partitions as clients.
- Compare IID vs non‑IID partitions. Plot accuracy vs rounds and communication (MB).
- Stretch: add client sampling probability proportional to recent loss.

---

### Module 3 — Stabilizing FL: FedProx, control‑variates, and server momentum
Intuition
- Non‑IID updates can “pull” the global model in conflicting directions. Add guardrails so local models do not stray too far and the server updates don’t zig‑zag.

Key equations
- FedProx (proximal term): client k minimizes F_k(θ) + (μ/2) ||θ − θ^t||^2.
- SCAFFOLD‑style control variates: use client c_k and server c to correct drift
  - Client update: θ ← θ − η (∇F_k(θ) − c_k + c).
  - Server updates c from aggregated changes.
- Server momentum: θ^{t+1} = θ^t + β v^t − η g^t, with v^t = β v^{t−1} + g^t and g^t aggregated gradient.

Mini‑example
- Re‑use Module 2 setup; enable FedProx with μ ∈ {0, 0.001, 0.01}. Observe convergence stability under severe label‑skew.

Small assignment
- Run FedAvg vs FedProx (and, if you can, a control‑variates baseline).
- Report rounds‑to‑target‑F1 and final worst‑client F1.
- Stretch: tune μ per client based on its drift (||θ_k − θ||).

---

### Module 4 — Multimodal fusion: early, late, and residual (tie‑in to [2])
Intuition
- Combining IMU with a heavier sensor (e.g., RGB or depth) can boost accuracy, but sensors may be noisy or redundant. Residual/intermediate fusion lets the model learn when to trust which modality.

Key equations
- Early fusion: h = φ([ϕ_IMU(X_I), ϕ_RGB(X_R)]).
- Late fusion: p = α p_IMU + (1−α) p_RGB, α learned or fixed.
- Residual fusion (toy form): h = ϕ_IMU(X_I) + G(ϕ_RGB(X_R); γ), where G learns when and how much to add; often G = Wϕ_RGB + b with a gate.
- Gated residual: h = ϕ_IMU + σ(g([ϕ_IMU, ϕ_RGB])) ⊙ Wϕ_RGB.

Mini‑example
- Two small encoders: 1D‑CNN for IMU; 2D‑CNN for RGB frames (or a precomputed keypoint vector). Fuse with a gated residual block; classify with a shared head.
- This mirrors the “residual fusion” spirit in [2] (they add Pareto client selection too).

Small assignment
- Implement early vs late vs residual fusion; train with the same data.
- Evaluate under a “noisy RGB” test (add noise or dark frames): which fusion degrades least?
- Stretch: visualize learned gate values vs ambient light or motion intensity. [2]

---

### Module 5 — Missing modalities and an IMU‑driven gate (energy/latency aware)
Intuition
- Real deployments lose sensors (camera off, occlusions). Train the model to be robust when a modality is missing, and gate the heavy branch to save energy.

Key equations
- Modality dropout during training: sample a mask m ∈ {0,1}^M and zero out missing modalities; minimize E_m [ℓ(fθ(X ⊙ m), y)].
- Energy/latency budget model: expected cost C = C_light + p_trigger · C_heavy, where p_trigger is gate’s trigger probability.
- Gate as a classifier: p_trigger = σ(g(ϕ_IMU(X_I))); wake heavy branch if p_trigger > τ.

Mini‑example
- Train residual fusion with 30% probability of dropping RGB at training time. Deploy with a gate that triggers RGB only when IMU confidence is low or “fall‑like.”

Small assignment
- Measure accuracy with and without modality dropout when RGB is absent at test time.
- Estimate expected latency/energy before vs after gating (even simple timing/energy proxies are fine).
- Stretch: learn τ to meet a target energy budget (constrained optimization).

---

### Module 6 — Client selection with Pareto metrics (tie‑in to [2])
Intuition
- Some clients are noisy (bad labels), incomplete (missing sensors), or stale (old data). Selecting a Pareto‑optimal subset each round improves stability without extra communication.

Key equations
- Define a metric vector s_k = [loss_k, recency_k, completeness_k, diversity_k, size_k].
- Pareto dominance: s_a dominates s_b if s_a is no worse in all metrics and strictly better in at least one.
- Selection: choose a maximal non‑dominated set (Pareto front), then subsample to the round’s budget.

Mini‑example
- Compute per‑client loss, last‑seen timestamp, modality completeness ratio, feature diversity proxy (e.g., variance), and dataset size. Build the Pareto front and sample K clients.

Small assignment
- Add Pareto client selection to Module 4/5 FL training.
- Compare convergence, final F1, and variance across rounds vs random selection.
- Stretch: adapt sampling probabilities within the Pareto front proportional to a “need” score (e.g., under‑represented modalities). [2]

---

### Module 7 — Personalization: adapters and online ELM heads (tie‑in to [8])
Intuition
- People differ. Keep a global backbone for shared invariants, and let each client own a tiny personalized component that adapts quickly. Online ELM (Extreme Learning Machine) is a simple, fast head for on‑device updates.

Key equations
- Parameter split: θ = (θ_g, φ_i), with θ_g global and φ_i client‑specific.
- Personalized objective: minimize F_i(θ_g, φ_i) + λ||φ_i||^2 subject to secure aggregation of updates to θ_g only.
- ELM head (closed‑form ridge regression): given hidden features H ∈ R^{N×H}, β_i = (H^T H + λI)^{-1} H^T Y.
- Online update (recursive): β_{t+1} = β_t + P_t h_t (y_t − h_t^T β_t), with P_t maintained via Sherman–Morrison.

Mini‑example
- Freeze the multimodal trunk; learn a per‑client 1–2 layer adapter or an ELM head on hidden features. Update φ_i nightly on a few misclassified windows.

Small assignment
- Implement per‑client adapters (few thousand params) and compare to a shared‑only model on worst‑client F1.
- Implement an ELM head and show it recovers performance after a simulated gait change (concept drift).
- Stretch: restrict uploads so only θ_g updates are shared, keeping φ_i private. [8]

---

### Module 8 — Privacy in FL: DP‑SGD and secure aggregation (tie‑in to [6])
Intuition
- FL hides raw data, but model updates can leak. Two standard protections: secure aggregation (server only sees the sum of encrypted updates) and differential privacy (DP), which clips and adds noise so any single user’s contribution is bounded.

Key equations
- Per‑update clipping: ĝ_k = g_k · min(1, C/||g_k||_2).
- DP noise: g̃_k = ĝ_k + N(0, σ^2 C^2 I).
- Aggregation with DP: θ^{t+1} = θ^t − η (1/|S_t|) ∑_{k∈S_t} g̃_k.
- Privacy budget (high level): accountant composes per‑round (ε, δ) to a total over T rounds.

Mini‑example
- Choose clipping C and noise multiplier σ to target ε ≈ 5–10 over T rounds (exact ε depends on subsampling rate and accountant). Expect a small accuracy hit.

Small assignment
- Add clipping and DP noise to Module 6 training; sweep σ and report accuracy vs ε (use a standard moments/RDP accountant).
- Track worst‑client F1 under DP vs no‑DP.
- Stretch: combine DP with personalization (Module 7) and report trade‑offs. [6]

---

### Module 9 — Robustness: Byzantine‑resilient aggregation and drift checks
Intuition
- Some clients may be compromised or broken. Robust aggregation dampens outliers; simple drift checks detect when a client’s distribution shifts too far.

Key equations
- Coordinate‑wise median: θ^{t+1}_j = median({θ_{k,j}^{t+1}}_{k∈S_t}).
- Trimmed mean (trim α fraction at each end): average middle (1−2α) fraction per coordinate.
- Norm bounding: reject clients with ||θ_k^{t+1} − θ^t||_2 > τ.
- Update similarity: cosine similarity to the server update; filter if below threshold.

Mini‑example
- Inject 20% malicious clients that push the fall logit upward regardless of input. Compare FedAvg vs trimmed mean (α=0.1) vs median.

Small assignment
- Implement trimmed mean and norm bounds in Module 6 FL.
- Report global F1 and stability (variance across rounds) under 0%, 10%, 20% adversaries.
- Stretch: add a tiny clean proxy dataset at the server to perform sanity checks on the aggregated model (activation clustering).

---

### Module 10 — Communication efficiency: quantization, sparsification, “important weights” (tie‑in to [10], with topology from [9])
Intuition
- Uplink bandwidth is precious, especially for heavier branches (vision). Compress updates and send only what matters most; consider decentralized topologies when a server is a bottleneck.

Key equations
- Uniform 8‑bit quantization: Q(x) = round(x/s) with scale s = (max−min)/255; dequantize with s.
- Top‑k sparsification: send only the k largest‑magnitude entries; maintain error feedback e ← (g − topk(g)) + e for next round.
- Importance mask M: send θ ⊙ M where M_j ∈ {0,1} indicates “important” parameters (as in [10]).
- Bytes per round: B ≈ (#nonzeros) × (bits per value)/8 + index overhead.

Mini‑example
- For the RGB/keypoint branch, transmit only 200k “important” weights; keep IMU branch full‑precision. Compare accuracy vs bytes/round.

Small assignment
- Implement 8‑bit quantization and top‑k; measure accuracy vs compression ratio.
- Implement a simple importance mask (e.g., by gradient magnitude) for the heavy branch and target ~40% comm reduction with <1 point F1 loss. [10]
- Stretch: compare centralized (server) vs ring‑based decentralized aggregation on the same task. [9]

---

### Module 11 — Semi‑supervised FL and multi‑stage confirmation (tie‑in to [5,4,3])
Intuition
- Labels are scarce; leverage unlabeled streams via pseudo‑labels or consistency training. Operationally, a lightweight first stage can trigger an expensive second stage (vision or mmWave) to reduce false alarms.

Key equations
- Pseudo‑labeling: for unlabeled x, if max_c pθ(c|x) > τ, add (x, argmax p) to training set.
- Consistency loss: L_u = E_{x,ξ} ||pθ(x) − pθ(aug_ξ(x))||^2.
- Two‑stage decision: d = I(p_IMU > τ1) · I(p_RGB > τ2) for confirmation; or a learned AND with a small combiner network.

Mini‑example
- Stage 1: IMU model (Module 1) generates candidate falls. Stage 2: small vision model confirms. Use pseudo‑labels from confident non‑falls to improve Stage 1.

Small assignment
- Add pseudo‑labeling to Module 6 training on unlabeled IMU windows; measure gains vs τ.
- Implement a two‑stage cascade; report precision, recall, and time‑to‑alert vs single‑stage.
- Stretch: make Stage‑2 participation event‑driven and report duty‑cycle/energy savings. [5,4,3]

---

### Module 12 — Edge deployment and measurement (closing the loop)
Intuition
- Realistic claims require end‑to‑end latency and energy, including I/O and preprocessing. Quantize, fuse ops, and measure p50/p95 latency.

Key equations
- Total latency: T_total = T_IO + T_pre + T_inf + T_post.
- Energy per event: E = ∫_0^{T_total} P(t) dt ≈ P_avg · T_total (if P is stable).
- Budgeting: ensure T_total < 50 ms (MCU IMU path) or < 100 ms (Pi/Jetson multimodal path).

Mini‑example
- Export IMU path to int8 (TFLM/CMSIS‑NN if MCU, TensorRT if Pi/Jetson). Time the full pipeline with realistic windowing and post‑processing thresholds.

Small assignment
- Measure end‑to‑end latency for IMU‑only and gated multimodal paths; report p50/p95.
- If possible, measure energy/event with a power meter or on‑device telemetry.
- Stretch: optimize preprocessing (FFT/keypoints) to reduce T_pre by operator fusion.

---

How the modules map to the report
- Multimodal residual fusion and client selection: Modules 4, 6 build directly on [2].
- Personalization via online ELM/adapters: Module 7 is anchored in [8].
- Differential privacy in FL across wearable datasets: Module 8 follows [6].
- Communication‑efficient vision updates and topologies: Module 10 aligns with [10,9].
- Hierarchical and multi‑stage pipelines: Module 11 echoes [3,4].
- Earlier multimodal FL framing/context: [1] provides background for Module 4.