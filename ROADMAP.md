# SDP-CROWN Automated Driving Verification Roadmap

This roadmap outlines the experimental design, verification test matrix, and long-term goals for certifying neural network-based steering models under ACDC physical weather bounds.

---

## 🎯 Active Project Objective: Scientific Verification Proof-of-Concept
The goal of this work is to mathematically quantify the effect of environmental disturbances (rain, fog, night, snow) on AI driving models using **SDP-CROWN**. By modeling these disturbances as linear pixel modifications (affine transformations) calibrated from the real-world **ACDC dataset**, we evaluate robustness *without running physical simulations*. We then validate these predictions against closed-loop simulation runs in **CARLA**.

---

## 🧪 Experimental Framework

### 1. Inputs & Physical Disturbance Bounds
We model weather perturbations using calibrated contrast and brightness bounds ($[\epsilon_{c,\min}, \epsilon_{c,\max}]$ and $[\epsilon_{b,\min}, \epsilon_{b,\max}]$) derived from the ACDC dataset. 

*   **Fog:** Mid-range contrast drop and brightness bias.
*   **Rain:** Light to moderate contrast drop and brightness bias (reflections, droplets).
*   **Night:** Heavy contrast drop and negative brightness bias (pitch-black/poor visibility).
*   **Snow:** Heavy contrast drop and high positive brightness bias (white snow accumulation).

### 2. AI Model Variants
For maximum transparency and control, we train compact steering networks under two paradigms:
1.  **Model 1 (Expert Clear):** Trained strictly under clear weather. Used to establish a baseline of vulnerability to unseen conditions.
2.  **Model 2 (Expert Mixed):** Trained across Clear, Fog, Rain, and Night. Used to establish if robust/multi-weather training translates to formal safety certificates.

### 3. Verification Test Matrix & Expected Outcomes

We evaluate models on a sequence of driving frames (using a steering deviation corridor of $\pm 0.1$ rad). The table below outlines the expected vs. current actual results:

| Model | Evaluation Environment | Expected Result | Current Actual (Town04 Highway, Cropped Input) |
| :--- | :--- | :--- | :--- |
| **Model 1 (Expert Clear)** | CARLA (Clear Simulation) | **PASS** (Low cross-track error) | **PASS** (Stable lane keeping) |
| **Model 1 (Expert Clear)** | CARLA (Fog/Rain/Night) | **FAIL** (Crashes expected) | **FAIL** (Stalls in Rain; Crashes in Fog/Night) |
| **Model 1 (Expert Clear)** | SDP-CROWN (Rain Bound) | **Light Pass** (Few % safe) | **100% Safe (10/10 frames)** |
| **Model 1 (Expert Clear)** | SDP-CROWN (Fog Bound) | **Light Pass** (Few % safe) | **100% Safe (10/10 frames)** |
| **Model 2 (Expert Mixed)** | CARLA (All Weathers) | **PASS** (No crashes) | **Improved** (Higher stability, though curves in rain remain a challenge) |

---

## 🔮 Future Roadmap (Out of Scope for Current Phase)

### Phase 1: High-Priority Physics Models
*   **Sun Glare & Nighttime High-Beams:** Implement parameterized Semantic Perturbation (SP) layers calibrated using the Flare7K/7K++ datasets.
*   **Localized Lens Blinding (Adversarial Patch):** Model mud, grime, and camera occlusion using WoodScape dataset intervals.

### Phase 2: Dynamic & Environmental Extensions
*   **Windshield Wipers & Rain Streaks:** Model localized spatial line occlusions that shift dynamically across sequential frames.
*   **Depth-Based Attenuation:** Integrate Koschmieder's Law with depth maps to decay visibility exponentially by distance.
*   **Vehicle Dynamics:** Verify stability under camera pitch and tilt variations caused by road bumps and vehicle payload changes.

### Phase 3: Engine & Tooling Optimizations
*   **VRAM Scaling Resolution:** Implement sub-graph partitioning to bypass quadratic memory cost of dense matrix-mode convolution bounds.
*   **Closed-Loop Reachability:** Map steering deviation bounds into kinematic vehicle models to certify closed-loop trajectory envelopes instead of static per-frame corridors.
