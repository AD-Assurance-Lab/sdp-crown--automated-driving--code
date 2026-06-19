# Trained AI Models & Verification Procedures

This directory houses the neural network weight checkpoints used for behavioral cloning and safety verification, along with documentation for testing them.

---

## 1. Model Architectures

We utilize two primary regression networks for end-to-end steering control:

### CarlaSteeringNet
*   **Architecture:** 3 Convolutional layers followed by 3 Fully-Connected layers with ReLU activations.
*   **Input Shape:** $3 \times 60 \times 80$ RGB image (preserves 4:3 aspect ratio).
*   **Output:** 1D regression output representing the steering control angle in range `[-1.0, 1.0]`.
*   **Design Rationale:** Designed specifically as a lightweight, verifier-friendly architecture. Verification scale is bounded by network parameter size; keeping the network under 5,000 ReLUs allows Semidefinite Programming (SDP) and CROWN bound propagation to run efficiently without running out of memory.

### MicroPilotNet (Reference/Debugging)
*   **Architecture:** 5 Convolutional layers followed by 4 Fully-Connected layers with ReLU activations.
*   **Input Shape:** $3 \times 37 \times 117$ RGB image (NVIDIA's cropped lane aspect ratio).
*   **Output:** 1D regression output representing the steering control angle in radians.

---

## 2. Model Checkpoints

The weight files stored in this folder are:

| Filename | Dataset | Purpose | Loss (Val MSE) |
| :--- | :--- | :--- | :--- |
| `carla_steering_net_clear.pth` | CARLA (Clear weather only) | Primary closed-loop lane following | `0.001218` |
| `carla_steering_net_mixed.pth` | CARLA (Clear, rain, fog, night) | Weather-resilient lane following | `0.004662` |
| `pilotnet_udacity.pth` | Udacity Lake/Jungle Track | Reference benchmark for offline verification | - |
| `mnist_*.pth` | MNIST | Classifier verification debugging | - |
| `cifar10_*.pth` | CIFAR-10 | Classifier verification debugging | - |

---

## 3. Closed-Loop Evaluation in CARLA

To verify that the trained controllers are valid and capable of keeping the vehicle within its lane, we run closed-loop testing in the CARLA simulator.

### Evaluation Protocol
*   The simulator is run in **synchronous mode** at 10 Hz.
*   The ego vehicle is spawned at **spawn point index 0** on `Town01`.
*   At each tick, the front hood camera frame is captured, resized to $80 \times 60$, normalized, and fed to `CarlaSteeringNet`.
*   The model-predicted steering command is applied to the vehicle.
*   Throttle and brake are controlled by the Traffic Manager to maintain a safe speed profile, while the vehicle is steered entirely by the model.
*   The script logs predicted steering versus nominal autopilot steering, generating a validation performance plot.

### Running the Evaluation
To evaluate the **clear-weather model** under clear weather:
```bash
./venv_sdp/bin/python tools/test_carla_model.py \
    --model-path models/carla_steering_net_clear.pth \
    --weather clear \
    --num-frames 1000 \
    --save-plot results/carla_closed_loop_clear.png \
    --save-csv results/carla_closed_loop_clear.csv
```

To evaluate the **mixed-weather model** under rain weather:
```bash
./venv_sdp/bin/python tools/test_carla_model.py \
    --model-path models/carla_steering_net_mixed.pth \
    --weather rain \
    --num-frames 1000 \
    --save-plot results/carla_closed_loop_mixed_rain.png \
    --save-csv results/carla_closed_loop_mixed_rain.csv
```

---

## 4. Offline CROWN Mathematical Verification

Formal safety verification is conducted offline by bounding steering deviations under parameterized semantic weather perturbations.

### Verification Protocol
*   Using `auto_LiRPA`, we compile the model into a bounded computational graph.
*   We inject a **Semantic Perturbation Layer** representing adverse weather conditions (modeled as contrast drop $\epsilon_c$ and brightness bias $\epsilon_b$ bounds).
*   The verifier computes the worst-case output steering bounds $[\theta_{\min}, \theta_{\max}]$ for each frame.
*   If the bounds stay within a safety corridor of $\pm 0.1$ radians around the nominal clear steering path, the frame is certified **SAFE**.

### Running the Verifier
To run mathematical verification for a weather condition using paper-calibrated bounds:
```bash
./venv_sdp/bin/python verify_steering.py \
    --weather rain \
    --eps_c_min -0.0279 \
    --eps_c_max 0.0 \
    --eps_b_min 0.0 \
    --eps_b_max 0.1003 \
    --num_frames 50 \
    --device cpu
```
*Note: Using `--device cpu` is recommended to avoid GPU memory limitations during dense matrix bounds propagation.*
