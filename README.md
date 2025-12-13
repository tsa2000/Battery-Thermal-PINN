# 🔋 Battery Thermal PINN: EV Thermal Management

**Physics-Informed Neural Networks for coupled thermal-fluid simulation in lithium-ion battery cooling systems**

---

## 🎯 Motivation

Traditional CFD solvers (OpenFOAM, ANSYS Fluent) require hours per simulation, limiting parametric design studies for battery packs. This PINN framework enables:
- **Fast predictions** after training (milliseconds vs hours)
- **Mesh-free operation** (eliminates complex preprocessing)
- **Differentiable outputs** for gradient-based optimization

Target application: Real-time thermal management in electric vehicles.

---

## 🧠 Physics Implementation

### Governing Equations

**Continuity (Incompressible Flow)**

∇·u = 0

**Navier-Stokes (Momentum)**

ρ(u·∇)u = -∇p + μ∇²u

**Energy (with Heat Source)**

ρc_p(u·∇T) = k∇²T + q̇

Where `q̇` represents volumetric Joule heating from battery internal resistance.

### Network Architecture

- **Input**: Spatial coordinates (x, y, z)
- **Output**: Velocity field (u, v, w), pressure (p), temperature (T)
- **Activation**: Hyperbolic tangent (smooth second derivatives)
- **Training**: Adam optimizer with adaptive physics loss weighting

---

## 📁 Project Structure

Battery-Thermal-PINN/
├── configs/
│   └── battery_sim.yaml   # Simulation parameters
├── src/
│   ├── physics.py         # PDE residual computation
│   ├── model.py           # Neural network architecture
│   ├── boundary.py        # Boundary conditions
│   └── utils.py           # Utilities
├── main.py                # Training script
└── requirements.txt

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/tsa2000/Battery-Thermal-PINN.git
cd Battery-Thermal-PINN

2️⃣ Install Dependencies

Locally:

pip install -r requirements.txt

Or in Google Colab:

!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install numpy matplotlib tqdm pyyaml

3️⃣ Training

# Start training with default configuration
python main.py --config configs/battery_sim.yaml

# Resume from a saved checkpoint
python main.py --resume results/checkpoint_epoch_3000.pth

4️⃣ Inference / Prediction

import torch
from src.model import ThermalPINN

# Load trained model
model = ThermalPINN(layers=[3, 64, 64, 64, 64, 5], activation='tanh')
model.load_state_dict(torch.load('results/battery_thermal_final.pth', map_location='cpu'))
model.eval()

# Predict at a point
x = torch.tensor([[0.05, 0.025, 0.005]], dtype=torch.float32)  # (x, y, z)
prediction = model(x)
print("Prediction [u, v, w, p, T]:", prediction)


---

## ⚙️ Configuration

Edit configs/battery_sim.yaml to customize:
	•	Material properties: rho, cp, k_th
	•	Heat generation: heat_source
	•	Domain geometry: x_min, x_max, etc.
	•	Boundary conditions: T_inlet, u_inlet
	•	Training: epochs, lr, batch_size

⸻

##🔬 Validation Approach
	1.	Analytical benchmarks: 1D heat conduction solutions
	2.	CFD cross-validation: OpenFOAM comparison for simple geometries
	3.	Energy balance: Heat generation vs removal verification

⸻

## 📊 Features

Implemented ✅
	•	Full PDE residual computation
	•	Neural network with physics-informed loss
	•	Boundary condition enforcement
	•	Training loop with checkpoints

In Progress 🔄
	•	Benchmark dataset generation
	•	Performance comparison metrics
	•	Hyperparameter optimization

⸻

## 🎓 Research Context

Exploring AI-accelerated CFD integration with battery management systems (BMS) for real-time thermal monitoring in EVs.

Key references:
	•	Raissi et al. (2019): Physics-informed neural networks
	•	Wang et al. (2022): Adaptive loss balancing for multi-physics PINNs

⸻

## 🛠️ Future Extensions
	•	Transient (time-dependent) simulations
	•	Multi-phase cooling systems
	•	Adaptive mesh refinement
	•	Turbulence modeling

⸻

## 📄 License

MIT License - see LICENSE file for details.

⸻

## 👤 Author

Thaer Abushawer
Mechanical Engineer | Energetics
Focus: AI-Enhanced Computational Methods for Thermal Systems
Thaer199@gmail.com￼

⸻

## 🙏 Acknowledgments

Developed as part of research in advanced thermal management systems for electric vehicles.

Built with PyTorch and the scientific Python ecosystem.
