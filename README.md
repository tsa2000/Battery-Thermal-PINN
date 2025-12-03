# Physics-Informed Battery Thermal Solver 🔋⚡

A deep learning framework for solving coupled thermal-fluid problems in lithium-ion battery cooling systems using **Physics-Informed Neural Networks (PINNs)**.

---

## 🎯 Motivation

Traditional CFD solvers (OpenFOAM, ANSYS Fluent) are computationally expensive for parametric studies in battery pack design. This PINN-based approach offers:
- **Fast forward solving** once trained
- **Mesh-free** operation (no complex pre-processing)
- **Differentiable** outputs for gradient-based optimization

This framework targets **real-world engineering applications** in electric vehicle thermal management systems.

---

## 🧠 Technical Overview

### Governing Equations

The solver enforces the following coupled PDEs:

**1. Continuity (Incompressible Flow)**
∇·u = 0

**2. Navier-Stokes (Momentum)**
ρ(u·∇)u = -∇p + μ∇²u

**3. Energy (with Heat Source)**
ρcp(u·∇T) = k∇²T + q̇

Where:
- `q̇` represents volumetric Joule heating from battery internal resistance
- Thermal properties are temperature-independent (valid for ΔT < 30K)

### Network Architecture

- **Input**: Spatial coordinates (x, y, z)
- **Output**: Flow field (u, v, w), pressure (p), temperature (T)
- **Activation**: Hyperbolic tangent (essential for smooth second derivatives)
- **Training**: Adam optimizer with adaptive loss weighting

---

## 📁 Project Structure

Battery-Thermal-PINN/
├── configs/                # YAML configuration files
│   └── battery_sim.yaml   # Default simulation parameters
├── src/
│   ├── physics.py         # PDE residual computation
│   ├── model.py           # Neural network architecture
│   ├── boundary.py        # Boundary condition handling
│   └── utils.py           # I/O and logging utilities
├── main.py                # Training script
└── requirements.txt       # Python dependencies

---

## 🚀 Quick Start

### Installation

Clone the repository
git clone https://github.com/yourusername/Battery-Thermal-PINN.git
cd Battery-Thermal-PINN
Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt

### Running a Simulation

Basic training
python main.py –config configs/battery_sim.yaml
Resume from checkpoint
python main.py –resume results/checkpoint_epoch_3000.pth

### Monitoring Training

Check the `results/training.log` file for detailed metrics, or watch the progress bar for real-time updates.

---

## ⚙️ Configuration

Edit `configs/battery_sim.yaml` to customize:
- Material properties (`rho`, `cp`, `k_th`)
- Heat generation rate (`heat_source`)
- Domain geometry (`x_min`, `x_max`, etc.)
- Boundary conditions (`T_inlet`, `u_inlet`)
- Training hyperparameters (`epochs`, `lr`, `batch_size`)

---

## 🧪 Validation Strategy

To ensure physical accuracy:
1. **Analytical benchmarks**: Compare with 1D heat conduction solutions
2. **CFD cross-validation**: Verify against OpenFOAM results for simple geometries
3. **Energy balance check**: Ensure heat generation equals heat removal at steady state

Target accuracy: **< 3% error** in peak temperature prediction

---

## 📊 Results

(This section will be populated after training)

Typical output includes:
- Temperature distribution at cell surface
- Velocity streamlines in cooling channels
- Pressure drop across the module
- Training convergence plots

---

## 🔬 Research Background

This work is part of a broader effort to integrate **AI-accelerated CFD** with **battery management systems (BMS)** for real-time thermal monitoring in EVs.

Key references:
- Raissi et al. (2019) - Original PINN framework
- Wang et al. (2022) - Adaptive loss balancing for multi-physics PINNs

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Implement adaptive mesh refinement
- [ ] Add support for transient (time-dependent) simulations
- [ ] Integrate with experimental validation data
- [ ] Extend to multi-phase cooling (liquid + phase change materials)

---

## 📄 License

MIT License - see `LICENSE` file for details.

---

## 👤 Author

**Thaer Abushawer**  
Mechanical Engineer | Computational Fluid Dynamics Researcher  
Focus: AI-Enhanced Thermal Management for Electric Vehicles

---

## 🙏 Acknowledgments

This project was developed as part of advanced propulsion systems research program.

Special thanks to the open-source community for PyTorch and scientific computing tools.

