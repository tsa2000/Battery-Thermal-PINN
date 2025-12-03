import torch
from torch.autograd import grad

class BatteryPhysics:
    """
    Governs coupled thermal-fluid physics in battery packs.
    
    Combines incompressible Navier-Stokes with energy transport,
    accounting for internal heat generation (Joule heating from 
    electrochemical reactions and ohmic losses).
    
    Key assumptions:
    - Incompressible flow (valid for liquid cooling)
    - Constant thermophysical properties
    - Uniform volumetric heat source (homogenized cell model)
    """
    
    def __init__(self, cfg):
        # Unpack material properties
        self.rho = cfg['physics']['rho']
        self.mu = cfg['physics']['mu']
        self.cp = cfg['physics']['cp']
        self.k_th = cfg['physics']['k_th']
        
        # Volumetric heat generation rate
        # In real applications, this would vary with SOC and C-rate
        self.q_dot = cfg['physics']['heat_source']
    
    def compute_residuals(self, model, x):
        """
        Computes PDE residuals at collocation points.
        
        Args:
            model: Neural network
            x: Tensor of shape (N, 3) containing (x, y, z) coordinates
            
        Returns:
            Tuple of residuals: (continuity, mom_x, mom_y, mom_z, energy)
        """
        x.requires_grad_(True)
        predictions = model(x)
        
        # Unpack network outputs
        u = predictions[:, 0:1]  # x-velocity
        v = predictions[:, 1:2]  # y-velocity
        w = predictions[:, 2:3]  # z-velocity
        p = predictions[:, 3:4]  # pressure
        T = predictions[:, 4:5]  # temperature
        
        # --- First-order spatial derivatives ---
        def spatial_grad(output, inputs):
            """Helper to compute (∂/∂x, ∂/∂y, ∂/∂z)"""
            g = grad(output, inputs, 
                    grad_outputs=torch.ones_like(output),
                    create_graph=True, retain_graph=True)[0]
            return g[:, 0:1], g[:, 1:2], g[:, 2:3]
        
        u_x, u_y, u_z = spatial_grad(u, x)
        v_x, v_y, v_z = spatial_grad(v, x)
        w_x, w_y, w_z = spatial_grad(w, x)
        p_x, p_y, p_z = spatial_grad(p, x)
        T_x, T_y, T_z = spatial_grad(T, x)
        
        # --- Second-order derivatives (Laplacian) ---
        def laplacian(first_derivs, inputs):
            """Compute ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²"""
            dx, dy, dz = first_derivs
            
            dxx = grad(dx, inputs, grad_outputs=torch.ones_like(dx), 
                      create_graph=True)[0][:, 0:1]
            dyy = grad(dy, inputs, grad_outputs=torch.ones_like(dy), 
                      create_graph=True)[0][:, 1:2]
            dzz = grad(dz, inputs, grad_outputs=torch.ones_like(dz), 
                      create_graph=True)[0][:, 2:3]
            
            return dxx + dyy + dzz
        
        u_lapl = laplacian((u_x, u_y, u_z), x)
        v_lapl = laplacian((v_x, v_y, v_z), x)
        w_lapl = laplacian((w_x, w_y, w_z), x)
        T_lapl = laplacian((T_x, T_y, T_z), x)
        
        # ========================================
        # GOVERNING EQUATIONS
        # ========================================
        
        # 1. Continuity (mass conservation)
        # ∇·u = 0
        continuity_residual = u_x + v_y + w_z
        
        # 2. Momentum equations (Navier-Stokes)
        # ρ(u·∇)u = -∇p + μ∇²u
        
        # Convective terms
        conv_u = u*u_x + v*u_y + w*u_z
        conv_v = u*v_x + v*v_y + w*v_z
        conv_w = u*w_x + v*w_y + w*w_z
        
        momentum_x = self.rho * conv_u + p_x - self.mu * u_lapl
        momentum_y = self.rho * conv_v + p_y - self.mu * v_lapl
        momentum_z = self.rho * conv_w + p_z - self.mu * w_lapl
        
        # 3. Energy equation
        # ρ·cp·(u·∇T) = k∇²T + q̇
        
        convection = u*T_x + v*T_y + w*T_z
        diffusion = self.k_th * T_lapl
        
        energy_residual = self.rho * self.cp * convection - diffusion - self.q_dot
        
        return (continuity_residual, 
                momentum_x, momentum_y, momentum_z, 
                energy_residual)
