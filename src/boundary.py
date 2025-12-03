import torch
import numpy as np

class BoundaryConditions:
    """
    Handles boundary condition enforcement for the battery domain.
    
    BCs are enforced weakly through loss terms rather than hard constraints.
    This approach is more flexible for complex geometries.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Domain boundaries
        self.x_min = cfg['domain']['x_min']
        self.x_max = cfg['domain']['x_max']
        self.y_min = cfg['domain']['y_min']
        self.y_max = cfg['domain']['y_max']
        self.z_min = cfg['domain']['z_min']
        self.z_max = cfg['domain']['z_max']
        
        # Thermal BCs
        self.T_inlet = cfg['boundary_conditions']['T_inlet']
        self.T_wall = cfg['boundary_conditions']['T_wall']
        
        # Velocity BCs
        self.u_inlet = cfg['boundary_conditions']['u_inlet']
    
    def sample_boundary_points(self, n_points_per_face=256):
        """
        Samples points on all six faces of the rectangular domain.
        
        Returns:
            Dictionary with keys: 'inlet', 'outlet', 'walls'
        """
        points = {}
        
        # Inlet face (x = x_min)
        y_inlet = np.random.uniform(self.y_min, self.y_max, (n_points_per_face, 1))
        z_inlet = np.random.uniform(self.z_min, self.z_max, (n_points_per_face, 1))
        x_inlet = np.full_like(y_inlet, self.x_min)
        points['inlet'] = np.hstack([x_inlet, y_inlet, z_inlet]).astype(np.float32)
        
        # Outlet face (x = x_max)
        y_outlet = np.random.uniform(self.y_min, self.y_max, (n_points_per_face, 1))
        z_outlet = np.random.uniform(self.z_min, self.z_max, (n_points_per_face, 1))
        x_outlet = np.full_like(y_outlet, self.x_max)
        points['outlet'] = np.hstack([x_outlet, y_outlet, z_outlet]).astype(np.float32)
        
        # Wall faces (y_min, y_max, z_min, z_max) - combined for simplicity
        n_wall = n_points_per_face // 4
        wall_pts = []
        
        # Bottom wall (y = y_min)
        x_w = np.random.uniform(self.x_min, self.x_max, (n_wall, 1))
        z_w = np.random.uniform(self.z_min, self.z_max, (n_wall, 1))
        y_w = np.full_like(x_w, self.y_min)
        wall_pts.append(np.hstack([x_w, y_w, z_w]))
        
        # Top wall (y = y_max)
        x_w = np.random.uniform(self.x_min, self.x_max, (n_wall, 1))
        z_w = np.random.uniform(self.z_min, self.z_max, (n_wall, 1))
        y_w = np.full_like(x_w, self.y_max)
        wall_pts.append(np.hstack([x_w, y_w, z_w]))
        
        # Front wall (z = z_min)
        x_w = np.random.uniform(self.x_min, self.x_max, (n_wall, 1))
        y_w = np.random.uniform(self.y_min, self.y_max, (n_wall, 1))
        z_w = np.full_like(x_w, self.z_min)
        wall_pts.append(np.hstack([x_w, y_w, z_w]))
        
        # Back wall (z = z_max)
        x_w = np.random.uniform(self.x_min, self.x_max, (n_wall, 1))
        y_w = np.random.uniform(self.y_min, self.y_max, (n_wall, 1))
        z_w = np.full_like(x_w, self.z_max)
        wall_pts.append(np.hstack([x_w, y_w, z_w]))
        
        points['walls'] = np.vstack(wall_pts).astype(np.float32)
        
        return points
    
    def compute_bc_loss(self, model, device):
        """
        Computes boundary condition violation loss.
        
        Returns:
            Scalar tensor representing BC loss
        """
        bc_points = self.sample_boundary_points()
        
        # --- Inlet BC ---
        x_inlet = torch.tensor(bc_points['inlet'], device=device, requires_grad=False)
        pred_inlet = model(x_inlet)
        
        # Enforce: u = u_inlet, v = 0, w = 0, T = T_inlet
        loss_inlet_u = torch.mean((pred_inlet[:, 0] - self.u_inlet)**2)
        loss_inlet_v = torch.mean(pred_inlet[:, 1]**2)
        loss_inlet_w = torch.mean(pred_inlet[:, 2]**2)
        loss_inlet_T = torch.mean((pred_inlet[:, 4] - self.T_inlet)**2)
        
        # --- Wall BC ---
        x_wall = torch.tensor(bc_points['walls'], device=device, requires_grad=False)
        pred_wall = model(x_wall)
        
        # Enforce: no-slip (u = v = w = 0) and fixed temperature
        loss_wall_u = torch.mean(pred_wall[:, 0]**2)
        loss_wall_v = torch.mean(pred_wall[:, 1]**2)
        loss_wall_w = torch.mean(pred_wall[:, 2]**2)
        loss_wall_T = torch.mean((pred_wall[:, 4] - self.T_wall)**2)
        
        # --- Outlet BC ---
        # Typically: zero gradient (Neumann) - approximated by weak enforcement
        # For simplicity, we don't strongly enforce outlet BCs here
        
        total_bc_loss = (loss_inlet_u + loss_inlet_v + loss_inlet_w + loss_inlet_T +
                        loss_wall_u + loss_wall_v + loss_wall_w + loss_wall_T)
        
        return total_bc_loss
