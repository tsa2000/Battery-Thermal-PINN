import yaml
import logging
import numpy as np
from pathlib import Path

def load_config(config_path):
    """Loads YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logger(save_dir):
    """
    Configures logging for training monitoring.
    Outputs to both console and file.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{save_dir}/training.log", mode='w')
        ]
    )
    
    return logging.getLogger(__name__)

class DomainSampler:
    """
    Handles sampling of collocation points from the computational domain.
    
    Future extension: Import mesh from .vtk/.msh files using meshio
    """
    
    def __init__(self, cfg, mesh_path=None):
        self.cfg = cfg
        self.mesh_path = mesh_path
        
        # Domain boundaries
        self.x_bounds = [cfg['domain']['x_min'], cfg['domain']['x_max']]
        self.y_bounds = [cfg['domain']['y_min'], cfg['domain']['y_max']]
        self.z_bounds = [cfg['domain']['z_min'], cfg['domain']['z_max']]
    
    def sample_interior(self, n_points=1000):
        """
        Uniform random sampling within the domain.
        
        For more complex geometries, this should be replaced with 
        importance sampling or mesh-based point selection.
        """
        x = np.random.uniform(*self.x_bounds, (n_points, 1))
        y = np.random.uniform(*self.y_bounds, (n_points, 1))
        z = np.random.uniform(*self.z_bounds, (n_points, 1))
        
        return np.hstack([x, y, z]).astype(np.float32)

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Saves model checkpoint with training state"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
