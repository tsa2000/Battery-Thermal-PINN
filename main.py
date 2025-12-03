import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Local imports
from src.utils import load_config, setup_logger, DomainSampler, save_checkpoint
from src.physics import BatteryPhysics
from src.model import ThermalPINN
from src.boundary import BoundaryConditions

def main():
    parser = argparse.ArgumentParser(
        description="Physics-Informed Neural Network for Battery Thermal Management"
    )
    parser.add_argument('--config', type=str, default='configs/battery_sim.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    args = parser.parse_args()
    
    # ========================================
    # 1. SETUP ENVIRONMENT
    # ========================================
    cfg = load_config(args.config)
    logger = setup_logger(cfg['train']['save_dir'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    
    # ========================================
    # 2. INITIALIZE COMPONENTS
    # ========================================
    physics_engine = BatteryPhysics(cfg)
    boundary_handler = BoundaryConditions(cfg)
    domain_sampler = DomainSampler(cfg)
    
    model = ThermalPINN(
        layers=cfg['model']['layers'],
        activation=cfg['model']['activation']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Loss weights
    w = cfg['train']['weights']
    
    # ========================================
    # 3. TRAINING LOOP
    # ========================================
    epochs = cfg['train']['epochs']
    batch_size = cfg['train']['batch_size']
    log_interval = cfg['train']['log_interval']
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Batch size: {batch_size}, Learning rate: {cfg['train']['lr']}")
    
    pbar = tqdm(range(start_epoch, epochs + 1), desc="Training Progress")
    
    for epoch in pbar:
        model.train()
        
        # Sample collocation points from domain interior
        interior_pts = domain_sampler.sample_interior(batch_size)
        x_interior = torch.tensor(interior_pts, dtype=torch.float32, 
                                 device=device, requires_grad=True)
        
        # Compute PDE residuals
        cont, mx, my, mz, energy = physics_engine.compute_residuals(model, x_interior)
        
        # Physics loss (L2 norm of residuals)
        loss_continuity = torch.mean(cont**2)
        loss_momentum = torch.mean(mx**2) + torch.mean(my**2) + torch.mean(mz**2)
        loss_energy = torch.mean(energy**2)
        
        # Boundary condition loss
        loss_bc = boundary_handler.compute_bc_loss(model, device)
        
        # Total weighted loss
        total_loss = (w['continuity'] * loss_continuity +
                     w['momentum'] * loss_momentum +
                     w['energy'] * loss_energy +
                     w['boundary'] * loss_bc)
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:05d} | "
                f"Total: {total_loss.item():.3e} | "
                f"Cont: {loss_continuity.item():.3e} | "
                f"Mom: {loss_momentum.item():.3e} | "
                f"Energy: {loss_energy.item():.3e} | "
                f"BC: {loss_bc.item():.3e}"
            )
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.2e}",
                'Energy': f"{loss_energy.item():.2e}"
            })
        
        # Save checkpoint every 1000 epochs
        if epoch % 1000 == 0:
            checkpoint_path = Path(cfg['train']['save_dir']) / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, total_loss.item(), checkpoint_path)
    
    # ========================================
    # 4. SAVE FINAL MODEL
    # ========================================
    final_model_path = Path(cfg['train']['save_dir']) / "battery_thermal_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training complete. Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
