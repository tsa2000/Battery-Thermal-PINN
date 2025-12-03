import torch
import torch.nn as nn

class ThermalPINN(nn.Module):
    """
    Multi-layer perceptron for physics-informed learning.
    
    Architecture notes:
    - Uses hyperbolic tangent (tanh) activation to ensure smooth 
      second derivatives needed for diffusion terms
    - Xavier initialization for stable gradient flow
    - Deeper networks (4-6 hidden layers) generally perform better 
      for complex PDEs
    """
    
    def __init__(self, layers=[3, 64, 64, 64, 64, 5], activation='tanh'):
        super(ThermalPINN, self).__init__()
        
        self.depth = len(layers) - 1
        
        # Activation function selection
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            # Not recommended for this problem due to vanishing second derivatives
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layer by layer
        self.layers_list = nn.ModuleList()
        for i in range(self.depth):
            self.layers_list.append(
                nn.Linear(layers[i], layers[i+1])
            )
        
        # Initialize weights using Xavier (Glorot) scheme
        # This prevents gradient explosion/vanishing in deep networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for all linear layers"""
        for layer in self.layers_list:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, 3) containing spatial coordinates
            
        Returns:
            Output tensor of shape (N, 5) containing [u, v, w, p, T]
        """
        for i, layer in enumerate(self.layers_list):
            x = layer(x)
            # Apply activation to all layers except the output layer
            if i < self.depth - 1:
                x = self.activation(x)
        
        return x
