"""
TacticsAI Graph Neural Network
GNN architecture for football tactics prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class TacticsGNN(nn.Module):
    """
    Graph Neural Network for predicting tactical matchup outcomes.

    Takes two tactical graphs (one per team) and predicts:
    - Dominance score (which team has tactical advantage)
    - Confidence (how certain the prediction is)
    - Battle zones (which areas of the pitch are contested)

    Architecture:
    - GATv2 layers for attention-based message passing with edge features
    - Residual connections for stable training
    - Dual-graph encoding with comparison head
    """

    def __init__(
        self,
        node_features: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_battle_zones: int = 9,  # 3x3 grid of pitch zones
        edge_features: int = 6,
        use_edge_features: bool = True,
    ):
        """
        Initialize the TacticsGNN.

        Args:
            node_features: Number of input features per player node (default 12)
            hidden_dim: Hidden dimension size
            num_layers: Number of GCN layers
            num_heads: Number of attention heads in GAT
            dropout: Dropout probability
            num_battle_zones: Number of pitch zones for battle prediction
            edge_features: Number of edge features (default 6)
            use_edge_features: Whether to use edge features in message passing
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for TacticsGNN")
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_battle_zones = num_battle_zones
        self.edge_features = edge_features
        self.use_edge_features = use_edge_features
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)

        # GATv2 layers for attention-based message passing with edge features
        # GATv2Conv is an improved version that supports edge features
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_features if use_edge_features else None,
                    add_self_loops=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Graph-level pooling will produce hidden_dim * 2 (mean + max)
        pool_dim = hidden_dim * 2
        
        # Comparison head (takes concatenated team embeddings)
        comparison_input_dim = pool_dim * 2  # Two teams
        
        self.comparison_head = nn.Sequential(
            nn.Linear(comparison_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output heads
        self.dominance_head = nn.Linear(hidden_dim // 2, 1)
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
        self.battle_zones_head = nn.Linear(hidden_dim // 2, num_battle_zones)
    
    def encode_team(self, data: Data) -> torch.Tensor:
        """
        Encode a team's tactical graph into a fixed-size embedding.

        Args:
            data: PyG Data object with node features, edge structure, and edge features

        Returns:
            Team embedding tensor of shape (batch_size, pool_dim)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Get edge features if available and if we're using them
        edge_attr = None
        if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GATv2 layers with edge features and residual connections
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            # Pass edge features to GAT layer
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)  # ELU activation works well with attention
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # Residual connection

        # Graph-level pooling (combine mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_embedding = torch.cat([x_mean, x_max], dim=-1)

        return graph_embedding

    def forward(
        self,
        graph_a: Data,
        graph_b: Data,
    ) -> dict:
        """
        Forward pass comparing two team tactics.

        Args:
            graph_a: First team's tactical graph
            graph_b: Second team's tactical graph

        Returns:
            Dict with:
            - 'dominance': Overall tactical dominance (0-1)
            - 'confidence': Prediction confidence (0-1)
            - 'battle_zones': Attention-derived zone control (9 values, 0-1 each)
            - 'attention_a': Zone attention for team A (for interpretability)
            - 'attention_b': Zone attention for team B (for interpretability)
        """
        # Encode both teams
        embed_a = self.encode_team(graph_a)
        embed_b = self.encode_team(graph_b)

        # Concatenate for comparison
        combined = torch.cat([embed_a, embed_b], dim=-1)

        # Comparison head
        features = self.comparison_head(combined)

        # Output predictions
        dominance = torch.sigmoid(self.dominance_head(features)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(features)).squeeze(-1)

        # Get attention-based zone analysis
        attention_a = self.get_zone_attention(graph_a)
        attention_b = self.get_zone_attention(graph_b)

        # Battle zones from attention comparison
        # Each zone's value represents relative tactical control
        battle_zones = self._compare_zone_attention(attention_a, attention_b)

        return {
            "dominance": dominance,
            "confidence": confidence,
            "battle_zones": battle_zones,  # Now attention-derived (0-1 per zone)
            "attention_a": attention_a,  # For interpretability
            "attention_b": attention_b,  # For interpretability
        }
    
    def predict_dominance(self, graph_a: Data, graph_b: Data) -> float:
        """
        Simple dominance prediction for inference.

        Args:
            graph_a: First team's tactical graph
            graph_b: Second team's tactical graph

        Returns:
            Dominance score (0-1, where >0.5 means team A dominates)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(graph_a, graph_b)
            return output["dominance"].item()

    def get_attention_weights(self, data: Data) -> list:
        """
        Extract attention weights from each GAT layer for interpretability.

        This is useful for visualizing which player relationships the model
        finds important for tactical analysis.

        Args:
            data: PyG Data object representing a team's tactical graph

        Returns:
            List of attention weight tensors, one per GAT layer.
            Each tensor has shape [num_edges, num_heads] showing attention
            scores for each edge and attention head.
        """
        self.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index

            # Get edge features if available
            edge_attr = None
            if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = data.edge_attr

            # Input projection
            x = self.input_proj(x)
            x = F.relu(x)

            attention_weights = []

            # Pass through each GAT layer and collect attention weights
            for gat, norm in zip(self.gat_layers, self.layer_norms):
                residual = x

                # Get output with attention weights
                x, (edge_index_att, attention) = gat(
                    x, edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True
                )

                # Store attention weights
                attention_weights.append(attention)

                x = norm(x)
                x = F.elu(x)
                x = x + residual

            return attention_weights

    def get_zone_attention(self, data: Data) -> torch.Tensor:
        """
        Aggregate attention weights by pitch zone (3x3 grid).

        Divides pitch into 3x3 grid and computes attention density per zone.
        This shows where the team is focusing tactical attention.

        Args:
            data: PyG Data object representing a team's tactical graph

        Returns:
            Tensor of shape (9,) representing normalized attention in each zone
        """
        # Get player positions from node features (first 2 dims are x, y normalized)
        positions = data.x[:, :2] * 100  # Denormalize to 0-100 scale

        # Get edge features if available
        edge_attr = None
        if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr

        # Need to pass through the network to get attention from last layer
        # Use intermediate computation
        x = self.input_proj(data.x)
        x = F.relu(x)

        # Pass through all but last GAT layer
        for i, (gat, norm) in enumerate(zip(self.gat_layers[:-1], self.layer_norms[:-1])):
            residual = x
            x = gat(x, data.edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        # Get attention from last GAT layer
        _, (edge_index, attention) = self.gat_layers[-1](
            x,
            data.edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
        )

        # Average attention across all heads
        attention = attention.mean(dim=1)  # [num_edges] after averaging heads

        # Initialize zone attention
        zones = torch.zeros(9, device=data.x.device)

        # Aggregate attention by zone
        for i in range(data.x.shape[0]):  # For each node
            # Get position
            pos = positions[i]
            x_pos, y_pos = pos[0].item(), pos[1].item()

            # Calculate zone index (0-8) for 3x3 grid
            zone_x = int(min(x_pos // 33.34, 2))  # 0, 1, or 2
            zone_y = int(min(y_pos // 33.34, 2))  # 0, 1, or 2
            zone_idx = zone_x + zone_y * 3

            # Sum attention weights for edges where this node is the source
            node_mask = edge_index[0] == i
            if node_mask.any():
                node_attention = attention[node_mask].sum()
                zones[zone_idx] += node_attention

        # Normalize to sum to 1 (probability distribution over zones)
        zones = F.softmax(zones, dim=0)

        return zones

    def _compare_zone_attention(
        self, att_a: torch.Tensor, att_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compare zone attention between two teams.

        Returns battle zone scores where:
        - Values near 0.5 = balanced contest in that zone
        - Values > 0.5 = team A has tactical advantage
        - Values < 0.5 = team B has tactical advantage

        Args:
            att_a: Zone attention for team A (shape: 9)
            att_b: Zone attention for team B (shape: 9)

        Returns:
            Tensor of shape (9,) with zone comparison scores (0-1)
        """
        # Avoid division by zero
        total = att_a + att_b + 1e-8

        # Normalize to 0-1 scale
        zone_scores = att_a / total

        return zone_scores


class SimpleTacticsGNN(nn.Module):
    """
    Simplified GNN for testing and quick experiments.

    Uses fewer layers and parameters for faster training.
    Uses GATv2Conv for attention-based aggregation.
    """

    def __init__(
        self,
        node_features: int = 12,
        hidden_dim: int = 32,
        num_heads: int = 2,
        dropout: float = 0.1,
        edge_features: int = 6,
        use_edge_features: bool = True,
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required")

        self.dropout = dropout
        self.use_edge_features = use_edge_features

        # Two GAT layers
        self.gat1 = GATv2Conv(
            node_features,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_features if use_edge_features else None,
        )
        self.gat2 = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_features if use_edge_features else None,
        )

        # Output: 2 teams * hidden_dim * 2 (mean+max pool)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Get edge features if available
        edge_attr = None
        if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr

        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr=edge_attr))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return torch.cat([x_mean, x_max], dim=-1)

    def forward(self, graph_a: Data, graph_b: Data) -> dict:
        embed_a = self.encode(graph_a)
        embed_b = self.encode(graph_b)

        combined = torch.cat([embed_a, embed_b], dim=-1)
        dominance = torch.sigmoid(self.fc(combined)).squeeze(-1)

        return {"dominance": dominance}
