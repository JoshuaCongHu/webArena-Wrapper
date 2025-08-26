import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: torch_geometric is optional, fallback to simple graph convolution
try:
    from torch_geometric.nn import GCNConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not found, using fallback graph convolution")


class SimpleGraphConv(nn.Module):
    """Fallback graph convolution when torch_geometric is not available"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # Simple graph convolution: A * X * W
        # x: [num_nodes, in_features]
        # adj_matrix: [num_nodes, num_nodes]
        x = self.linear(x)
        return torch.matmul(adj_matrix, x)


class OrchestratorPolicy(nn.Module):
    """Generates DAG decomposition and agent assignments"""
    
    def __init__(self, 
                 obs_dim: int = 512,
                 hidden_dim: int = 256,
                 max_nodes: int = 10,
                 num_agents: int = 4,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.num_agents = num_agents
        
        # Observation encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=obs_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.obs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Graph state encoder (GNN or fallback)
        if HAS_TORCH_GEOMETRIC:
            self.graph_conv1 = GCNConv(hidden_dim, hidden_dim)
            self.graph_conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.graph_conv1 = SimpleGraphConv(hidden_dim, hidden_dim)
            self.graph_conv2 = SimpleGraphConv(hidden_dim, hidden_dim)
        
        # Graph state projection
        self.graph_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Feature combination
        self.feature_combiner = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # DAG generation head
        self.dag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * max_nodes)
        )
        
        # Agent assignment head
        self.agent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_nodes * num_agents)
        )
        
        # Task difficulty estimation head
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_nodes)  # Difficulty per node
        )
        
        # Initialize parameters
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, obs: torch.Tensor, 
                graph_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate DAG and agent assignments
        
        Args:
            obs: Observation tensor [batch_size, seq_len, obs_dim] or [batch_size, obs_dim]
            graph_state: Graph state tensor [batch_size, num_nodes, hidden_dim]
        
        Returns:
            adj_matrix: DAG adjacency matrix [batch_size, max_nodes, max_nodes]
            agent_assignments: Agent assignment probabilities [batch_size, max_nodes, num_agents]
            node_difficulties: Estimated difficulty per node [batch_size, max_nodes]
        """
        batch_size = obs.shape[0]
        
        # Handle different observation shapes
        if obs.dim() == 2:  # [batch_size, obs_dim]
            obs = obs.unsqueeze(1)  # [batch_size, 1, obs_dim]
        
        # Encode observation with Transformer
        obs_encoded = self.obs_encoder(obs)  # [batch_size, seq_len, obs_dim]
        obs_encoded = obs_encoded.mean(dim=1)  # Global average pooling [batch_size, obs_dim]
        
        # Encode graph state
        if graph_state is not None:
            # Project to hidden dimension
            graph_features = self.graph_proj(graph_state)  # [batch_size, num_nodes, hidden_dim]
            
            # Apply graph convolution
            graph_encoded_list = []
            for b in range(batch_size):
                # Create a simple adjacency matrix for current graph
                adj_matrix = self._create_adjacency_matrix(graph_features[b])
                
                if HAS_TORCH_GEOMETRIC:
                    edge_index = self._adj_to_edge_index(adj_matrix)
                    x = self.graph_conv1(graph_features[b], edge_index)
                    x = F.relu(x)
                    x = self.graph_conv2(x, edge_index)
                else:
                    x = self.graph_conv1(graph_features[b], adj_matrix)
                    x = F.relu(x)
                    x = self.graph_conv2(x, adj_matrix)
                
                # Global pooling
                graph_encoded = x.mean(dim=0)  # [hidden_dim]
                graph_encoded_list.append(graph_encoded)
            
            graph_encoded = torch.stack(graph_encoded_list)  # [batch_size, hidden_dim]
        else:
            # If no graph state, use zeros
            graph_encoded = torch.zeros(batch_size, self.hidden_dim, device=obs.device)
        
        # Combine features
        combined_features = torch.cat([obs_encoded, graph_encoded], dim=-1)
        combined_features = self.feature_combiner(combined_features)
        
        # Generate DAG adjacency matrix
        adj_logits = self.dag_head(combined_features)
        adj_matrix = adj_logits.view(batch_size, self.max_nodes, self.max_nodes)
        
        # Enforce DAG constraints (upper triangular + sigmoid)
        mask = torch.triu(torch.ones_like(adj_matrix[0]), diagonal=1).unsqueeze(0).expand_as(adj_matrix)
        adj_matrix = adj_matrix * mask
        adj_matrix = torch.sigmoid(adj_matrix)
        
        # Generate agent assignments
        agent_logits = self.agent_head(combined_features)
        agent_assignments = agent_logits.view(batch_size, self.max_nodes, self.num_agents)
        agent_assignments = F.softmax(agent_assignments, dim=-1)
        
        # Estimate node difficulties
        difficulty_logits = self.difficulty_head(combined_features)
        node_difficulties = torch.sigmoid(difficulty_logits)  # [batch_size, max_nodes]
        
        return adj_matrix, agent_assignments, node_difficulties
    
    def _create_adjacency_matrix(self, node_features: torch.Tensor) -> torch.Tensor:
        """Create a simple adjacency matrix from node features"""
        num_nodes = node_features.shape[0]
        
        # Compute pairwise similarities
        similarities = torch.matmul(node_features, node_features.t())
        similarities = torch.sigmoid(similarities)
        
        # Create adjacency matrix (connect nodes with high similarity)
        threshold = 0.5
        adj_matrix = (similarities > threshold).float()
        
        # Remove self-loops
        adj_matrix.fill_diagonal_(0)
        
        # Add identity for self-connections in GCN
        adj_matrix += torch.eye(num_nodes, device=adj_matrix.device)
        
        return adj_matrix
    
    def _adj_to_edge_index(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Convert adjacency matrix to edge index format for torch_geometric"""
        edge_indices = torch.nonzero(adj_matrix, as_tuple=True)
        edge_index = torch.stack(edge_indices, dim=0)
        return edge_index
    
    def sample_dag(self, adj_matrix: torch.Tensor, threshold: float = 0.5) -> nx.DiGraph:
        """Sample a DAG from the adjacency matrix"""
        adj = adj_matrix.detach().cpu().numpy()
        
        # Apply threshold
        adj_binary = (adj > threshold).astype(float)
        
        # Create NetworkX graph
        G = nx.DiGraph()
        n_nodes = adj.shape[0]
        
        # Add nodes
        for i in range(n_nodes):
            if adj_binary[i].sum() > 0 or adj_binary[:, i].sum() > 0:  # Only add nodes with connections
                G.add_node(i)
        
        # Add edges
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_binary[i, j] > 0:
                    G.add_edge(i, j, weight=adj[i, j])
        
        # Verify DAG property
        if not nx.is_directed_acyclic_graph(G):
            # Remove cycles by removing edges with lowest weights
            G = self._make_dag(G)
        
        return G
    
    def _make_dag(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles to make the graph a DAG"""
        while not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = nx.find_cycle(graph)
                # Remove the edge with minimum weight in the cycle
                min_weight = float('inf')
                min_edge = None
                
                for edge in cycle:
                    weight = graph[edge[0]][edge[1]].get('weight', 1.0)
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = (edge[0], edge[1])
                
                if min_edge:
                    graph.remove_edge(*min_edge)
            except nx.NetworkXNoCycle:
                break  # No more cycles
                
        return graph
    
    def compute_dag_metrics(self, dag: nx.DiGraph) -> Dict[str, float]:
        """Compute metrics for the generated DAG"""
        if len(dag.nodes()) == 0:
            return {'nodes': 0, 'edges': 0, 'diameter': 0, 'avg_degree': 0}
        
        metrics = {
            'nodes': dag.number_of_nodes(),
            'edges': dag.number_of_edges(),
            'avg_degree': np.mean([d for n, d in dag.degree()]) if dag.degree() else 0
        }
        
        if nx.is_weakly_connected(dag):
            metrics['diameter'] = nx.diameter(dag.to_undirected())
        else:
            metrics['diameter'] = -1  # Disconnected
            
        return metrics