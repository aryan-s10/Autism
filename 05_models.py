import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
 
DATA_PATH = "Output/graph_data.pt"
class GCN(nn.Module):
    """
    Standard 2-layer GCN.
 
    Layer 1: in_channels (42) -> hidden_channels
    Layer 2: hidden_channels -> out_channels (2, one logit per class)
 
    Each GCNConv performs: for every node, aggregate (structurally-
    weighted average of) neighbor features, then apply a learned
    linear transform. Stacking 2 layers means each node's final
    representation is influenced by its neighbors' neighbors
    (2-hop receptive field) -- important for a k-NN similarity graph,
    since it lets information propagate beyond immediate neighbors.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
 
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # dropout during training only -- standard regularization,
        # extra important here since we only have 492 training nodes
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits -- CrossEntropyLoss applies log-softmax internally
    
class GAT(nn.Module):
    """
    2-layer GAT with multi-head attention on the first layer.
 
    Layer 1: in_channels (42) -> hidden_channels, with `heads` parallel
        attention heads whose outputs are CONCATENATED. Multi-head
        attention is analogous to multiple "similarity perspectives"
        learned simultaneously -- one head might learn to weight
        neighbors by AQ-10 similarity patterns, another by demographic
        similarity, etc. (though in practice heads aren't hand-assigned
        like this -- they emerge from training.)
 
    Layer 2: (hidden_channels * heads) -> out_channels, with heads=1
        and concat=False. The output layer collapses back down to
        raw per-class logits, so we don't want multiple concatenated
        heads here -- just one final attention-weighted aggregation.
 
    dropout=0.6 is deliberately higher than the GCN's 0.5: GAT has
    more parameters (per-head attention weights) than GCN on the same
    channel sizes, and our dataset is small (704 nodes), so it's more
    prone to overfitting without stronger regularization.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                              concat=False, dropout=dropout)
        self.dropout = dropout
 
    def forward(self, x, edge_index, return_attention_weights=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention_weights:
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU is the standard activation choice in the original GAT paper
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention_weights:
            x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
            return x, (att1, att2)
        x = self.conv2(x, edge_index)
        return x

data = torch.load(DATA_PATH, weights_only=False)
print(f"Loaded graph: {data}")
 
in_channels = data.num_node_features  # 42
out_channels = 2  # ASD negative / positive
hidden_channels = 16
 
gcn = GCN(in_channels, hidden_channels, out_channels)
gat = GAT(in_channels, hidden_channels, out_channels, heads=8)
 
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
print("\n" + "=" * 60)
print("MODEL ARCHITECTURES")
print("=" * 60)
print(gcn)
print(f"GCN trainable parameters: {count_params(gcn):,}")
print()
print(gat)
print(f"GAT trainable parameters: {count_params(gat):,}")
 
print("\n" + "=" * 60)
print("FORWARD PASS SANITY CHECK (untrained weights, just checking shapes)")
print("=" * 60)
gcn.eval()
gat.eval()
with torch.no_grad():
    out_gcn = gcn(data.x, data.edge_index)
    out_gat = gat(data.x, data.edge_index)
print(f"GCN output shape: {out_gcn.shape} (expected [704, 2])")
print(f"GAT output shape: {out_gat.shape} (expected [704, 2])")
assert out_gcn.shape == (data.num_nodes, out_channels)
assert out_gat.shape == (data.num_nodes, out_channels)
print("\nBoth models produce correctly shaped output. Ready for Phase 6 (training).")

