"""Stage 4: GCN architecture for graph classification.

Input: Data(x=[n_nodes, 2], edge_index=[2, n_edges], batch=[n_nodes])
GCNConv(2->64) -> ReLU -> GCNConv(64->64) -> ReLU -> GCNConv(64->64) -> ReLU
-> global_mean_pool -> Linear(64->32) -> ReLU -> Dropout(0.3) -> Linear(32->num_classes)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class ECGGCN(torch.nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, num_classes=5, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin1 = torch.nn.Linear(hidden_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    g1 = Data(x=torch.randn(10, 2), edge_index=torch.randint(0, 10, (2, 20)))
    g2 = Data(x=torch.randn(15, 2), edge_index=torch.randint(0, 15, (2, 30)))
    batch = Batch.from_data_list([g1, g2])

    model = ECGGCN()
    out = model(batch.x, batch.edge_index, batch.batch)
    print("Output shape:", out.shape)  # [2, num_classes]
