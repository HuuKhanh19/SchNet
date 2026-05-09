
"""SchNet with multi-conformer hierarchical readout.

Forward signature:

    forward(z, pos, atom_to_conf, num_confs) -> [B, 1]

where:
    z              [total_atoms]      atomic numbers (repeated K times per molecule)
    pos            [total_atoms, 3]   3D coordinates
    atom_to_conf   [total_atoms]      GLOBAL conformer index per atom (0..total_confs-1)
    num_confs      [B]                # of conformers per molecule in the batch

Hierarchical readout:
    per-atom scalar  --(atom_readout: SUM)-->  per-conf scalar
                     --(conf_readout: MEAN)-->  per-mol scalar

`atom_to_conf` doubles as the radius_graph batch index, so edges never cross
conformers.

Includes:
    - `interaction_dropout`   F.dropout on each block's residual update
    - `init_output_bias`      sets lin2.bias so initial prediction ≈ mean target
"""

from math import pi as PI
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver


# =============================================================================
# Building blocks
# =============================================================================

class GaussianSmearing(nn.Module):
    """Expand pairwise distance into a basis of equally-spaced Gaussians."""

    def __init__(self, start: float = 0.0, stop: float = 5.0,
                 num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    """softplus(x) - log(2). Smooth, zero-centred at x=0."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """Continuous-filter convolution."""

    def __init__(self, in_channels: int, out_channels: int, num_filters: int,
                 nn_module: Sequential, cutoff: float):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn_module
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        # Cosine cutoff: smooth decay to zero at r=cutoff.
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class InteractionBlock(nn.Module):
    """SchNet interaction block: filter-net + CFConv + dense."""

    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels,
                           num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class RadiusInteractionGraph(nn.Module):
    """Build edges within `cutoff` per `batch` index."""

    def __init__(self, cutoff: float, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


# =============================================================================
# SchNet
# =============================================================================

class SchNet(nn.Module):
    """SchNet with multi-conformer hierarchical readout."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        atom_readout: str = "add",       # atoms -> conf  (extensive: SUM)
        conf_readout: str = "mean",      # conf  -> mol   (average across confs)
        interaction_dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.atom_readout_name = atom_readout
        self.conf_readout_name = conf_readout
        self.interaction_dropout = float(interaction_dropout)

        # Atom embedding (z=0 reserved for padding).
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        # Edge graph + distance expansion.
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Hierarchical readouts.
        self.atom_readout = aggr_resolver(atom_readout)
        self.conf_readout = aggr_resolver(conf_readout)

        # N interaction blocks.
        self.interactions = ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ])

        # Output net: H -> H/2 -> 1 (per atom).
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    # ------------------------------------------------------------------
    def init_output_bias(self, mean_target: float, mean_n_atoms: float) -> None:
        """Init `lin2.bias` so initial molecule prediction ≈ `mean_target`.

        Math (with atom_readout=sum, conf_readout=mean):
            per-atom output ≈ lin2.bias              (after weight shrink)
            conf-level      = sum_atoms = N * lin2.bias
            mol-level       = mean_confs(N * bias)   = N * lin2.bias

        So setting lin2.bias = mean_target / mean_n_atoms makes the initial
        molecule-level prediction match the training mean. Lin1/lin2 weights
        are shrunk ×0.01 so the bias dominates initially; the model then
        learns the residual on top.

        For atom_readout='mean', use lin2.bias = mean_target (no division).
        Adjust the call site if you change the readout config.
        """
        bias_val = mean_target / max(mean_n_atoms, 1.0)
        with torch.no_grad():
            self.lin1.weight.data *= 0.01
            self.lin1.bias.data.fill_(0)
            self.lin2.weight.data *= 0.01
            self.lin2.bias.data.fill_(bias_val)
        print(
            f"init_output_bias: lin2.bias = {bias_val:.6f}  "
            f"(mean_target={mean_target:.4f}, mean_n_atoms={mean_n_atoms:.1f})"
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        z: Tensor,                # [total_atoms]
        pos: Tensor,              # [total_atoms, 3]
        atom_to_conf: Tensor,     # [total_atoms]  global conf id per atom
        num_confs: Tensor,        # [B]            confs per molecule
    ) -> Tensor:                  # [B, 1]
        # 1. Embed atoms.
        h = self.embedding(z)

        # 2. Edges per conformer (atom_to_conf serves as the batch index, so
        #    radius_graph won't cross conformers).
        edge_index, edge_weight = self.interaction_graph(pos, atom_to_conf)
        edge_attr = self.distance_expansion(edge_weight)

        # 3. Interaction blocks (residual + optional dropout on the update).
        for interaction in self.interactions:
            delta = interaction(h, edge_index, edge_weight, edge_attr)
            if self.interaction_dropout > 0.0:
                delta = F.dropout(
                    delta, p=self.interaction_dropout, training=self.training
                )
            h = h + delta

        # 4. Per-atom scalar output.
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)                 # [total_atoms, 1]

        # 5. Atoms -> conformers.
        h_conf = self.atom_readout(h, atom_to_conf, dim=0)  # [total_confs, 1]

        # 6. Build conf_to_mol on the fly from num_confs.
        B = num_confs.size(0)
        num_confs = num_confs.view(-1)
        conf_to_mol = torch.repeat_interleave(
            torch.arange(B, device=num_confs.device), num_confs
        )

        # 7. Conformers -> molecules.
        out = self.conf_readout(h_conf, conf_to_mol, dim=0)  # [B, 1]
        return out

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"hidden={self.hidden_channels}, "
            f"filters={self.num_filters}, "
            f"interactions={self.num_interactions}, "
            f"gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff}, "
            f"atom_readout={self.atom_readout_name}, "
            f"conf_readout={self.conf_readout_name}, "
            f"dropout={self.interaction_dropout})"
        )