"""Graph-aware DataLoader.

`torch.utils.data.DataLoader` doesn't know how to batch graphs (per-graph
variable node count, edge_index needing offsets). We delegate that to
`torch_geometric.data.Batch.from_data_list`, which:

    1. Concatenates per-node tensors (z, pos, x, ...) along the node axis.
    2. Shifts edge_index per graph by cumulative node count so indices stay
       consistent in the batched view.
    3. Adds a `batch` LongTensor [total_nodes] mapping each node -> graph id.

If you ever want to drop torch_geometric entirely, `graph_collate` is the
single function you'd reimplement.
"""

from torch.utils.data import DataLoader
from torch_geometric.data import Batch


def graph_collate(data_list):
    """Collate a list of `torch_geometric.data.Data` into one `Batch`."""
    return Batch.from_data_list(data_list)


class GraphDataLoader(DataLoader):
    """`torch.utils.data.DataLoader` with graph-aware `collate_fn`."""

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=graph_collate,
            **kwargs,
        )