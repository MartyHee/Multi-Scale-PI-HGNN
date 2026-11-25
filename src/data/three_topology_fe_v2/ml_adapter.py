"""Classical-ML representation from the common graph (no separate parser)."""
from __future__ import annotations

import numpy as np


def graph_to_ml(data):
    node_x = data["mesh_node"].x.detach().cpu().numpy()
    node_y = data["mesh_node"].y_disp.detach().cpu().numpy()
    beam_x = data["beam_element"].x.detach().cpu().numpy()
    beam_y = data["beam_element"].y_force.detach().cpu().numpy()
    ei = data["mesh_node", "belongs_to_beam", "beam_element"].edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    n_beams = beam_x.shape[0]
    sums = np.zeros((n_beams, node_x.shape[1]), dtype=np.float64)
    counts = np.zeros(n_beams, dtype=np.float64)
    np.add.at(sums, dst, node_x[src])
    np.add.at(counts, dst, 1.0)
    beam_endpoint_mean = sums / np.maximum(counts, 1.0)[:, None]
    beam_in = np.concatenate([beam_x, beam_endpoint_mean], axis=1)
    return node_x, node_y, beam_in, beam_y


def collect(dataset, sample_ids):
    Xn, Yn, Xb, Yb = [], [], [], []
    for sid in sample_ids:
        d = dataset[sid]
        n, y, b, by = graph_to_ml(d)
        Xn.append(n); Yn.append(y); Xb.append(b); Yb.append(by)
    return (np.concatenate(Xn), np.concatenate(Yn),
            np.concatenate(Xb), np.concatenate(Yb))


class MLBatchPredictor:
    """Expose a fitted classical-ML model through the graph-model interface.

    ``engine.validate`` / ``MultiSliceEvaluator`` call ``model(batch)`` and
    inverse-transform both prediction and target with the same stats.  This
    adapter computes the *same* tabular representation as :func:`graph_to_ml`
    directly on a PyG batch, so ML and gradient models share one evaluation
    implementation (metrics, graph-equal reduction, export, stats source).
    """

    def __init__(self, node_predict, beam_predict):
        self.node_predict = node_predict
        self.beam_predict = beam_predict

    def eval(self):
        return self

    def train(self):
        return self

    def to(self, device):  # no-op; sklearn models live on CPU
        return self

    def __call__(self, batch):
        import torch
        node_x = batch["mesh_node"].x
        beam_x = batch["beam_element"].x
        ei = batch["mesh_node", "belongs_to_beam", "beam_element"].edge_index
        src, dst = ei[0], ei[1]
        sums = torch.zeros((beam_x.shape[0], node_x.shape[1]),
                           dtype=torch.float64, device=node_x.device)
        sums.index_add_(0, dst, node_x[src].to(torch.float64))
        counts = torch.zeros(beam_x.shape[0], dtype=torch.float64, device=node_x.device)
        counts.index_add_(0, dst, torch.ones(dst.shape[0], dtype=torch.float64,
                                             device=node_x.device))
        endpoint_mean = sums / counts.clamp(min=1.0)[:, None]
        beam_in = torch.cat([beam_x.to(torch.float64), endpoint_mean], dim=1)
        xn = node_x.detach().cpu().numpy().astype(np.float64, copy=False)
        xb = beam_in.detach().cpu().numpy().astype(np.float64, copy=False)
        pn = np.asarray(self.node_predict(xn), dtype=np.float64)
        pb = np.asarray(self.beam_predict(xb), dtype=np.float64)
        return (torch.as_tensor(pn, dtype=torch.float32, device=node_x.device),
                torch.as_tensor(pb, dtype=torch.float32, device=node_x.device))
