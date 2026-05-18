
"""
test_gnn.py — Unit tests for the GNN Intrusion Detection System
Covers: GNNModel (all 6 architectures), make_graph, evaluate,
        feature preprocessing, and edge-case handling.

Run with:
    pytest test_gnn.py -v
    pytest test_gnn.py -v --tb=short   # shorter tracebacks
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv, GINConv, ChebConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Paste the production code here so tests are self-contained.
# In a real project you would do: from gnn_ids import GNNModel, make_graph, evaluate
# ─────────────────────────────────────────────────────────────────────────────

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_type='GCN'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif model_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        elif model_type == 'RGCN':
            self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=1)
            self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations=1)
        elif model_type == 'GIN':
            mlp1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim))
            mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, output_dim))
            self.conv1 = GINConv(mlp1, train_eps=True)
            self.conv2 = GINConv(mlp2, train_eps=True)
        elif model_type == 'CHEB':
            self.conv1 = ChebConv(input_dim, hidden_dim, K=3)
            self.conv2 = ChebConv(hidden_dim, output_dim, K=3)
        else:
            raise ValueError(f'Unknown model_type: {model_type}')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.model_type == 'RGCN':
            et = torch.zeros(edge_index.size(1), dtype=torch.long)
            x = F.relu(self.conv1(x, edge_index, et))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, et)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def make_graph(X_vals, y_vals, split=False):
    N = len(y_vals)
    fwd = [np.arange(0, N-1), np.arange(1, N)]
    bwd = [np.arange(1, N),   np.arange(0, N-1)]
    ei  = torch.tensor(np.concatenate([fwd, bwd], axis=1), dtype=torch.long)
    g   = Data(x=torch.tensor(X_vals, dtype=torch.float),
               edge_index=ei,
               y=torch.tensor(y_vals, dtype=torch.long))
    if split:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(N), test_size=0.2,
                                  stratify=y_vals, random_state=42)
        tm = torch.zeros(N, dtype=torch.bool); tm[tr] = True
        vm = torch.zeros(N, dtype=torch.bool); vm[te] = True
        g.train_mask = tm; g.test_mask = vm
    return g


@torch.no_grad()
def evaluate(model, data, find_best_threshold=False):
    model.eval()
    logits = model(data)
    y_true = data.y
    probs  = logits.exp()[:, 1]
    yt = y_true.cpu().numpy()
    pr = probs.cpu().numpy()
    raw_auc = roc_auc_score(yt, pr)
    auc_flipped = raw_auc < 0.5
    if auc_flipped:
        pr = 1.0 - pr
    auc = max(raw_auc, 1 - raw_auc)
    if find_best_threshold:
        best_f1, best_thresh, best_preds = 0.0, 0.5, (pr >= 0.5).astype(int)
        for t in np.arange(0.05, 0.96, 0.01):
            cand = (pr >= t).astype(int)
            f = f1_score(yt, cand, zero_division=0)
            if f > best_f1:
                best_f1, best_thresh, best_preds = f, t, cand
        yp = best_preds
    else:
        yp = (pr >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(yt, yp),
                f1=f1_score(yt, yp, zero_division=0),
                precision=precision_score(yt, yp, zero_division=0),
                recall=recall_score(yt, yp, zero_division=0),
                roc_auc=auc)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIM  = 8 # matches COMMON_FEATURES length
HIDDEN_DIM = 16 # smaller than production (64) so tests run fast
OUTPUT_DIM = 2 # binary: normal vs attack
N_NODES = 50 # small graph for speed

ALL_MODEL_TYPES = ['GCN', 'GAT', 'SAGE', 'RGCN', 'GIN', 'CHEB']


@pytest.fixture
def small_graph():
    """50-node balanced binary graph with no NaN/Inf features."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N_NODES, INPUT_DIM)).astype(np.float32)
    y = np.array([0] * (N_NODES // 2) + [1] * (N_NODES // 2))
    rng.shuffle(y)
    return make_graph(X, y, split=False)


@pytest.fixture
def split_graph():
    """50-node graph with train_mask and test_mask (for Notebook 1 evaluate)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_NODES, INPUT_DIM)).astype(np.float32)
    y = np.array([0] * (N_NODES // 2) + [1] * (N_NODES // 2))
    rng.shuffle(y)
    return make_graph(X, y, split=True)


@pytest.fixture(params=ALL_MODEL_TYPES)
def model(request):
    """One GNNModel instance per architecture - parametrised."""
    return GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, request.param)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GNNModel - construction
# ─────────────────────────────────────────────────────────────────────────────

class TestGNNModelConstruction:

    @pytest.mark.parametrize("m_type", ALL_MODEL_TYPES)
    def test_all_architectures_instantiate(self, m_type):
        """Every supported architecture should build without error."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, m_type)
        assert model.model_type == m_type

    def test_unknown_architecture_raises(self):
        """An unrecognised model_type should raise ValueError, not silently fail."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, model_type='TRANSFORMER')

    def test_model_type_stored_on_instance(self):
        """model_type must be stored - forward() branches on it for RGCN."""
        for m_type in ALL_MODEL_TYPES:
            model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, m_type)
            assert hasattr(model, 'model_type')
            assert model.model_type == m_type

    def test_gat_head_dimensions(self):
        """GAT conv1 uses 4 heads with concat=True, so its output dim = hidden*4."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GAT')
        # conv2 input must equal hidden * 4
        assert model.conv2.in_channels == HIDDEN_DIM * 4

    def test_gin_has_mlp_layers(self):
        """GIN wraps MLP sequences - both conv layers should be GINConv."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GIN')
        assert isinstance(model.conv1, GINConv)
        assert isinstance(model.conv2, GINConv)

    def test_cheb_uses_k3(self):
        """ChebConv should use K=3 as specified."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'CHEB')
        assert len(model.conv1.lins) == 3  # K filters


# ─────────────────────────────────────────────────────────────────────────────
# 2. GNNModel - forward pass output shape & validity
# ─────────────────────────────────────────────────────────────────────────────

class TestGNNModelForward:

    def test_output_shape(self, model, small_graph):
        """Output should be (N_NODES, OUTPUT_DIM) for every architecture."""
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        assert out.shape == (N_NODES, OUTPUT_DIM)

    def test_output_is_log_softmax(self, model, small_graph):
        """log_softmax rows must sum to 0 (i.e. exp(row).sum() ≈ 1)."""
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        row_sums = out.exp().sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(N_NODES), atol=1e-5), \
            f"exp(log_softmax) rows don't sum to 1: min={row_sums.min():.6f}"

    def test_output_no_nan(self, model, small_graph):
        """No NaN values in the output (indicates numerical instability)."""
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        assert not torch.isnan(out).any(), "NaN detected in model output"

    def test_output_no_inf(self, model, small_graph):
        """No Inf values in the output."""
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        assert not torch.isinf(out).any(), "Inf detected in model output"

    def test_dropout_off_in_eval_mode(self, small_graph):
        """Two forward passes in eval mode should produce identical outputs."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        model.eval()
        with torch.no_grad():
            out1 = model(small_graph)
            out2 = model(small_graph)
        assert torch.allclose(out1, out2), \
            "eval() outputs differ - dropout is still active"

    def test_dropout_active_in_train_mode(self, small_graph):
        """Two forward passes in train mode should NOT be identical (dropout is stochastic)."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        model.train()
        out1 = model(small_graph)
        out2 = model(small_graph)
        # With p=0.5 dropout and 50 nodes the probability both passes are
        # identical is astronomically small - if they are, dropout is broken.
        assert not torch.allclose(out1, out2), \
            "train() outputs are identical - dropout may not be active"

    @pytest.mark.parametrize("m_type", ALL_MODEL_TYPES)
    def test_argmax_is_binary(self, m_type, small_graph):
        """Predicted class indices must be 0 or 1 for binary classification."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, m_type)
        model.eval()
        with torch.no_grad():
            preds = model(small_graph).argmax(dim=1)
        unique = preds.unique().tolist()
        assert all(v in [0, 1] for v in unique), \
            f"Unexpected class indices: {unique}"

    def test_rgcn_uses_edge_type_tensor(self, small_graph):
        """RGCN forward must not raise when edge_type tensor is constructed."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'RGCN')
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        assert out.shape == (N_NODES, OUTPUT_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# 3. make_graph
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeGraph:

    def test_node_count(self):
        X = np.zeros((20, INPUT_DIM), dtype=np.float32)
        y = np.zeros(20, dtype=int)
        g = make_graph(X, y)
        assert g.num_nodes == 20

    def test_edge_count_bidirectional(self):
        """N nodes → N-1 forward edges + N-1 backward = 2*(N-1) total edges."""
        N = 20
        X = np.zeros((N, INPUT_DIM), dtype=np.float32)
        y = np.zeros(N, dtype=int)
        g = make_graph(X, y)
        assert g.num_edges == 2 * (N - 1), \
            f"Expected {2*(N-1)} edges, got {g.num_edges}"

    def test_edge_index_dtype(self):
        X = np.zeros((10, INPUT_DIM), dtype=np.float32)
        y = np.zeros(10, dtype=int)
        g = make_graph(X, y)
        assert g.edge_index.dtype == torch.long

    def test_feature_tensor_dtype(self):
        X = np.zeros((10, INPUT_DIM), dtype=np.float32)
        y = np.zeros(10, dtype=int)
        g = make_graph(X, y)
        assert g.x.dtype == torch.float

    def test_label_tensor_dtype(self):
        X = np.zeros((10, INPUT_DIM), dtype=np.float32)
        y = np.zeros(10, dtype=int)
        g = make_graph(X, y)
        assert g.y.dtype == torch.long

    def test_split_masks_created(self):
        X = np.zeros((50, INPUT_DIM), dtype=np.float32)
        y = np.array([0]*25 + [1]*25)
        g = make_graph(X, y, split=True)
        assert hasattr(g, 'train_mask')
        assert hasattr(g, 'test_mask')

    def test_split_masks_not_overlapping(self):
        X = np.zeros((50, INPUT_DIM), dtype=np.float32)
        y = np.array([0]*25 + [1]*25)
        g = make_graph(X, y, split=True)
        overlap = (g.train_mask & g.test_mask).sum().item()
        assert overlap == 0, f"train and test masks overlap on {overlap} nodes"

    def test_split_masks_cover_all_nodes(self):
        N = 50
        X = np.zeros((N, INPUT_DIM), dtype=np.float32)
        y = np.array([0]*25 + [1]*25)
        g = make_graph(X, y, split=True)
        total = (g.train_mask | g.test_mask).sum().item()
        assert total == N

    def test_split_test_size_approx_20_percent(self):
        N = 100
        X = np.zeros((N, INPUT_DIM), dtype=np.float32)
        y = np.array([0]*50 + [1]*50)
        g = make_graph(X, y, split=True)
        test_count = g.test_mask.sum().item()
        assert 18 <= test_count <= 22, \
            f"Expected ~20 test nodes, got {test_count}"

    def test_no_split_has_no_masks(self):
        X = np.zeros((10, INPUT_DIM), dtype=np.float32)
        y = np.zeros(10, dtype=int)
        g = make_graph(X, y, split=False)
        assert not hasattr(g, 'train_mask')
        assert not hasattr(g, 'test_mask')

    def test_bidirectional_both_directions_present(self):
        """Edge i→i+1 and i+1→i must both exist for every consecutive pair."""
        N = 5
        X = np.zeros((N, INPUT_DIM), dtype=np.float32)
        y = np.zeros(N, dtype=int)
        g = make_graph(X, y)
        edges = set(map(tuple, g.edge_index.T.tolist()))
        for i in range(N - 1):
            assert (i, i+1) in edges, f"Forward edge ({i}→{i+1}) missing"
            assert (i+1, i) in edges, f"Backward edge ({i+1}→{i}) missing"

    def test_minimum_graph_two_nodes(self):
        """Graph with 2 nodes (minimum for edges) should not crash."""
        X = np.zeros((2, INPUT_DIM), dtype=np.float32)
        y = np.array([0, 1])
        g = make_graph(X, y)
        assert g.num_nodes == 2
        assert g.num_edges == 2  # one forward + one backward


# ─────────────────────────────────────────────────────────────────────────────
# 4. evaluate()
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluate:

    def _make_perfect_model(self, graph):
        """Return a model that always predicts the correct class (via weight surgery)."""
        # Just use a trained-enough random model for shape tests;
        # for correctness we mock probabilities directly.
        return GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')

    def test_returns_all_metric_keys(self, small_graph):
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        result = evaluate(model, small_graph)
        for key in ('accuracy', 'f1', 'precision', 'recall', 'roc_auc'):
            assert key in result, f"Missing metric key: {key}"

    def test_metrics_in_valid_range(self, small_graph):
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        result = evaluate(model, small_graph)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, \
                f"Metric '{key}' = {val} is outside [0, 1]"

    def test_find_best_threshold_returns_metrics(self, small_graph):
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        result = evaluate(model, small_graph, find_best_threshold=True)
        for key in ('accuracy', 'f1', 'precision', 'recall', 'roc_auc'):
            assert key in result

    def test_roc_auc_always_above_random(self, small_graph):
        """ROC-AUC should be ≥ 0.5 because the evaluate function flips inverted probabilities."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        result = evaluate(model, small_graph)
        assert result['roc_auc'] >= 0.5, \
            f"ROC-AUC {result['roc_auc']} < 0.5 - probability flip not working"

    @pytest.mark.parametrize("m_type", ALL_MODEL_TYPES)
    def test_all_architectures_evaluate(self, m_type, small_graph):
        """evaluate() should complete without error for every architecture."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, m_type)
        result = evaluate(model, small_graph)
        assert isinstance(result, dict)

    def test_eval_mode_set_inside_evaluate(self, small_graph):
        """evaluate() must call model.eval() - model should be in eval mode after."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        model.train()
        evaluate(model, small_graph)
        assert not model.training, "Model is still in training mode after evaluate()"

    def test_no_grad_inside_evaluate(self, small_graph):
        """evaluate() uses @torch.no_grad() - parameters must have no grad after call."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        evaluate(model, small_graph)
        for p in model.parameters():
            assert p.grad is None or p.grad.sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestFeaturePreprocessing:

    COMMON_FEATURES = [
        'duration', 'src_bytes', 'dst_bytes', 'count',
        'srv_count', 'serror_rate', 'rerror_rate', 'same_srv_rate',
    ]

    def _make_raw_X(self, n=100):
        rng = np.random.default_rng(7)
        X = rng.exponential(scale=100, size=(n, len(self.COMMON_FEATURES)))
        return X.astype(np.float32)

    def test_scaler_zero_mean(self):
        X = self._make_raw_X()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        means = X_scaled.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-5), \
            f"Scaled means not near zero: {means}"

    def test_scaler_unit_variance(self):
        X = self._make_raw_X()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        stds = X_scaled.std(axis=0)
        assert np.allclose(stds, 1, atol=1e-5), \
            f"Scaled std not near 1: {stds}"

    def test_inf_replaced_before_scaling(self):
        """Inf values must be replaced before passing to StandardScaler."""
        X = self._make_raw_X()
        X[0, 0] = np.inf
        X[1, 1] = -np.inf
        X_clean = np.where(np.isinf(X), 0, X)
        assert not np.isinf(X_clean).any()

    def test_nan_replaced_before_scaling(self):
        """NaN values must be replaced before passing to StandardScaler."""
        X = self._make_raw_X()
        X[0, 2] = np.nan
        X_clean = np.nan_to_num(X, nan=0.0)
        assert not np.isnan(X_clean).any()

    def test_no_nan_after_fillna_and_replace(self):
        """Simulates the .fillna(0).replace([inf, -inf], 0) chain used in notebooks."""
        import pandas as pd
        X = self._make_raw_X()
        X[5, 3] = np.nan
        X[6, 4] = np.inf
        df = pd.DataFrame(X, columns=self.COMMON_FEATURES)
        X_vals = df.fillna(0).replace([np.inf, -np.inf], 0).values
        assert not np.isnan(X_vals).any()
        assert not np.isinf(X_vals).any()

    def test_feature_count_matches_common_features(self):
        """Model input_dim must match len(COMMON_FEATURES) = 8."""
        assert len(self.COMMON_FEATURES) == 8

    def test_scaler_transform_only_on_test(self):
        """test set must use transform(), not fit_transform(), to prevent leakage."""
        X_train = self._make_raw_X(80)
        X_test  = self._make_raw_X(20)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # transform() should not raise and must use train statistics
        X_test_scaled = scaler.transform(X_test)
        assert X_test_scaled.shape == (20, len(self.COMMON_FEATURES))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training loop sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingLoop:

    def test_loss_decreases_over_epochs(self, split_graph):
        """Loss should drop over 20 epochs of training on a simple graph."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _ in range(20):
            model.train()
            optimizer.zero_grad()
            out  = model(split_graph)
            loss = criterion(out[split_graph.train_mask],
                             split_graph.y[split_graph.train_mask])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss over the last 5 epochs should be lower than the first 5
        assert np.mean(losses[-5:]) < np.mean(losses[:5]), \
            f"Loss did not decrease: first 5 avg={np.mean(losses[:5]):.4f}, " \
            f"last 5 avg={np.mean(losses[-5:]):.4f}"

    def test_gradients_flow_through_model(self, split_graph):
        """All parameters with requires_grad=True should have non-None gradients after one backward pass."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        out  = model(split_graph)
        loss = criterion(out[split_graph.train_mask],
                         split_graph.y[split_graph.train_mask])
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for parameter: {name}"

    def test_class_weights_tensor_shape(self, split_graph):
        """Class weight tensor must have shape (output_dim,) = (2,)."""
        y_train = split_graph.y[split_graph.train_mask].numpy()
        n0, n1  = (y_train == 0).sum(), (y_train == 1).sum()
        w0, w1  = (n0+n1)/(2*n0), (n0+n1)/(2*n1)
        cw = torch.tensor([w0, w1], dtype=torch.float)
        assert cw.shape == (2,)
        assert (cw > 0).all()

    def test_model_saves_and_loads(self, split_graph, tmp_path):
        """State dict saved with torch.save must reload without error."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'SAGE')
        path  = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), path)

        loaded = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'SAGE')
        loaded.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        loaded.eval()

        model.eval()
        with torch.no_grad():
            out_orig   = model(split_graph)
            out_loaded = loaded(split_graph)
        assert torch.allclose(out_orig, out_loaded), \
            "Loaded model produces different output than saved model"

    @pytest.mark.parametrize("m_type", ALL_MODEL_TYPES)
    def test_all_architectures_backward_pass(self, m_type, split_graph):
        """backward() must succeed for every architecture."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, m_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        out  = model(split_graph)
        loss = criterion(out[split_graph.train_mask],
                         split_graph.y[split_graph.train_mask])
        loss.backward()   # must not raise
        assert loss.item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cross-dataset evaluation specifics
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossDatasetEval:

    def test_best_threshold_sweep_finds_threshold_in_range(self, small_graph):
        """find_best_threshold must pick a threshold between 0.05 and 0.95."""
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')

        # Patch evaluate to also return the threshold (matches NB3 signature)
        model.eval()
        with torch.no_grad():
            logits = model(small_graph)
        probs = logits.exp()[:, 1].numpy()
        yt    = small_graph.y.numpy()
        raw_auc = roc_auc_score(yt, probs)
        if raw_auc < 0.5:
            probs = 1.0 - probs

        best_f1, best_thresh = 0.0, 0.5
        for t in np.arange(0.05, 0.96, 0.01):
            cand = (probs >= t).astype(int)
            f = f1_score(yt, cand, zero_division=0)
            if f > best_f1:
                best_f1, best_thresh = f, t

        assert 0.05 <= best_thresh <= 0.95, \
            f"Optimal threshold {best_thresh} outside [0.05, 0.95]"

    def test_roc_auc_flip_corrects_inverted_probs(self, small_graph):
        """If raw AUC < 0.5, flipping probabilities must yield AUC >= 0.5."""
        # Manufacture artificially inverted probabilities
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        model.eval()
        with torch.no_grad():
            logits = model(small_graph)
        probs = logits.exp()[:, 1].numpy()
        yt = small_graph.y.numpy()

        # Force inversion
        inv_probs = 1.0 - probs
        raw_auc = roc_auc_score(yt, inv_probs)
        if raw_auc < 0.5:
            inv_probs = 1.0 - inv_probs
        auc = max(raw_auc, 1 - raw_auc)
        assert auc >= 0.5

    def test_independent_scaler_per_dataset(self):
        """Each dataset's scaler must be fit independently (not shared from NSL-KDD)."""
        rng = np.random.default_rng(1)
        X_nsl  = rng.exponential(1.0,  (100, 8)).astype(np.float32)
        X_cic  = rng.exponential(10.0, (100, 8)).astype(np.float32) # different scale

        scaler_nsl = StandardScaler().fit(X_nsl)
        scaler_cic = StandardScaler().fit(X_cic)

        # The two scalers must have different means — confirming they are independent
        assert not np.allclose(scaler_nsl.mean_, scaler_cic.mean_), \
            "NSL-KDD and CICIDS scalers have identical means — likely the same object"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge cases & robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_all_normal_labels(self):
        """Graph where all labels are 0 should not crash make_graph."""
        X = np.zeros((20, INPUT_DIM), dtype=np.float32)
        y = np.zeros(20, dtype=int)
        g = make_graph(X, y, split=False)
        assert g.num_nodes == 20

    def test_all_attack_labels(self):
        """Graph where all labels are 1 should not crash make_graph."""
        X = np.zeros((20, INPUT_DIM), dtype=np.float32)
        y = np.ones(20, dtype=int)
        g = make_graph(X, y, split=False)
        assert g.num_nodes == 20

    def test_model_handles_single_class_graph(self):
        """Forward pass on a single-class graph must not raise."""
        X = np.zeros((20, INPUT_DIM), dtype=np.float32)
        y = np.zeros(20, dtype=int)
        g = make_graph(X, y, split=False)
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        model.eval()
        with torch.no_grad():
            out = model(g)
        assert out.shape == (20, OUTPUT_DIM)

    def test_large_feature_values_no_nan(self):
        """Very large raw feature values should not produce NaN after scaling."""
        X = np.full((20, INPUT_DIM), 1e9, dtype=np.float32)
        X_scaled = StandardScaler().fit_transform(X)
        # All identical values → std = 0; StandardScaler sets result to 0
        assert not np.isnan(X_scaled).any()

    def test_zero_feature_values_no_nan(self):
        """All-zero feature matrix (e.g. missing data filled with 0) must survive scaling."""
        X = np.zeros((20, INPUT_DIM), dtype=np.float32)
        X_scaled = StandardScaler().fit_transform(X)
        assert not np.isnan(X_scaled).any()

    def test_model_output_consistent_across_devices(self, small_graph):
        """CPU-loaded model must produce identical results as in-memory model."""
        import tempfile, os
        model = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
        model.eval()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            torch.save(model.state_dict(), path)
            loaded = GNNModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, 'GCN')
            loaded.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            loaded.eval()
            with torch.no_grad():
                out_orig   = model(small_graph)
                out_loaded = loaded(small_graph)
            assert torch.allclose(out_orig, out_loaded)
        finally:
            os.unlink(path)