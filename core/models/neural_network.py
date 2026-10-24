"""GKX feed-forward return regressions. See docs/working_notes/Neural Networks.md.

The objective is mean half squared error + l1 * sum(abs(dense weights)).
Biases and batch-normalization parameters are not penalized. Targets stay in
their supplied units. Validation is used for stopping, never gradient updates.
"""

import os

# Required by deterministic CUDA matrix multiplication; set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn

from core.model_inputs import positive_integer, target_vector


ARCHITECTURES = {f"NN{depth}": (32, 16, 8, 4, 2)[:depth] for depth in range(1, 6)}


class ReturnNetwork(nn.Module):
    """Dense -> ReLU -> BatchNorm for each hidden layer; linear scalar output.

    Explicit implementation choices: Glorot uniform kernels, zero biases,
    BN eps=1e-5/momentum=.1, affine gamma=1/beta=0 and running statistics.
    """

    def __init__(self, n_features, hidden_sizes):
        super().__init__()
        positive_integer(n_features, "n_features")
        if not hidden_sizes:
            raise ValueError("Provide at least one hidden layer.")
        layers = []
        width = n_features
        for units in hidden_sizes:
            positive_integer(units, "hidden layer width")
            layers.extend([nn.Linear(width, units), nn.ReLU(),
                           nn.BatchNorm1d(units, eps=1e-5, momentum=.1)])
            width = units
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X):
        return self.layers(X).squeeze(-1)

    def weight_penalty(self):
        """Sum over dense kernels, including the output; no parameter-count divisor."""
        return sum(layer.weight.abs().sum() for layer in self.modules()
                   if isinstance(layer, nn.Linear))


def penalized_loss(network, prediction, target, l1):
    """Exact (B,) contract prevents silently broadcasting a (B,1) prediction."""
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("NN predictions and targets must have identical (batch,) shapes.")
    squared = .5 * (prediction - target).square().mean()
    return squared, l1 * network.weight_penalty()


def _matrix(X, n_features=None):
    if np.iscomplexobj(X):
        raise ValueError("NN inputs must be real.")
    array = np.asarray(X, dtype=np.float32, order="C")
    if (array.ndim != 2 or array.shape[1] < 1 or
            (n_features is not None and array.shape[1] != n_features) or
            not np.isfinite(array).all()):
        raise ValueError("NN inputs must be a finite matrix with the fitted feature width.")
    return array


def predict_network(network, X, *, device, batch_size):
    """Evaluation mode freezes BN statistics; preserve every row in input order."""
    prediction = np.empty(len(X), dtype=np.float64)
    network.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            inputs = torch.from_numpy(X[start:start + batch_size]).to(device)
            values = network(inputs)
            if values.shape != (len(inputs),) or not torch.isfinite(values).all():
                raise FloatingPointError("Invalid NN predictions.")
            prediction[start:start + len(inputs)] = values.cpu().numpy()
    return prediction


class NeuralNetworkRegressor:
    """An equal-weight ensemble of independently initialized, early-stopped NNs.

    Hyperparameter selection belongs to the caller. All members here share one
    learning rate and L1 value; each restores its own best validation epoch.
    No train+validation refit, target scaling, dropout, clipping or warm start.
    """

    def __init__(self, *, hidden_sizes=(32,), learning_rate=.001, l1=.00001,
                 seeds=(2020,), batch_size=10000, max_epochs=100, patience=5,
                 device="cpu", predict_batch_size=10000):
        self.hidden_sizes = tuple(hidden_sizes)
        for width in self.hidden_sizes:
            positive_integer(width, "hidden layer width")
        if not self.hidden_sizes:
            raise ValueError("Provide at least one hidden layer.")
        for name, value in (("batch_size", batch_size), ("max_epochs", max_epochs),
                            ("patience", patience), ("predict_batch_size", predict_batch_size)):
            positive_integer(value, name)
        if batch_size < 2:
            raise ValueError("Batch normalization requires batch_size >= 2.")
        if (isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0 or
                isinstance(l1, bool) or not np.isfinite(l1) or l1 < 0):
            raise ValueError("Learning rate must be positive and L1 nonnegative.")
        self.seeds = tuple(seeds)
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds) or
                any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in self.seeds)):
            raise ValueError("Provide distinct unsigned 32-bit integer seeds.")
        if device not in {"cpu", "cuda"}:
            raise ValueError("NN device must be cpu or cuda.")
        self.learning_rate, self.l1 = float(learning_rate), float(l1)
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.device, self.predict_batch_size = device, predict_batch_size

    def fit(self, X, y, *, X_validation, y_validation, progress=None):
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; install the NN extra and check the GPU allocation.")
        for features, target in ((X, y), (X_validation, y_validation)):
            if isinstance(features, pd.DataFrame) and isinstance(target, pd.Series):
                if not features.index.equals(target.index):
                    raise ValueError("NN X/y indices must match in the same order.")
        self.feature_names_ = tuple(X.columns) if isinstance(X, pd.DataFrame) else None
        if isinstance(X, pd.DataFrame) and isinstance(X_validation, pd.DataFrame):
            if not X.columns.equals(X_validation.columns) or not X.columns.is_unique:
                raise ValueError("Training and validation feature order must match and be unique.")
        X = _matrix(X)
        V = _matrix(X_validation, X.shape[1])
        y = target_vector(y, len(X))
        vy = target_vector(y_validation, len(V))
        if len(X) < 2:
            raise ValueError("NN training requires at least two rows.")
        self.n_features_in_ = X.shape[1]
        self.models_, self.diagnostics_ = [], []
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        train_x = torch.from_numpy(X)
        train_y = torch.from_numpy(y.astype(np.float32))
        if not torch.isfinite(train_y).all():
            raise ValueError("Nonfinite NN targets after float32 conversion.")
        for seed in self.seeds:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            shuffle = torch.Generator(device="cpu").manual_seed(seed)
            network = ReturnNetwork(self.n_features_in_, self.hidden_sizes).to(self.device)
            initial = {name: value.detach().cpu().clone() for name, value in network.named_parameters()}
            optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate,
                betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False,
                foreach=False, fused=False)
            initial_prediction = predict_network(network, X, device=self.device, batch_size=self.predict_batch_size)
            initial_mse = float(np.mean((y - initial_prediction)**2))
            initial_validation = predict_network(network, V, device=self.device, batch_size=self.predict_batch_size)
            history, best_state, best_score, best_epoch, stale = [], None, np.inf, 0, 0
            for epoch in range(1, self.max_epochs + 1):
                network.train()
                batches = list(torch.randperm(len(X), generator=shuffle).split(self.batch_size))
                # BN cannot train on one row. Merge a final singleton; do not drop it.
                if len(batches[-1]) == 1:
                    batches[-2] = torch.cat((batches[-2], batches[-1]))
                    batches.pop()
                online_squared, online_penalty, gradient_norm = 0., 0., None
                for indices in batches:
                    inputs, targets = train_x[indices].to(self.device), train_y[indices].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    squared, penalty = penalized_loss(network, network(inputs), targets, self.l1)
                    loss = squared + penalty
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite NN loss at seed={seed}, epoch={epoch}.")
                    loss.backward()
                    gradients = [p.grad for p in network.parameters() if p.grad is not None]
                    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                        raise FloatingPointError("Missing or nonfinite NN gradients.")
                    if gradient_norm is None:
                        gradient_norm = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
                    optimizer.step()
                    online_squared += float(squared.detach()) * len(indices) / len(X)
                    online_penalty += float(penalty.detach()) * len(indices) / len(X)
                train_prediction = predict_network(network, X, device=self.device, batch_size=self.predict_batch_size)
                validation_prediction = predict_network(network, V, device=self.device, batch_size=self.predict_batch_size)
                score = float(np.mean((vy - validation_prediction)**2))
                record = dict(epoch=epoch, train_mse=float(np.mean((y - train_prediction)**2)),
                    validation_mse=score, online_half_mse=online_squared, online_l1_penalty=online_penalty,
                    first_batch_gradient_l2=gradient_norm,
                    train_prediction_std=float(train_prediction.std()),
                    validation_prediction_std=float(validation_prediction.std()))
                if not all(np.isfinite(value) for value in record.values()):
                    raise FloatingPointError("Nonfinite NN epoch diagnostics.")
                history.append(record)
                if score < best_score:
                    best_score, best_epoch, stale = score, epoch, 0
                    best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
                else:
                    stale += 1
                if progress is not None and (epoch == 1 or epoch % 5 == 0 or stale >= self.patience):
                    progress(f"seed={seed} epoch={epoch}: train MSE={record['train_mse']:.8g}, "
                             f"validation MSE={score:.8g}, best epoch={best_epoch}")
                if stale >= self.patience:
                    break
            network.load_state_dict(best_state, strict=True)
            network.cpu().eval()
            update = sum(float((value.detach().cpu() - initial[name]).square().sum())
                         for name, value in network.named_parameters()) ** .5
            dense_weights = torch.cat([layer.weight.detach().flatten() for layer in network.modules()
                                       if isinstance(layer, nn.Linear)])
            self.diagnostics_.append(dict(seed=seed, best_epoch=best_epoch, epochs_run=epoch,
                stop_reason="patience" if stale >= self.patience else "max_epochs",
                best_validation_mse=best_score, initial_train_mse=initial_mse,
                initial_validation_mse=float(np.mean((vy - initial_validation)**2)),
                parameter_update_l2=update, dense_weight_l1=float(dense_weights.abs().sum()),
                nonzero_dense_weights=int(torch.count_nonzero(dense_weights)),
                dense_weight_count=dense_weights.numel(), history=history))
            self.models_.append(network)
        return self

    def predict(self, X):
        if not hasattr(self, "models_"):
            raise ValueError("Fit the NN before prediction.")
        if isinstance(X, pd.DataFrame) and self.feature_names_ is not None:
            if tuple(X.columns) != self.feature_names_:
                raise ValueError("Prediction feature order differs from training.")
        X = _matrix(X, self.n_features_in_)
        prediction = np.zeros(len(X), dtype=np.float64)
        for network in self.models_:
            prediction += predict_network(network, X, device=self.device, batch_size=self.predict_batch_size)
            network.cpu()
        return prediction / len(self.models_)

    def export_state(self):
        """JSON numeric arrays only, including BN buffers; no executable pickle."""
        return dict(version=1, n_features=self.n_features_in_, hidden_sizes=list(self.hidden_sizes),
                    dtype="float32", feature_names=self.feature_names_,
                    predict_batch_size=self.predict_batch_size,
                    members=[{key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}
                             for model in self.models_])

    @classmethod
    def from_state(cls, state, *, device="cpu"):
        if state.get("version") != 1 or state.get("dtype") != "float32" or not state.get("members"):
            raise ValueError("Unsupported or empty NN saved state.")
        model = cls(hidden_sizes=state["hidden_sizes"], device=device,
                    predict_batch_size=state["predict_batch_size"])
        model.n_features_in_ = state["n_features"]
        model.feature_names_ = tuple(state["feature_names"]) if state.get("feature_names") is not None else None
        model.models_ = []
        for member in state["members"]:
            network = ReturnNetwork(model.n_features_in_, model.hidden_sizes)
            expected = network.state_dict()
            if member.keys() != expected.keys():
                raise ValueError("Saved NN state keys differ from its architecture.")
            restored = {}
            for key, reference in expected.items():
                value = torch.tensor(member[key], dtype=reference.dtype)
                if value.shape != reference.shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid saved NN tensor {key}.")
                restored[key] = value
            network.load_state_dict(restored, strict=True)
            model.models_.append(network.eval())
        return model
