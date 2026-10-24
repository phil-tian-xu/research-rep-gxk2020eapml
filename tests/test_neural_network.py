"""Numerical training checks, early stopping, leakage guards and portable state."""

import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from core.models.neural_network import NeuralNetworkRegressor, ReturnNetwork, penalized_loss


@unittest.skipUnless(HAS_TORCH, "Install the optional NN dependencies: uv sync --extra nn")
class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_objective_and_kernel_gradient_are_independent_of_bias_and_bn_penalties(self):
        model = ReturnNetwork(2, (3,))
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(.2)
                    layer.bias.fill_(7)
                if isinstance(layer, nn.BatchNorm1d):
                    layer.weight.fill_(9)
                    layer.bias.fill_(11)
        prediction = torch.tensor([.1, -.2], requires_grad=True)
        target = torch.tensor([.3, .4])
        squared, penalty = penalized_loss(model, prediction, target, .03)
        self.assertAlmostEqual(float(squared.detach()), .5 * np.mean([.2**2, .6**2]), places=7)
        self.assertAlmostEqual(float(penalty.detach()), .03 * (2*3 + 3) * .2, places=7)
        penalty.backward()
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                assert_allclose(layer.weight.grad.numpy(), .03)
                self.assertIsNone(layer.bias.grad)
        with self.assertRaisesRegex(ValueError, "shapes"):
            penalized_loss(model, prediction[:, None], target, .03)

    def test_synthetic_signal_learning_reproducibility_and_json_round_trip(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(320, 4)).astype(np.float32)
        y = .2 * np.maximum(X[:, 0], 0) - .15 * X[:, 1] - .1
        options = dict(hidden_sizes=(16,), seeds=(71, 72), learning_rate=.01, l1=0.,
                       batch_size=64, max_epochs=100, patience=15)
        first = NeuralNetworkRegressor(**options).fit(X[:256], y[:256], X_validation=X[256:], y_validation=y[256:])
        second = NeuralNetworkRegressor(**options).fit(X[:256], y[:256], X_validation=X[256:], y_validation=y[256:])
        prediction = first.predict(X[256:])
        self.assertLess(np.mean((prediction - y[256:])**2), .003)
        self.assertGreater(prediction.std(), .05)
        self.assertTrue(all(d["parameter_update_l2"] > .01 for d in first.diagnostics_))
        assert_array_equal(prediction, second.predict(X[256:]))
        restored = NeuralNetworkRegressor.from_state(json.loads(json.dumps(first.export_state())))
        assert_array_equal(prediction, restored.predict(X[256:]))
        from core.models.neural_network import predict_network
        individual = [predict_network(network, X[256:], device="cpu", batch_size=10000)
                      for network in first.models_]
        assert_array_equal(prediction, np.mean(individual, axis=0))
        self.assertFalse(np.array_equal(individual[0], individual[1]))
        for network in restored.models_:
            self.assertFalse(network.training)
            before = {k: v.clone() for k, v in network.state_dict().items()}
            restored.predict(X)
            for key, value in network.state_dict().items():
                assert_array_equal(value.numpy(), before[key].numpy())
        state = first.export_state()
        state["members"][0]["layers.0.weight"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "tensor"):
            NeuralNetworkRegressor.from_state(state)

    def test_early_stopping_restores_best_weights_and_bn_buffers_and_keeps_singleton(self):
        from core.models.neural_network import predict_network
        rng = np.random.default_rng(5)
        X = rng.normal(size=(33, 3)).astype(np.float32)
        V = rng.normal(size=(17, 3)).astype(np.float32)
        scores = iter([1., .4, .2, .5, .6])

        def controlled(network, values, **kwargs):
            if len(values) == 17:
                return np.full(17, next(scores))
            return predict_network(network, values, **kwargs)

        with patch("core.models.neural_network.predict_network", side_effect=controlled):
            model = NeuralNetworkRegressor(seeds=(4,), batch_size=16, max_epochs=20, patience=2).fit(
                X, X[:, 0] * .1, X_validation=V, y_validation=np.zeros(17))
        info = model.diagnostics_[0]
        self.assertEqual((info["best_epoch"], info["epochs_run"], info["stop_reason"]), (2, 4, "patience"))
        self.assertAlmostEqual(info["best_validation_mse"], .04)
        bn = next(layer for layer in model.models_[0].modules() if isinstance(layer, nn.BatchNorm1d))
        self.assertEqual(int(bn.num_batches_tracked), 4)  # Two batches x best epoch 2.

    def test_signed_linear_output_and_invalid_options(self):
        from core.models.neural_network import ARCHITECTURES
        for depth in range(1, 6):
            model = ReturnNetwork(3, ARCHITECTURES[f"NN{depth}"]).eval()
            dense = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
            self.assertEqual([layer.out_features for layer in dense], list((32, 16, 8, 4, 2)[:depth]) + [1])
            with torch.no_grad():
                model.layers[-1].weight.zero_()
                model.layers[-1].bias.fill_(-.01)
            assert_allclose(model(torch.zeros(2, 3)).detach().numpy(), -.01)
        for options in (dict(batch_size=1), dict(learning_rate=0), dict(l1=-1), dict(seeds=(1, 1))):
            with self.assertRaises(ValueError):
                NeuralNetworkRegressor(**options)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available() if HAS_TORCH else False, "CUDA unavailable")
    def test_cuda_training_and_cpu_restoration(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(80, 4)).astype(np.float32)
        model = NeuralNetworkRegressor(seeds=(6,), device="cuda", batch_size=32, max_epochs=4).fit(
            X[:64], X[:64, 0] * .1, X_validation=X[64:], y_validation=X[64:, 0] * .1)
        restored = NeuralNetworkRegressor.from_state(model.export_state())
        assert_allclose(model.predict(X), restored.predict(X), atol=1e-6, rtol=1e-5)
        self.assertGreater(model.diagnostics_[0]["parameter_update_l2"], 0)


if __name__ == "__main__":
    unittest.main()
