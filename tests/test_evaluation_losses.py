import unittest
from functools import partial

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from core.losses import huber_loss, mse_loss


class PredictionLossTests(unittest.TestCase):
    def test_known_values_and_common_callable_interface(self):
        observed = np.array([1., 2., 3.])
        predicted = np.array([1., 4., 0.])
        for loss_fn, expected in ((mse_loss, 13 / 3), (partial(huber_loss, delta=1.), 8 / 3)):
            self.assertAlmostEqual(loss_fn(observed, predicted), expected)
            self.assertIsInstance(loss_fn(observed, predicted), float)
        assert_array_equal(observed, [1., 2., 3.])
        assert_array_equal(predicted, [1., 4., 0.])

    def test_lists_and_series(self):
        self.assertEqual(mse_loss([1., 2.], [2., 3.]), 1.)
        self.assertEqual(mse_loss(pd.Series([1., 2.], index=["a", "b"]),
                                  pd.Series([2., 3.], index=["a", "b"])), 1.)

    def test_shapes_missing_values_and_complex_inputs_rejected(self):
        for observed, predicted in (([1., 2.], [[1.], [2.]]), ([1.], [1., 2.]),
                                    (1., 1.), ([], []), ([np.nan], [1.]),
                                    ([1.], [np.inf]), ([1j], [1.])):
            for loss_fn in (mse_loss, partial(huber_loss, delta=1.)):
                with self.subTest(observed=observed, predicted=predicted), self.assertRaises(ValueError):
                    loss_fn(observed, predicted)

    def test_series_index_mismatch_is_not_silently_aligned(self):
        observed = pd.Series([1., 2.], index=["a", "b"])
        predicted = pd.Series([2., 1.], index=["b", "a"])
        for loss_fn in (mse_loss, partial(huber_loss, delta=1.)):
            with self.assertRaises(ValueError):
                loss_fn(observed, predicted)

    def test_huber_threshold_is_explicit_and_positive(self):
        with self.assertRaises(TypeError):
            huber_loss([0.], [1.])
        for delta in (None, 0, -1, np.inf, np.nan):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                huber_loss([0.], [1.], delta=delta)


if __name__ == "__main__":
    unittest.main()
