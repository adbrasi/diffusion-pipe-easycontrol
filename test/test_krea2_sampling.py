import math
import unittest

from tools.krea2_sampling import build_krea2_timesteps


class Krea2SamplingTest(unittest.TestCase):
    def test_schedule_matches_resolution_endpoints(self):
        low, low_mu = build_krea2_timesteps(16 * 16, 4)
        high, high_mu = build_krea2_timesteps(80 * 80, 4)

        self.assertAlmostEqual(low_mu, 0.5)
        self.assertAlmostEqual(high_mu, 1.15)
        self.assertEqual(low[0], 1.0)
        self.assertEqual(low[-1], 0.0)
        self.assertTrue(all(a > b for a, b in zip(low, low[1:])))
        self.assertTrue(all(a > b for a, b in zip(high, high[1:])))

    def test_512_resolution_uses_interpolated_mu(self):
        # 512 / (VAE f8 * patch 2) = 32 tokens per side.
        timesteps, mu = build_krea2_timesteps(32 * 32, 4)

        self.assertAlmostEqual(mu, 0.58125)
        expected_mid = math.exp(mu) * 0.5 / (1.0 + (math.exp(mu) - 1.0) * 0.5)
        self.assertAlmostEqual(timesteps[2], expected_mid)

    def test_explicit_mu_is_respected(self):
        _, mu = build_krea2_timesteps(32 * 32, 8, mu=1.15)
        self.assertEqual(mu, 1.15)

    def test_invalid_inputs_fail_early(self):
        with self.assertRaises(ValueError):
            build_krea2_timesteps(0, 8)
        with self.assertRaises(ValueError):
            build_krea2_timesteps(1024, 0)


if __name__ == '__main__':
    unittest.main()
