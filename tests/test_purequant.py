import math
import unittest

from purequant import (attribution, backtest, bonds, derivatives, factor,
                       futures, hedge, linalg, optimize, risk, stats)
from purequant.types import OptionRight, PriceSeries


class TestLinalg(unittest.TestCase):
    def test_solve_and_inverse(self):
        a = [[3.0, 2.0], [1.0, 2.0]]
        x = linalg.solve(a, [7.0, 5.0])
        self.assertAlmostEqual(x[0], 1.0, places=6)
        self.assertAlmostEqual(x[1], 2.0, places=6)
        inv = linalg.inverse(a)
        ident = linalg.matmul(a, inv)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(ident[i][j], 1.0 if i == j else 0.0, places=6)

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            linalg.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


class TestStats(unittest.TestCase):
    def test_cov_corr(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        self.assertAlmostEqual(stats.correlation(x, y), 1.0, places=6)

    def test_ols_recovers_slope(self):
        x = list(range(10))
        y = [2.0 * xi + 1.0 for xi in x]
        res = stats.ols(y, [x], add_intercept=True)
        self.assertAlmostEqual(res.beta[0], 2.0, places=4)
        self.assertAlmostEqual(res.intercept, 1.0, places=4)
        self.assertAlmostEqual(res.r2, 1.0, places=6)

    def test_zscore_zero_std(self):
        self.assertEqual(stats.zscore([5.0, 5.0, 5.0]), [0.0, 0.0, 0.0])


class TestDerivatives(unittest.TestCase):
    def test_put_call_parity(self):
        s, k, t, r, sig = 100.0, 95.0, 0.5, 0.03, 0.25
        c = derivatives.bsm_price(s, k, t, r, sig, OptionRight.CALL)
        p = derivatives.bsm_price(s, k, t, r, sig, OptionRight.PUT)
        # c - p == s - k*e^{-rt}
        self.assertAlmostEqual(c - p, s - k * math.exp(-r * t), places=4)

    def test_call_delta_bounds(self):
        g = derivatives.bsm_greeks(100, 100, 1, 0.03, 0.2, OptionRight.CALL)
        self.assertTrue(0 < g.delta < 1)
        self.assertGreater(g.gamma, 0)

    def test_string_right_matches_enum(self):
        # "call"/"put" strings must behave identically to the OptionRight enum.
        for right in ("call", "put"):
            s = derivatives.bsm_price(100, 100, 1, 0.03, 0.2, right)
            e = derivatives.bsm_price(100, 100, 1, 0.03, 0.2, OptionRight(right))
            self.assertAlmostEqual(s, e, places=9)
        self.assertAlmostEqual(
            derivatives.bsm_price(100, 100, 1, 0.03, 0.2, "call"), 9.4134, places=3)

    def test_implied_vol_roundtrip(self):
        price = derivatives.bsm_price(100, 100, 1, 0.03, 0.3, OptionRight.CALL)
        iv = derivatives.implied_vol(price, 100, 100, 1, 0.03, OptionRight.CALL)
        self.assertAlmostEqual(iv, 0.3, places=3)


class TestBonds(unittest.TestCase):
    def test_par_bond_prices_at_par(self):
        b = bonds.Bond(face=100, coupon_rate=0.05, years_to_maturity=10, freq=2)
        self.assertAlmostEqual(bonds.price_from_yield(b, 0.05), 100.0, places=4)

    def test_ytm_inverts_price(self):
        b = bonds.Bond(face=100, coupon_rate=0.04, years_to_maturity=7, freq=2)
        price = bonds.price_from_yield(b, 0.035)
        self.assertAlmostEqual(bonds.yield_to_maturity(b, price), 0.035, places=5)

    def test_duration_positive(self):
        b = bonds.Bond(face=100, coupon_rate=0.05, years_to_maturity=10, freq=2)
        r = bonds.analyze(b, 0.05)
        self.assertGreater(r.macaulay_duration, r.modified_duration)
        self.assertGreater(r.convexity, 0)


class TestFutures(unittest.TestCase):
    def test_fair_value_carry(self):
        self.assertGreater(futures.fair_value(100, r=0.05, t=1), 100)

    def test_index_hedge_short(self):
        h = futures.index_futures_hedge(1_000_000, 5000, 50)
        self.assertLess(h.contracts, 0)  # long book -> short hedge


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.rets = [-0.02, 0.01, 0.03, -0.015, 0.005, -0.04, 0.02, 0.01, -0.01, 0.025]

    def test_var_cvar_order(self):
        v = risk.historical_var(self.rets, 0.95)
        c = risk.cvar(self.rets, 0.95)
        self.assertGreaterEqual(c, v)

    def test_max_drawdown_known(self):
        self.assertAlmostEqual(risk.max_drawdown([-0.5, 0.0]), 0.5, places=6)

    def test_beta_of_self_is_one(self):
        self.assertAlmostEqual(risk.asset_beta(self.rets, self.rets), 1.0, places=6)

    def test_risk_contributions_sum_to_one(self):
        rets = {"A": [0.01, -0.02, 0.03, -0.01, 0.02],
                "B": [-0.01, 0.02, -0.015, 0.01, -0.02]}
        rc = risk.risk_contributions({"A": 0.6, "B": 0.4}, rets)
        self.assertAlmostEqual(sum(rc.contributions.values()), 1.0, places=6)


class TestOptimize(unittest.TestCase):
    def setUp(self):
        self.cov = [[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.01]]
        self.syms = ["A", "B", "C"]

    def test_min_variance_weights_sum_one(self):
        w = optimize.min_variance(self.cov, self.syms)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=4)
        self.assertGreater(w["C"], w["B"])  # lowest-vol asset gets most weight

    def test_risk_parity_equalizes(self):
        w = optimize.risk_parity(self.cov, self.syms)
        wl = [w[s] for s in self.syms]
        sw = linalg.matvec(self.cov, wl)
        rc = [wl[i] * sw[i] for i in range(3)]
        tot = sum(rc)
        for r in rc:
            self.assertAlmostEqual(r / tot, 1 / 3, places=2)


class TestFactor(unittest.TestCase):
    def test_ic_perfect(self):
        fv = {chr(65 + i): float(i) for i in range(6)}
        fr = {chr(65 + i): 0.01 * i for i in range(6)}
        ic = factor.information_coefficient(fv, fr)
        self.assertAlmostEqual(ic.ic, 1.0, places=6)

    def test_thin_universe_no_crash(self):
        # single-name universe must not raise (z-score undefined -> neutral)
        prices = {"X": PriceSeries("X", [], [])}
        scored = factor.FactorModel().score(["X"], {}, prices, {"X": "Tech"})
        self.assertEqual(len(scored), 1)


class TestHedge(unittest.TestCase):
    def test_min_variance_hedge_reduces_var(self):
        a = [0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.025]
        h = [0.018, -0.012, 0.028, -0.019, 0.011, 0.013, -0.022]
        hr = hedge.min_variance_hedge_ratio(a, h)
        eff = hedge.hedge_effectiveness(a, h, hr)
        self.assertGreater(eff["variance_reduction"], 0)


class TestBacktest(unittest.TestCase):
    def test_rebalanced_runs(self):
        # two synthetic price series
        import math as _m
        n = 260
        p1 = [100 * (1.0003) ** i for i in range(n)]
        p2 = [100 * (1 + 0.0002 * _m.sin(i / 10)) ** 1 for i in range(n)]
        prices = {"A": PriceSeries("A", list(range(n)), p1),
                  "B": PriceSeries("B", list(range(n)), p2)}
        res = backtest.backtest_momentum(prices, ["A", "B"], lookback=60, top_n=1)
        self.assertTrue(len(res.equity_curve) > 1)


class TestAttribution(unittest.TestCase):
    def test_factor_attribution_shapes(self):
        port = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.015, -0.005]
        facs = {"mkt": [0.008, -0.018, 0.025, -0.012, 0.019, 0.001, 0.014, -0.006]}
        fa = attribution.factor_attribution(port, facs)
        self.assertIn("mkt", fa.exposures)


if __name__ == "__main__":
    unittest.main()
