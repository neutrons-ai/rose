"""
Tests for rose.planner.instrument.
"""

from __future__ import annotations

import numpy as np
import pytest

from rose.planner.instrument import InstrumentSimulator, load_measurement

# ── InstrumentSimulator defaults ─────────────────────────────────


class TestInstrumentDefaults:
    def test_default_q_grid_length(self):
        sim = InstrumentSimulator()
        assert len(sim.q_values) == 50

    def test_default_dq_scalar(self):
        sim = InstrumentSimulator()
        assert np.allclose(sim.dq_values, 0.025)

    def test_default_relative_error(self):
        sim = InstrumentSimulator()
        assert np.allclose(sim.relative_errors, 0.10)


# ── Custom Q grid ────────────────────────────────────────────────


class TestCustomQ:
    def test_explicit_q(self):
        q = np.linspace(0.01, 0.2, 20)
        sim = InstrumentSimulator(q_values=q, dq_values=0.02)
        assert len(sim.q_values) == 20
        assert np.allclose(sim.dq_values, 0.02)

    def test_explicit_q_array_dq(self):
        q = np.linspace(0.01, 0.2, 10)
        dq = q * 0.05
        sim = InstrumentSimulator(q_values=q, dq_values=dq)
        assert np.allclose(sim.dq_values, dq)

    def test_mismatched_dq_raises(self):
        q = np.linspace(0.01, 0.2, 10)
        bad_dq = np.ones(5)
        with pytest.raises(ValueError, match="must match"):
            InstrumentSimulator(q_values=q, dq_values=bad_dq)


# ── add_noise ────────────────────────────────────────────────────


class TestAddNoise:
    def test_returns_tuple(self):
        sim = InstrumentSimulator()
        reflectivity = np.ones(50) * 0.5
        noisy, errors = sim.add_noise(reflectivity)
        assert noisy.shape == reflectivity.shape
        assert errors.shape == reflectivity.shape

    def test_noise_changes_signal(self):
        sim = InstrumentSimulator()
        reflectivity = np.ones(50) * 0.5
        noisy, _ = sim.add_noise(reflectivity)
        # Extremely unlikely that all values remain exactly the same
        assert not np.allclose(noisy, reflectivity)


# ── load_measurement ─────────────────────────────────────────────


class TestLoadMeasurement:
    def test_load_4col_file(self, tmp_path):
        data = np.column_stack(
            [
                np.linspace(0.01, 0.2, 10),
                np.ones(10) * 0.5,
                np.ones(10) * 0.01,
                np.ones(10) * 0.002,
            ]
        )
        fpath = tmp_path / "data.dat"
        np.savetxt(fpath, data)

        result = load_measurement(str(fpath))
        assert "q" in result
        assert len(result["q"]) == 10

    def test_too_few_columns_raises(self, tmp_path):
        data = np.column_stack(
            [
                np.linspace(0.01, 0.2, 10),
                np.ones(10),
            ]
        )
        fpath = tmp_path / "bad.dat"
        np.savetxt(fpath, data)

        with pytest.raises(ValueError, match="4 columns"):
            load_measurement(str(fpath))

    def test_simulator_from_file(self, tmp_path):
        data = np.column_stack(
            [
                np.linspace(0.01, 0.2, 10),
                np.ones(10) * 0.5,
                np.ones(10) * 0.01,
                np.ones(10) * 0.002,
            ]
        )
        fpath = tmp_path / "data.dat"
        np.savetxt(fpath, data)

        sim = InstrumentSimulator(data_file=str(fpath))
        assert len(sim.q_values) == 10
