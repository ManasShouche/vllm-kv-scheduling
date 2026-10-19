"""Regression for reporting stalls on real streaming event counts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab.analyze import plots


class PlotScaleTests(unittest.TestCase):
    def test_large_stream_retains_every_gap_without_per_event_artists(self):
        ticks = np.arange(10001, dtype=float) / 100
        row = {"event_times_s": ticks.tolist(), "actual_dispatch_s": 0,
               "arrival_s": 0, "status": "ok"}
        meta = {"labels": {}, "config": {"ttft_slo_s": 2.5,
                "event_gap_slo_s": .1, "phases": [{"duration_s": 120}]}}
        with tempfile.TemporaryDirectory() as tmp, patch("matplotlib.pyplot.close"):
            try:
                plots(Path(tmp), meta, [row], [])
                fig = plt.gcf()
                self.assertEqual(len(fig.axes[1].collections), 1)
                offsets = fig.axes[1].collections[0].get_offsets()
                np.testing.assert_allclose(offsets[:, 0], ticks[1:])
                np.testing.assert_allclose(offsets[:, 1], np.diff(ticks))
                self.assertGreater((Path(tmp)/"timeline.png").stat().st_size, 0)
            finally:
                fig = plt.gcf()
        plt.close(fig)
