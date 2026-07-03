from __future__ import annotations

import numpy as np
import pandas as pd

from beatvecnet.evaluation import aggregate_beat_roi_well


def test_hierarchical_aggregation_roi_equal_weight_with_unequal_beats():
    df = pd.DataFrame(
        [
            {"task_name": "t", "well_id": "w1", "roi_id": "r1", "y_true": 0, "prob_0": 1.0, "prob_1": 0.0, "prob_2": 0.0, "prob_3": 0.0},
            {"task_name": "t", "well_id": "w1", "roi_id": "r1", "y_true": 0, "prob_0": 1.0, "prob_1": 0.0, "prob_2": 0.0, "prob_3": 0.0},
            {"task_name": "t", "well_id": "w1", "roi_id": "r1", "y_true": 0, "prob_0": 1.0, "prob_1": 0.0, "prob_2": 0.0, "prob_3": 0.0},
            {"task_name": "t", "well_id": "w1", "roi_id": "r2", "y_true": 0, "prob_0": 0.0, "prob_1": 1.0, "prob_2": 0.0, "prob_3": 0.0},
        ]
    )
    roi, well = aggregate_beat_roi_well(df, n_classes=4)
    r1 = roi[roi["roi_id"] == "r1"].iloc[0]
    r2 = roi[roi["roi_id"] == "r2"].iloc[0]
    assert np.allclose([r1.prob_0, r1.prob_1], [1.0, 0.0])
    assert np.allclose([r2.prob_0, r2.prob_1], [0.0, 1.0])
    assert np.allclose([well.iloc[0].prob_0, well.iloc[0].prob_1], [0.5, 0.5])
