from __future__ import annotations

import numpy as np
import pandas as pd

from beatvecnet.amplitude_normalization import day0_beat_scale, fit_day0_well_scales, normalize_manifest_tensors


def test_day0_scale_and_normalized_tensor(tmp_path):
    x1 = np.zeros((8, 2, 10), dtype=np.float32)
    x2 = np.zeros((8, 2, 10), dtype=np.float32)
    post = np.ones((8, 2, 10), dtype=np.float32) * 6.0
    x1[0, 0, -1] = 2.0
    x2[0, 0, -1] = 4.0
    for name, arr in [("d0a.npy", x1), ("d0b.npy", x2), ("post.npy", post)]:
        np.save(tmp_path / name, arr)
    assert day0_beat_scale(x1) == 2.0
    manifest = pd.DataFrame(
        {
            "path": [str(tmp_path / "d0a.npy"), str(tmp_path / "d0b.npy"), str(tmp_path / "post.npy")],
            "well_id": ["A", "A", "A"],
            "day": [0, 0, 7],
        }
    )
    scales = fit_day0_well_scales(manifest, repo_root=tmp_path, day_col="day")
    assert np.allclose(scales["day0_scale"].iloc[0], 3.0)
    out_manifest = normalize_manifest_tensors(manifest, scales, repo_root=tmp_path, output_dir=tmp_path / "out")
    norm_post = np.load(tmp_path / out_manifest["path"].iloc[2])
    assert np.allclose(norm_post, 2.0)
