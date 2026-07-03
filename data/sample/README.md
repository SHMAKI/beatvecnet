# Sample dataset documentation

## Public availability
This repository includes a small **sample dataset** under `data/sample/` for demonstration and code validation purposes.
The sample dataset is licensed under **CC BY-NC 4.0**.

The full dataset used in the study is not publicly released in this repository.

## Purpose of the sample dataset
The sample dataset is intended to:
- demonstrate the expected input format for the public code,
- allow users to run training and evaluation on a small example dataset,
- support validation of the repository setup.

It is **not** intended to reproduce the full statistical analyses reported in the paper.
It does not include the Well/ROI hierarchy metadata used for the primary study-level well evaluation.

## Recording vs beat resampling
- Raw videos were recorded at **150 fps**.
- Beat-gated resampling uses **150 phase bins per beat** (a normalized beat-phase grid).

These two quantities are different: the former refers to video acquisition, whereas the latter refers to temporal normalization of individual beats.

## Directory structure
The sample dataset is organized as follows:

```text
data/sample/
├── README.md
├── LICENSE
├── manifest.csv
└── tensors/
    ├── s0001.npy
    ├── s0002.npy
    └── ...
```

## Manifest format

`data/sample/manifest.csv` contains one row per sample.
Expected columns:
- sample_id: unique sample identifier
- split: dataset split (train, val, or test)
- path: relative path to the tensor file
- class: class label name
- class_id: integer class label
- cond_raw: original condition label

Optional hierarchy columns supported by the code, when present in external manifests:
- task_name
- well_id
- roi_id
- beat_id

The bundled sample manifest does not contain these optional hierarchy columns.

Example:
```csv
sample_id,split,path,class,class_id,cond_raw
s0001,test,data/sample/tensors/s0001.npy,Ctrl,0,Ctrl
s0002,test,data/sample/tensors/s0002.npy,DOX,1,DOX_0.2_uM
```

## Tensor format

Each file in data/sample/tensors/ is a NumPy array (.npy) representing one sample.

Expected shape:
- (C, H, T) = (8, 500, 150)

where:

- C = 8 input channels
- H = 500 spatial dimension
- T = 150 beat-phase bins

Depending on preprocessing history, some internal datasets may use a different axis order, but the public dataset loader in this repository converts supported inputs to (C, H, T) format.

## Notes

The sample dataset is intentionally small and simplified relative to the full internal dataset. The tensors are prebuilt public sample tensors for workflow validation and should not be day 0-normalized again for the README quickstart.

Internal-only source paths and infrastructure-specific metadata are not included in the public manifest.

Additional preprocessing pipelines used in the full study are not required to run the sample dataset.
