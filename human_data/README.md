# Mental Rotation VR — Experimental Human Data
This dataset contains behavioral data from mental rotation experiments in [VR environment](https://github.com/BRAIN-Aalto/VR-mental-rotation/tree/dev) collected for [A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](https://arxiv.org/abs/2512.13517).

## File: `mr_vr_no_action.csv`

Experimental data from a Shepard-Metzler mental rotation task in a VR environment. Subjects are presented with two 3D objects at different poses — related by a rotation in depth along the vertical y-axis — and judge whether they are identical (match) or mirror images (mismatch).

## File: `mr_vr_action.csv`

Experimental data from the same Shepard-Metzler mental rotation task, with the addition that subjects can optionally rotate one of the objects (in depth along the vertical y-axis) via thumbstick interaction to assist their similarity judgment.



## Column Descriptions
### Core Trial Columns (both  `mr_vr_action.csv` and  `mr_vr_no_action.csv`)

| Column | Type | Description |
|---|---|---|
| `session_ID` | int | Unique identifier for each experimental session |
| `trial_ID` | int | Unique identifier for each trial within a session |
| `trial_condition` | str | Ground truth similarity condition of the trial (`Match` or `Mismatch`)|
| `subject_ID` | str | Randomly generated anonymous identifier (hex format, e.g. `0x00`) |
| `cohort_ID` | int | Cohort group the subject belongs to (-1, 1, or 2) |
| `response_time_sec` | float | Time taken by the subject to make a similarity decision, in seconds |
| `response_decision` | str | Subject's similarity decision (`Match` or `Mismatch`) |
| `shape_description` | str | Symbolic description of the Shepard-Metzler object |
| `angular_disparity` | int | Angular difference between the two presented objects, in degrees |
| `object_angles` | dict | Initial presentation angle of each object in the stimulus, in degrees (e.g. `{'1': 38.65, '2': 278.65}`) |
| `vr_setup` | str | VR setup condition (`Action`/`No-Action`)|

### Action-Specific Columns (`mr_vr_action.csv` only)

| Column | Type | Description |
|---|---|---|
| `thumbstick_angle` | list | Time series of cumulative rotation angles during thumbstick interaction, in degrees |
| `thumbstick_time_sec` | list | Timestamps of each thumbstick angle sample, in seconds |
| `action_count` | int | Total number of actions taken during the trial |
| `action_angles` | dict | Rotation angle spanned by each action if any, in degrees (e.g. `{0: 20.0, 1: 51.23}`) |
| `action_durations` | dict | Duration of each action taken, if any |
| `first_action_onset` | float | Timestamp of the start of the first action, if any |
| `last_action_offset` | float | Timestamp of the end of the last action, if any |
| `action_time_intervals` | dict | Time intervals between each action taken, if any |


## Citation
```bibtex
@article{khazoum2025deep,
  title={A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments},
  author={Khazoum, Raymond and Fernandes, Daniela and Krylov, Aleksandr and Li, Qin and Deny, Stephane},
  journal={arXiv preprint arXiv:2512.13517},
  year={2025}
}
```