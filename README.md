<div align="center">
<h1>A Deep Learning Model of Mental Rotation <br> Informed by Interactive VR Experiments</h1>

<a href="https://github.com/rkhz/menrot"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="Code"></a>
<a href="https://arxiv.org/abs/2512.13517"><img src="https://img.shields.io/badge/arXiv-2512.13517-b31b1b" alt="arXiv"></a>

[Raymond Khazoum](https://scholar.google.com/citations?user=JBUB_kwAAAAJ&hl=en), [Daniela Fernandes](https://scholar.google.com/citations?user=YMlpki8AAAAJ&hl=en), [Aleksandr Krylov](https://github.com/aleksandr-krylov), [Qin Li](https://openreview.net/profile?id=~Qin_Li12), [Stéphane Deny](https://scholar.google.com/citations?user=Z3jU6mQAAAAJ&hl=en)

**Aalto University**

</div>

## Overview
We propose a mechanistic deep neural model of human mental rotation that combines equivariant and symbolic representations, showing that the model solves the task while recapitulating human behavioral patterns.

## Installation
Requires Python `3.12+`.
```bash
# Clone the repository
git clone https://github.com/rkhz/menrot.git

# Move into the directory
cd menrot

# Create and activate virtual environment
python -m venv .venv   
source .venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Build Dataset
```bash
python scripts/build_dataset.py --task <task> --split <split>
```

| Argument | Values |
|----------|--------|
| `--task` | `renderer` (for Module I), `symbolic` (for Module II), `cognitive` (for Module III) |
| `--split` | `train`, `val`, `test` |


Data is written to `$WRKDIR/data/menrot_<task>/<split>/`. If `WRKDIR` is not set, you will be prompted for a path, or set it explicitly:

```bash
export WRKDIR=/path/to/data
```

### 2. Train Modules
> TODO

### 3. Evaluation on the Mental Rotation Task
> TODO


## Project Layout
```
menrot/
├── menrot/
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── menrot_cognitive.py
│   │   ├── menrot_renderer.py
│   │   └── menrot_symbolic.py
│   ├── distributed/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── models/
│   │   ├── configs/
│   │   │   └── vsm_encoder.json
│   │   ├── __init__.py
│   │   ├── eqnr.py
│   │   ├── mlp.py
│   │   └── vsm.py
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── __init__.py
│   │   │   ├── conv_block.py
│   │   │   ├── decoder.py
│   │   │   ├── encoder.py
│   │   │   ├── projection.py
│   │   │   ├── res_block.py
│   │   │   ├── rotation.py
│   │   │   ├── spherical_mask.py
│   │   │   └── transformers.py
│   │   ├── __init__.py
│   │   └── functional.py
│   ├── optim/
│   │   ├── __init__.py
│   │   └── scheduler_handler.py
│   ├── representations/
│   │   ├── __init__.py
│   │   ├── spatial.py
│   │   └── symbolic.py
│   ├── utils/
│   │   ├── data/
│   │   │   ├── configs/
│   │   │   │   ├── objects_shape-test.csv
│   │   │   │   ├── objects_shape-train.csv
│   │   │   │   ├── objects_shape-val.csv
│   │   │   │   ├── shape-test.csv
│   │   │   │   ├── shape-train.csv
│   │   │   │   └── shape-val.csv
│   │   │   ├── __init__.py
│   │   │   ├── builder.py
│   │   │   ├── dataset.py
│   │   │   ├── distributed.py
│   │   │   └── sampler.py
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   └── config.py
│   └── __init__.py
├── human_data/
│   ├── mr_vr_action.csv
│   ├── mr_vr_no_action.csv
│   └── README.md
├── scripts/
│   └── build_dataset.py
├── LICENSE
├── README.md
└── requirements.txt
```


## Related Repositories
- [VR Mental Rotation](https://github.com/BRAIN-Aalto/VR-mental-rotation/tree/dev) - VR application used to conduct the human study
- [Shepard and Metzler 3D Shape Renderer](https://github.com/aleksandr-krylov/shepard-metzler-shape-renderer) - Renderer used to generate the stimuli 


## Citation
```bibtex
@article{khazoum2025deep,
  title={A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments},
  author={Khazoum, Raymond and Fernandes, Daniela and Krylov, Aleksandr and Li, Qin and Deny, Stephane},
  journal={arXiv preprint arXiv:2512.13517},
  year={2025}
}
```