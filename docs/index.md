# Home

RMA kinetics is a library providing models for simulating synthetic serum marker
dynamics in various context, including constitutive and drug induced expression.

Currently, there are three main models:

1. [ConstitutiveRMA](./api/models/constitutive.md)
2. [TetRMA](./api/models/tetoff.md)
3. [ChemogeneticRMA](./api/models/chemogenetic.md)

A basic [web application](https://rma-kinetics.up.railway.app) is available for
simple testing. Please see the [web app guide](web/guide.md) for more information.

## Installation

A python package is available on [PyPI](https://pypi.org/project/rma-kinetics/)
and can be installed with [uv](https://docs.astral.sh/uv/), pip,
and possibly other pip compatible alternatives.

To get started with uv, create a new project with `uv init` in a clean working
directory, or use `uv add` directly. For example,

### uv

```bash
uv add rma-kinetics
# or uv pip install rma-kinetics
```

The same can be done with standard pip,

### pip

```bash
pip install rma-kinetics
```

### Source

To install the package from source, make sure git is installed on your system and
follow the workflow detailed below.

1. Clone the repository

```bash
git clone https://github.com/szablowskilab/rma-kinetics.git
cd rma-kinetics
```

2. Create a new virtual environment. Below is an example using UV to sync
dependencies (Linux/Macos).

```bash
uv sync
source .venv/bin/activate # optional if using `uv add` later
```

3. Install the rma-kinetics package (`uv pip install -e .`).

You can then import the package in scripts or make modifications to source for
testing.

## Quick Start

```python
from rma_kinetics.models import ConstitutiveRMA
import matplotlib.pyplot as plt

# initialize constitutive RMA model
model = ConstitutiveRMA(
    rma_prod_rate=7e-3,
    rma_rt_rate=1,
    rma_deg_rate=7e-3
)

# simulate model and plot plasma RMA
solution = model.simulate(t0=0, t1=72, y0=(0,0))

print(f"Plasma RMA at final timepoint: {solution.plasma_rma[-1]}")
print(f"Brain RMA at final timepoint: {solution.brain_rma[-1]}")

solution.plot_plasma_rma()
plt.gcf()
```

All RMA models can be run by calling the `simulate` method detailed below.

::: rma_kinetics.models.AbstractModel.simulate
    options:
      show_source: false
      heading: "simulate"

Simulations return a [`Solution`](./api/solution.md) object which can be used to inspect the results.

Please see the [API reference](./api/models/constitutive.md) or [examples](https://github.com/szablowskilab/rma-kinetics/tree/main/examples) for more details.

## Citation

Will be updated shortly.
