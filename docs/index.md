# Home

RMA kinetics is a library providing models for simulating synthetic serum marker
dynamics in various context, including constitutive and drug induced expression.

Currently, there are three main models:

1. [ConstitutiveRMA](./api/models/constitutive.md)
2. [TetRMA](./api/models/tetoff.md)
3. [ChemogeneticRMA](./api/models/chemogenetic.md)

## Installation

### uv

```bash
uv add rma-kinetics
# or uv pip install rma-kinetics
```

### pip

```bash
pip install rma-kinetics
```

### Source

Clone the repository

```bash
git clone https://github.com/szablowskilab/rma-kinetics.git
cd rma-kinetics
```

Create a new virtual environment. Below is an example using UV to sync
dependencies (Linux/Macos).

```bash
uv sync
source .venv/bin/activate
```

Install the rma-kinetics package (`uv pip install -e .`).



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
