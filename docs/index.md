# Home

RMA kinetics is a library providing models for simulating synthetic serum marker
dynamics in various context, including constitutive and drug induced expression.

Currently, there are three main models:

1. [ConstitutiveRMA](./api/models/constitutive.md)
2. [TetRMA](./api/models/tetoff.md)
3. [ChemogeneticRMA](./api/models/chemogenetic.md)

## Installation

```bash
uv add rma_kinetics

# or with standard pip
# pip install rma_kinetics
```

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

**Method:** `simulate`

:::rma_kinetics.models.AbstractModel.simulate

Please see the [API reference](./api/models/constitutive.md) or [examples](https://github.com/szablowskilab/rma-kinetics/tree/main/examples) for more details.

## Citation

If you found this library useful, please cite: [(bioarxiv link)]()

```bibtex
@article{buitrago2025rma
  title={Modeling synthetic serum markers for monitoring deep tissue gene expression},
  author={Nicolas Buitrago, Josefina Brau, Jerzy Szablowski},
  year={2025},
}
```
Also consider starring the project on [GitHub](https://github.com/szablowskilab/rma-kinetics).
