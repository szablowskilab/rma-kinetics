# RMA Kinetics

Source code for "Modeling kinetics of synthetic serum markers for monitoring deep tissue gene expression".

## Getting Started

### Basic Usage

There are three main model classes for constitutive or drug induced RMA expression.

1. ConstitutiveRMA.
2. TetRMA (Tet-Off based marker expression).
3. ChemogeneticRMA (CNO induced marker expression with Tet-Off gating).

Models can be initialized from these classes and ran using the `simulate` method.

For example, to run the constitutive RMA expression model for 72 hours,

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
solution.plot_plasma_rma()
plt.gcf()
```

## Web App

To run locally, install marimo (available in the development dependencies) in
your virtual environment and run `marimo run app/main.py` or use the provided Just
command `just app-serve` to start the server.
Alternatively, if using uvx, `uvx marimo run app/main.py`.

## Notebooks

Notebooks can be found in the `notebooks` directory and require installing
development dependencies.

Note that the `diffopt` package requires OpenMPI. Please install openMPI or use
the provided [Nix shell](./shell.nix) (i.e., with `nix shell` from the project root).

