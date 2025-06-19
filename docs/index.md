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
```

All RMA models can be run by calling the `simulate` method detailed below.

### Method: `simulate`

:::rma_kinetics.models.AbstractModel.simulate

Please see the [API reference]() or [examples]() for more details.

## Citation

If you found this library useful, please cite: [(bioarxiv link)]()

```bibtex
@paper{buitrago2025rma
  title={Modeling synthetic serum markers for monitoring deep tissue gene expression},
  author={Nicolas Buitrago, Josefina Brau, Jerzy Szablowski},
  year={2025},
}
```
Also consider starring the project on [GitHub]().
