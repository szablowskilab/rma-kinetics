# Constitutive RMA

::: rma_kinetics.models.ConstitutiveRMA
    options:
      members:
        - simulate
        - _model

## Example

```python
from rma_kinetics.models import ConstitutiveRMA
import matplotlib.pyplot as plt

model = ConstitutiveRMA(5e-3, 0.6, 7e-3)
solution = model.simulate(t0=0, t1=72, y0=(0,0))
solution.plot_plasma_rma()
plt.gcf()

# we can also get plasma and brain RMA solutions directly
plasma_rma = solution.plasma_rma
brain_rma = solution.brain_rm
```
