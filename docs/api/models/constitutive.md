# Constitutive RMA

Simple constituve expression of released markers of activity.

::: rma_kinetics.models.ConstitutiveRMA

## Example

```python
from rma_kinetics.models import ConstitutiveRMA
import matplotlib.pyplot as plt

model = ConstitutiveRMA(5e-3, 0.6, 7e-3)
solution = model.run(t0=0, t1=72)
solution.plot_plasma_rma()
plt.gcf()

# we can also get plasma and brain RMA solutions directly
plasma_rma = solution.plasma_rma()
brain_rma = solution.brain_rma()
# ...
```

## Model Equations

$$\begin{align}
\dot{[RMA_{B}]} &= k_{RMA} - k_{RT}[RMA_{B}] \tag{1} \\
\dot{[RMA_{P}]} &= k_{RT} - \gamma_{RMA}[RMA_{P}] \tag{2}
\end{align}
$$
