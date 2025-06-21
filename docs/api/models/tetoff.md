# Tet-Off RMA

Tetracycline transcriptional activator (tTA) induced RMA expression model.

::: rma_kinetics.models.TetRMA

## Example

```python
from rma_kinetics.models import TetRMA, DoxPKConfig
from diffrax import SaveAt
from jax import numpy as jnp

# add dox feeding at 30 mg/kg food from time 0 to 48 hours
dox_model_config = DoxPKConfig(
    dose=30,
    t0=0,
    t1=48
)

# make a simple TetOff RMA model with no leaky expression
model = TetRMA(
    rma_prod_rate=7e-3,
    rma_rt_rate=0.6,
    rma_deg_rate=7e-3,
    dox_model_config=dox_model_config,
    dox_kd=10,
    tta_prod_rate=8e-3,
    tta_deg_rate=8e-3,
    tta_kd=1,
)

# simulate from 0 to 96 hours
t0 = 0; t1 = 96

# brain and plasma dox steady state concentrations
brain_dox_ss = dox_model_config.brain_dox_ss
plasma_dox_ss = dox_model_config.plasma_dox_ss

# initial conditions
# species order is brain RMA, plasma RMA, tTA, brain dox, plasma dox
y0 = (0, 0, 1, brain_dox_ss, plasma_dox_ss)
solution = model.simulate(
    t0=t0,
    t1=t1,
    dt0=0.1,
    y0=y0,
    saveat=SaveAt(ts=jnp.linspace(t0, t1, t1))
)

# print the plasma RMA concentration at the final timepoint
plasma_rma = solution.plasma_rma
print(f"Plasma RMA at {t1} hours: {plasma_rma[-1]:.3f} nM")

# plot the plasma RMA trajectory
solution.plot_plasma_rma()
plt.show()
```

## Model Equations

Note that this model assumes constitutive expression of tTA.

$$\begin{align}
\dot{[TA]} &= k_{TA} - \gamma_{TA}[TA] \tag{1} \\
[TA]_{SS} &= \frac{k_{TA}}{\gamma_{TA}} \tag{2}
\end{align}
$$

Doxycycline is the preferred inhibitor (although tetracycline or other
derivatives may be used by updating the [DoxPKConfig](./dox/config.md)).
The fraction of the transcriptional activator available for inducing RMA
expression is then modeled with a Hill function,

$$\begin{align}
\theta_{tTA} &= \frac{1}{1 + \frac{[Dox]}{K_{D_{Dox}}}} \tag{3} \\
\dot{[RMA_{B}]} &= \frac{\beta_{0_{RMA}} + \beta_{RMA}\left(\frac{\theta_{TA}[TA]}{K_{D_{TA}}}\right)^{n_{tTA}}}{1 + \left(\frac{\theta_{TA}[TA]}{K_{D_{TA}}}\right)^{n_{tTA}}} - k_{RT}[RMA_{B}] \tag{4} \\
\dot{[RMA_{P}]} &= k_{RT}[RMA_{B}] - \gamma_{RMA}[RMA_{P}] \tag{5}
\end{align}
$$
