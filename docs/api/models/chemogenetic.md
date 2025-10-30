# ChemogeneticRMA

::: rma_kinetics.models.ChemogeneticRMA
    options:
      members:
        - simulate
        - _tet_rma_model
        - _model

## Example

```python
from rma_kinetics.models import ChemogeneticRMA, DoxPKConfig, CnoPKConfig

# add dox feeding at 30 mg/kg from time 0 oto 48 hours using the default rates.
dox_model_config = DoxPKConfig(dose=30, t0=0, t1=48)

# inject 1mg/kg CNO (assuming 30 g mouse).
# we'll also use the default rates here.
mouse_weight = 0.03
cno_model_config = CnoPKConfig(dose=1*mouse_weight)

model = ChemogeneticRMA(
    rma_prod_rate=7e-3,
    rma_rt_rate=0.6,
    rma_deg_rate=7e-3,
    dox_model_config=dox_model_config,
    dox_kd=10,
    tta_prod_rate=8e-3,
    tta_deg_rate=8e-3,
    tta_kd=1,
    cno_model_config=cno_model_config,
    cno_t0=48
)

y0 = (
    0, 0, 0,
    dox_model_config.brain_dox_ss, dox_model_config.plasma_dox_ss,
    0, 0, 0, 0, 0
)

solution = model.simulate(0, 96, y0)
```
