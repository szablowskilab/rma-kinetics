from rma_kinetics.models import ConstitutiveRMA
from diffrax import PIDController, SaveAt
from jax import numpy as jnp

import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = ConstitutiveRMA(rma_prod_rate=7e-3, rma_rt_rate=0.6, rma_deg_rate=7e-3)

    # simulate 3 wks
    t0 = 0; t1 = 504
    results = model.simulate(
        t0=t0,
        t1=t1,
        dt0=1,
        stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
        saveat=SaveAt(ts=jnp.linspace(t0, t1, t1))
    )

    results.plot_plasma_rma()
    plt.show()
