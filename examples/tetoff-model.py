from rma_kinetics.models import TetRMA, DoxPKConfig
from diffrax import SaveAt
from jax import numpy as jnp

import matplotlib.pyplot as plt

if __name__ == "__main__":
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
