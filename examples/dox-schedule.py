from rma_kinetics.models import TetRMA, DoxPKConfig
from diffrax import SaveAt
from jax import numpy as jnp, config as jax_config
jax_config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

"""
The DoxPKConfig only allows for setting up a single dox administration window.
If we want to setup a schedule for repeated dosing of dox, we can easily
do that by looping over the administration times and calling `simulate`
for each dose.
"""

if __name__ == "__main__":
    # setup the dox administration periods. For example, let's say we want
    # to simulate 3-weeks total and administer dox in the 1st and 3rd week
    # each dose here is mg/kg food and will be used for a week
    dox_schedule = [
        (0, 168),
        (0,0), # no dox adminstration here (we also set the dose to 0 mg/kg)
        (0, 168)
    ]

    # if for some reason we wanted to also change the dose, we could do that
    # as well and zip the schedules with the doses.
    # dox_doses = [
    #     40,
    #     0,
    #     100
    # ]

    y0 = (0, 0, 10, 0, 0) # assuming tTA is at steady state already
    window_ts = jnp.linspace(0, 168, 168) # saving points every hour for each run
    # we'll accumulate the plasma RMA and brain dox to plot later
    plasma_rma = jnp.array([])
    brain_dox = jnp.array([])

    print("Simulating dox administration")
    for i, time in enumerate(dox_schedule):
        # dose at 40mg/kg for 1st week
        # no dox the 2nd week
        # 100mg/kg the 3rd week
        dox_model_config = DoxPKConfig(
            dose=40,
            t0=time[0],
            t1=time[1]
        )

        model = TetRMA(
            rma_prod_rate=7e-3,
            rma_rt_rate=0.6,
            rma_deg_rate=7e-3,
            dox_model_config=dox_model_config,
            dox_kd=10,
            tta_prod_rate=8e-2,
            tta_deg_rate=8e-3,
            tta_kd=10,
            leaky_rma_prod_rate=7e-5
        )

        print(f"period {i+1}")
        solution = model.simulate(t0=0, t1=168, y0=y0, saveat=SaveAt(ts=window_ts))
        plasma_rma = jnp.concatenate([plasma_rma, solution.plasma_rma])
        brain_dox = jnp.concatenate([brain_dox, solution.brain_dox])

        # update the state for the next run
        y0 = (
            solution.brain_rma[-1],
            solution.plasma_rma[-1],
            solution.tta[-1],
            solution.brain_dox[-1],
            solution.plasma_dox[-1]
        )

    # plot RMA and dox
    full_time = jnp.linspace(0, 504, 504)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time (hr)')
    ax1.set_ylabel('Plasma RMA (nM)', color=color)
    ax1.plot(full_time, plasma_rma, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Brain Dox (nM)', color=color)
    ax2.plot(full_time, brain_dox, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()
