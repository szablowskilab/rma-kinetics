import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import TetRMA, DoxPKConfig

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import seaborn as sb

    from sensitivity import global_sensitivity, plot_mu, plot_sigma
    from diffrax import PIDController, SaveAt

    from jax import config
    config.update("jax_enable_x64", True)

    sb.set_context("talk")
    return DoxPKConfig, PIDController, SaveAt, TetRMA, jnp, plt


@app.cell
def _(DoxPKConfig):
    # example sanity check
    dox_hyclate_percent = 0.87 # dox hyclate ~ 87% dox
    mouse_weight = 0.03 # kg

    dox_pk = DoxPKConfig(
        vehicle_intake_rate=1.875e-4, # mg food / hr - 4.5 mg / day
        bioavailability=0.90,
        vehicle_dose=625 * dox_hyclate_percent, # mg / kg food
        absorption_rate=0.8, # 1/hr
        elimination_rate=0.2, # 1 / hr
        brain_transport_rate=0.2, # 1/ hr
        plasma_transport_rate=1, # 1 / hr
        t0=0,
        t1=48,
        plasma_vd=0.7*mouse_weight, # L
    )
    return (dox_pk,)


@app.cell
def _(TetRMA, dox_pk):

    model = TetRMA(
        rma_prod_rate=5e-3,
        rma_rt_rate=1,
        rma_deg_rate=7e-3,
        dox_model_config=dox_pk,
        dox_kd=10,
        tta_prod_rate=8e-3, # nM/hr
        tta_deg_rate=8e-3, # 1 / hr
        tta_kd=1, # nM
        leaky_rma_prod_rate=5e-5
    )
    return (model,)


@app.cell
def _(PIDController, SaveAt, jnp, model):
    sim_t0 = 0
    sim_t1 = 144
    fs = 2

    sol = model.simulate(
        t0=0,
        t1=144,
        dt0=0.1,
        y0=(0, 0, 1, 0, 0),
        saveat=SaveAt(ts=jnp.linspace(sim_t0, sim_t1, sim_t1*fs)),
        stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
    )
    return (sol,)


@app.cell
def _(plt, sol):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(12,8))

    ax[0,0].plot(sol.ts, sol.ys[0], 'k')
    ax[0,0].set_ylabel("Brain RMA (nM)")

    ax[0,1].plot(sol.ts, sol.ys[1], 'k')
    ax[0,1].set_ylabel("Plasma RMA (nM)")

    ax[1,0].plot(sol.ts, sol.ys[3], 'k')
    ax[1,0].set_ylabel("Brain Dox (nM)")
    ax[1,0].set_xlabel("Time (hr)")

    ax[1,1].plot(sol.ts, sol.ys[4], 'k')
    ax[1,1].set_ylabel("Plasma Dox (nM)")
    ax[1,0].set_xlabel("Time (hr)")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(plt, sol):
    plt.plot(sol.ts, sol.ys[2])
    plt.xlabel("Time (hr)")
    plt.ylabel("tTA steady state (nM)")
    plt.gca()
    return


@app.cell
def _():
    # sensitivity analysis
    return


if __name__ == "__main__":
    app.run()
