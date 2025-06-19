import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import seaborn as sb
    from sensitivity import global_sensitivity, plot_mu, plot_sigma

    from rma_kinetics.models import DoxPK, DoxPKConfig
    from diffrax import diffeqsolve, Kvaerno3, PIDController, SaveAt, ODETerm
    from jax import config
    from matplotlib.ticker import FuncFormatter

    config.update("jax_enable_x64", True)
    sb.set_context('talk')
    return (
        DoxPK,
        DoxPKConfig,
        FuncFormatter,
        Kvaerno3,
        ODETerm,
        PIDController,
        SaveAt,
        diffeqsolve,
        global_sensitivity,
        jnp,
        mo,
        plot_mu,
        plot_sigma,
        plt,
        sb,
    )


@app.cell
def _(DoxPK, DoxPKConfig):
    dox_hyclate_percent = 0.87 # dox hyclate ~ 87% dox
    mouse_weight = 0.03 # kg
    dox_pk = DoxPK(
        DoxPKConfig(
            vehicle_intake_rate=1.875e-4, # mg food / hr - 4.5 mg / day
            bioavailability=0.90,
            vehicle_dose=625 * dox_hyclate_percent, # mg / kg food
            absorption_rate=0.8, # 1/hr
            elimination_rate=0.2, # 1 / hr
            brain_transport_rate=0.2, # 1/ hr
            plasma_transport_rate=1, # 1 / hr
            t0=0,
            t1=72,
            plasma_vd=0.7*mouse_weight # L
        )
    )
    return dox_hyclate_percent, dox_pk, mouse_weight


@app.cell
def _(Kvaerno3, ODETerm, PIDController, SaveAt, diffeqsolve, dox_pk, jnp):
    plasma_dox_ss =dox_pk.absorption_rate/dox_pk.elimination_rate*dox_pk.intake_rate
    brain_dox_ss = dox_pk.brain_transport_rate / dox_pk.plasma_transport_rate * plasma_dox_ss
    sol = diffeqsolve(
        ODETerm(dox_pk._model),
        t0=0,
        t1=168,
        dt0=0.1,
        y0=(0, 0),
        stepsize_controller=PIDController(atol=1e-6, rtol=1e-6),
        solver=Kvaerno3(),
        saveat=SaveAt(ts=jnp.linspace(0, 168, 168*2))
    )
    return (sol,)


@app.cell
def _(FuncFormatter, jnp, plt, sb):
    def plot_dox_concentration(time: jnp.ndarray, dox_conc: tuple[jnp.ndarray, jnp.ndarray]) -> None:
        fig, ax = plt.subplots()
        formatter = FuncFormatter(lambda y, _: f'{y * 1e-3:.1f}')
        ax.yaxis.set_major_formatter(formatter)

        plt.plot(time, dox_conc[0], color='tab:red', label='Plasma')
        plt.plot(time, dox_conc[1], color='tab:blue', label='Brain')

        ax.set_ylabel('[Dox] µM')
        ax.set_xlabel('Time (hr)')

        plt.legend(frameon=False)

        sb.despine()
        plt.tight_layout()
        plt.show()
    return (plot_dox_concentration,)


@app.cell
def _(plot_dox_concentration, sol):
    plot_dox_concentration(sol.ts, sol.ys)
    return


@app.cell
def _():
    # sensitivity analysis

    vehicle_intake_range = [1.67e-4, 2.5e-4] # kg food / hr
    bioavailability_range = [0.7, 0.95]
    absorption_range = [0.55, 1.57] # 0.85 +- 0.41 hrs
    elimination_range = [0.02, 0.2] # ~3-25 hr half life
    brain_plasma_partition_range = [0.1, 0.3] # 10-30% brain to plasma concentration ratio
    volume_dist_range = [0.7, 1.4]

    param_space = {
        "num_vars": 6,
        "names": [
            "vehicle_intake_rate",
            "bioavailability",
            "absorption_rate",
            "elimination_rate",
            "brain_plasma_partition_range",
            "volume_dist"
        ],
        "bounds": [
            vehicle_intake_range,
            bioavailability_range,
            absorption_range,
            elimination_range,
            brain_plasma_partition_range,
            volume_dist_range
        ],
        "outputs": "Y"
    }
    return (param_space,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Fixed parameters

    - dox chow dose: 625 mg dox / kg food
    - dox hyclate fraction ~ 87%
    - plasma transport rate = 1 / hr (brain transport rate is based on brain penetration)
    - 30 g mouse
    """
    )
    return


@app.cell
def _(
    DoxPK,
    DoxPKConfig,
    Kvaerno3,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
    dox_hyclate_percent,
    jnp,
    mouse_weight,
):
    t0 = 0
    t1 = 144
    fs = 2

    def map_dox_model(params: jnp.ndarray):
        model = DoxPK(
            DoxPKConfig(
                vehicle_intake_rate=params[0],
                bioavailability=params[1],
                vehicle_dose=625 * dox_hyclate_percent, # mg / kg food
                absorption_rate=params[2],
                elimination_rate=params[3],
                brain_transport_rate=params[4],
                plasma_transport_rate=1,
                t0=0,
                t1=72,
                plasma_vd=params[5]*mouse_weight
        ))

        sol = diffeqsolve(
            ODETerm(model._model),
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=(0, 0),
            stepsize_controller=PIDController(atol=1e-6, rtol=1e-6),
            solver=Kvaerno3(),
            saveat=SaveAt(ts=jnp.linspace(t0, t1, t1*fs))
        )

        return sol.ys[1]
    return fs, map_dox_model, t0, t1


@app.cell
def _(global_sensitivity, map_dox_model, param_space):
    sols, morris = global_sensitivity(map_dox_model, param_space, 1250)
    return morris, sols


@app.cell
def _(fs, jnp, morris, sols, t0, t1):
    x = jnp.linspace(t0, t1, t1*fs)
    mean_sol = jnp.mean(sols, axis=0)
    morris_means = jnp.array([s['mu_star'] for s in morris])
    morris_conf = jnp.array([s['mu_star_conf'] for s in morris])
    morris_std = jnp.array([s['sigma'] for s in morris])
    return morris_conf, morris_means, morris_std, x


@app.cell
def _(morris_conf, morris_means, plot_mu, plt, sb, x):
    labels = [
        "Iv", # vehicle intake rate
        "F", # bioavailability
        "Ka", # absorption rate
        "Kel", # elimination rate
        "Kb", # brain transport rate (brain penetration)
        "Vd", # plasma volume of distribution
    ]

    mu_fig, mu_ax = plot_mu(x, morris_means, morris_conf, labels)

    plt.xlabel("Time (hr)")
    plt.ylabel("Morris µ*")
    plt.legend(frameon=False, bbox_to_anchor=(1,1))
    sb.despine()
    plt.tight_layout()
    plt.show()
    return (labels,)


@app.cell
def _(labels, morris_std, plot_sigma, plt, sb, x):
    sigma_fig, sigma_ax = plot_sigma(x, morris_std, labels)

    plt.xlabel("Time (hr)")
    plt.ylabel("Morris std.")
    plt.legend(frameon=False, bbox_to_anchor=(1,1))
    sb.despine()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
