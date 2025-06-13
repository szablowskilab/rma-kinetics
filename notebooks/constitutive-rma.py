import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import ConstitutiveRMA
    from gluc import get_gluc_conc
    from sensitivity import global_sensitivity
    from diffrax import PIDController, SaveAt, Kvaerno5
    from diffopt.multiswarm import ParticleSwarm, get_best_loss_and_params
    from functools import partial
    from jax import (
        numpy as jnp,
        config,
    )

    import marimo as mo
    import polars as pl
    import seaborn as sb
    import matplotlib.pyplot as plt
    import os

    config.update("jax_enable_x64", True)
    sb.set_theme(context="talk", style="ticks", font="Arial")
    return (
        ConstitutiveRMA,
        Kvaerno5,
        PIDController,
        ParticleSwarm,
        SaveAt,
        get_best_loss_and_params,
        get_gluc_conc,
        global_sensitivity,
        jnp,
        mo,
        os,
        partial,
        pl,
        plt,
        sb,
    )


@app.cell
def _(mo):
    dataset = mo.ui.radio(options=["CA1", "CP", "SN"], value="CA1", label="Dataset")
    dataset
    return (dataset,)


@app.cell
def _(dataset, get_gluc_conc, os, pl):
    data_dir = os.path.join("notebooks","data","aav_rma_timecourse",dataset.value)
    raw_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
    gluc_df = get_gluc_conc(raw_df)
    return data_dir, gluc_df


@app.cell
def _(dataset, gluc_df, plt, sb):
    sb.barplot(gluc_df, x="time", y="gluc", errorbar="sd", color="lightgrey")
    plt.title(f"AAV-hSyn-Gluc {dataset.value} Timecourse")
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.gca()
    return


@app.cell
def _(
    ConstitutiveRMA,
    Kvaerno5,
    PIDController,
    ParticleSwarm,
    SaveAt,
    get_best_loss_and_params,
    gluc_df,
    jnp,
    partial,
    pl,
):
    def loss(params, args):
        observed, sim_config = args
        model = ConstitutiveRMA(*params)
        predicted = model.simulate(**sim_config)
        return jnp.sum((observed - predicted.ys[1])**2)

    sim_config = {
        "t0": 0,
        "t1": 504,
        "dt0": 0.1,
        "y0": (0,0),
        "saveat": SaveAt(ts=jnp.array([0, 336, 504])),
        "stepsize_controller": PIDController(atol=1e-5, rtol=1e-5),
        "solver": Kvaerno5()
    }

    bounds = [
        (0, 1), # prod
        (0.54, 1), # rt
        (4e-3, 1e-2) # deg
    ]


    mean_gluc = gluc_df.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc")
    ])

    observed = mean_gluc.sort("time")["mean_gluc"].to_jax()

    loss_fn = partial(loss, args=(observed, sim_config))
    swarm = ParticleSwarm(nparticles=100, ndim=3, xlow=[b[0] for b in bounds], xhigh=[b[1] for b in bounds])
    pso_result = swarm.run_pso(loss_fn)
    best_loss, best_params = get_best_loss_and_params(
        pso_result["swarm_loss_history"],
        pso_result["swarm_x_history"]
    )
    return best_loss, best_params, sim_config


@app.cell
def _(best_loss, best_params, data_dir, jnp, os):
    print(f"Prod rate: {best_params[0]}")
    print(f"RT rate: {best_params[1]}")
    print(f"Deg rate: {best_params[2]}")
    mse = best_loss / 3
    print(f"MSE: {mse}")
    jnp.save(os.path.join(data_dir, "param_est.npy"), best_params)
    return (mse,)


@app.cell
def _(
    ConstitutiveRMA,
    SaveAt,
    best_params,
    data_dir,
    gluc_df,
    jnp,
    mse,
    os,
    plt,
    sb,
    sim_config,
):
    # visual inspection
    sim_config["saveat"] = SaveAt(ts=jnp.linspace(0, 504, 504))
    model = ConstitutiveRMA(*best_params)
    solution = model.simulate(**sim_config)

    plt.plot(solution.ts, solution.ys[1], 'k')
    plt.errorbar(gluc_df["time"], gluc_df["gluc"], color="k", fmt="o")

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    plt.text(0.05, 0.99, f"MSE = {mse:.3e}",
             transform=plt.gca().transAxes,
             ha='left', va='top', fontsize=16)
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "fit.svg"))
    plt.gca()
    return


@app.cell
def _(ConstitutiveRMA, best_params, global_sensitivity, jnp, sim_config):
    # sensitivity analysis
    def map_model(params):
        model = ConstitutiveRMA(*params)
        solution = model.simulate(**sim_config)
        return solution.ys[1]

    range = jnp.array([-0.5, 0.5])
    param_space = {
        "num_vars": 3,
        "names": ["rma_prod_rate", "rma_rt_rate", "rma_deg_rate"],
        "bounds": [p * (1 + range) for p in best_params],
        "outputs": "Y"
    }

    morris_y, morris_sens = global_sensitivity(map_model, param_space, 250)
    time = jnp.linspace(sim_config["t0"], sim_config["t1"], sim_config["t1"])
    y_mean = jnp.mean(morris_y, axis=0)
    mu_star = jnp.array([s['mu_star'] for s in morris_sens])
    mu_conf = jnp.array([s['mu_star_conf'] for s in morris_sens])
    sigma = jnp.array([s['sigma'] for s in morris_sens])
    return mu_conf, mu_star, sigma, time


@app.cell
def _(data_dir, mu_conf, mu_star, os, plt, sb, time):
    param_labels = ["$k_{RMA}$", "$k_{RT}$", "$\\gamma_{RMA}$"]
    linestyles = ["-", ":", "--"]

    for _i, _label in enumerate(param_labels):
        _mu_star = mu_star[:,_i]
        _mu_conf = mu_conf[:,_i]
        plt.plot(time, _mu_star, label=_label, linestyle=linestyles[_i])
        plt.fill_between(
            time,
            _mu_star - _mu_conf,
            _mu_star + _mu_conf,
            alpha=0.25
        )

    plt.xlabel("Time (hr)")
    plt.ylabel("Mean Morris Sensitivity, $µ^*$")
    plt.legend(frameon=False)
    sb.despine()
    plt.savefig(os.path.join(data_dir, "morris_mean.svg"))
    plt.gca()
    return linestyles, param_labels


@app.cell
def _(data_dir, linestyles, os, param_labels, plt, sb, sigma, time):
    for _i, _label in enumerate(param_labels):
        _sigma = sigma[:, _i]
        plt.plot(time, _sigma, label=_label, linestyle=linestyles[_i])


    plt.xlabel("Time (hr)")
    plt.ylabel("Std. Morris Sensitivity, $\\sigma$")
    plt.legend(frameon=False)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, mu_conf, mu_star, os, param_labels, plt, sb):
    # sensitivity at Tf
    plt.bar(param_labels, [
        mu_star[-1,0],
        mu_star[-1,1],
        mu_star[-1,2],
    ],
    yerr=[
        mu_conf[-1,0],
        mu_conf[-1,1],
        mu_conf[-1,2],
    ], color='lightgrey')


    plt.ylabel("Mean Morris Sensitivty, $µ^*$")

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean_tf.svg"))
    plt.gca()
    return


@app.cell
def _(mu_conf, mu_star):
    print(f"kRMA mu*: {mu_star[-1,0]} +- {mu_conf[-1, 0]}")
    print(f"kRT mu*: {mu_star[-1,1]} +- {mu_conf[-1, 1]}")
    print(f"gammaRMA mu*: {mu_star[-1,2]} +- {mu_conf[-1, 2]}")
    return


@app.cell
def _(data_dir, os, param_labels, plt, sb, sigma):
    plt.bar(param_labels, [
        sigma[-1,0],
        sigma[-1,1],
        sigma[-1,2],
    ],color='lightgrey')


    plt.ylabel("Std. Morris Sensitivty, $\\sigma$")

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std_tf.svg"))
    plt.gca()
    return


@app.cell
def _(sigma):
    print(f"kRMA std: {sigma[-1,0]}")
    print(f"kRT std: {sigma[-1,1]}")
    print(f"gammaRMA std: {sigma[-1,2]}")
    return


if __name__ == "__main__":
    app.run()
