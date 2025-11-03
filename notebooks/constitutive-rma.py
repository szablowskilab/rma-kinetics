import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import ConstitutiveRMA
    from gluc import get_gluc_conc
    from sensitivity import global_sensitivity
    from diffrax import PIDController, SaveAt, Kvaerno5, diffeqsolve
    from diffopt.multiswarm import ParticleSwarm, get_best_loss_and_params
    from functools import partial
    from jax import (
        numpy as jnp,
        config,
    )
    from sklearn.metrics import r2_score

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
        diffeqsolve,
        get_best_loss_and_params,
        get_gluc_conc,
        global_sensitivity,
        jnp,
        mo,
        os,
        partial,
        pl,
        plt,
        r2_score,
        sb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## RMA datasets

    Measurements were made at 0, 2, and 3 weeks after delivery of AAV-hSyn-RMA-IRES-GFP. Each dataset corresponds to the targeted brain region (CA1 hippocampus, caudate putamen, or substantia nigra).
    """
    )
    return


@app.cell
def _(mo):
    dataset = mo.ui.radio(options=["CA1", "CP", "SN"], value="CA1", label="Dataset")
    dataset
    return (dataset,)


@app.cell
def _(dataset, get_gluc_conc, os, pl):
    base_dir = os.path.join("notebooks", "data", "aav_rma_timecourse")
    data_dir = os.path.join(base_dir,dataset.value)
    raw_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
    gluc_df = get_gluc_conc(raw_df, 'rma')
    return base_dir, data_dir, gluc_df


@app.cell
def _(dataset, gluc_df, plt, sb):
    sb.barplot(gluc_df, x="time", y="gluc", errorbar="sd", color="lightgrey")
    plt.title(f"AAV-hSyn-Gluc {dataset.value} Timecourse")
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fitting course-grain ODE model to *in vivo* RMA measurements""")
    return


@app.cell
def _(
    ConstitutiveRMA,
    Kvaerno5,
    PIDController,
    ParticleSwarm,
    SaveAt,
    diffeqsolve,
    get_best_loss_and_params,
    gluc_df,
    jnp,
    partial,
    pl,
):
    def loss(params, args):
        observed, sim_config = args
        model = ConstitutiveRMA(*params)
        predicted = diffeqsolve(model._terms(), **sim_config)
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
        (1e-5, 1), # prod
        (0.54, 1), # rt
        (0.0048630532, 0.0104337257) # deg
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
    return best_loss, best_params, observed, sim_config


@app.cell
def _(best_loss, best_params, data_dir, jnp, os):
    print(f"Prod rate: {best_params[0]}")
    print(f"RT rate: {best_params[1]}")
    print(f"Deg rate: {best_params[2]}")
    mse = best_loss / 3
    print(f"MSE: {mse}")
    jnp.save(os.path.join(data_dir, "param_est.npy"), best_params)
    return


@app.cell
def _(gluc_df, pl):
    gluc_var = gluc_df.group_by("time").agg([
        pl.col("gluc").var().alias("gluc_var")
    ])
    gluc_var
    return


@app.cell
def _(
    ConstitutiveRMA,
    SaveAt,
    best_params,
    data_dir,
    diffeqsolve,
    gluc_df,
    jnp,
    observed,
    os,
    plt,
    r2_score,
    sb,
    sim_config,
):
    # visual inspection

    sim_config["saveat"] = SaveAt(ts=jnp.linspace(0, 504, 504))
    model = ConstitutiveRMA(*best_params)
    solution = diffeqsolve(model._terms(), **sim_config)

    plt.plot(solution.ts, solution.ys[1], 'k')
    plt.errorbar(gluc_df["time"], gluc_df["gluc"], color="k", fmt="o")
    _time = [0, 336, 504]
    predicted = jnp.array([solution.ys[1][t]for t in _time])
    r2 = r2_score(observed, predicted)
    print(f"R2: {r2}")


    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "fit.svg"))
    jnp.save(os.path.join(data_dir, "solution.npy"), solution.ys[1])
    plt.gca()
    return (solution,)


@app.cell
def _(jnp):
    jnp.std(jnp.array([4.86e-3, 6.47e-3, 1.04e-2]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Initial sensitivity analysis

    Morris sensitivity is used initially to determine relative importance of each parameter.
    """
    )
    return


@app.cell
def _(
    ConstitutiveRMA,
    best_params,
    diffeqsolve,
    global_sensitivity,
    jnp,
    sim_config,
):
    # sensitivity analysis
    def map_model(params):
        model = ConstitutiveRMA(*params)
        solution = diffeqsolve(model._terms(), **sim_config)
        return solution.ys[1]

    range = jnp.array([-0.5, 0.5])
    #best_params[2] = 0.6 # test fast deg
    #best_params[0] = best_params[0] / 10 # test slow prod
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
    plt.tight_layout()
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Sensitivity at final time point""")
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
def _(data_dir, jnp, mu_conf, mu_star, os, param_labels, plt, sb):
    # normalize sensitivies
    norm_mu_star = mu_star[-1] / jnp.max(mu_star[-1])
    norm_mu_conf = mu_conf[-1] / jnp.max(mu_star[-1])

    # normalized sensitivity at Tf
    plt.bar(param_labels, [
        norm_mu_star[0],
        norm_mu_star[1],
        norm_mu_star[2],
    ],
    yerr=[
        norm_mu_conf[0],
        norm_mu_conf[1],
        norm_mu_conf[2],
    ], color='lightgrey', width=0.5)


    plt.ylabel("Relative Ranking")

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_tf.svg"))
    plt.gca()

    return norm_mu_conf, norm_mu_star


@app.cell
def _(mu_conf, mu_star):
    print(f"kRMA mu*: {mu_star[-1,0]} +- {mu_conf[-1, 0]}")
    print(f"kRT mu*: {mu_star[-1,1]} +- {mu_conf[-1, 1]}")
    print(f"gammaRMA mu*: {mu_star[-1,2]} +- {mu_conf[-1, 2]}")
    return


@app.cell
def _(norm_mu_conf, norm_mu_star):
    print(f"norm kRMA mu*: {norm_mu_star[0]} +- {norm_mu_conf[0]}")
    print(f"norm kRT mu*: {norm_mu_star[1]} +- {norm_mu_conf[1]}")
    print(f"norm gammaRMA mu*: {norm_mu_star[2]} +- {norm_mu_conf[2]}")
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
    plt.savefig(os.path.join(data_dir, "morris_std_tf"))
    plt.gca()
    return


@app.cell
def _(sigma):
    print(f"kRMA std: {sigma[-1,0]}")
    print(f"kRT std: {sigma[-1,1]}")
    print(f"gammaRMA std: {sigma[-1,2]}")
    return


@app.cell
def _(data_dir, jnp, os, param_labels, plt, sb, sigma):
    # normalize morris std
    norm_sigma = sigma[-1] / jnp.max(sigma[-1])
    plt.bar(param_labels, [
        norm_sigma[0],
        norm_sigma[1],
        norm_sigma[2],
    ],color='lightgrey', width=0.5)


    plt.ylabel("Relative Nonlinearity or Interaction")

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_tf.svg"))
    plt.gca()
    return (norm_sigma,)


@app.cell
def _(norm_sigma):
    print(f"norm kRMA std: {norm_sigma[0]}")
    print(f"norm kRT std: {norm_sigma[1]}")
    print(f"norm gammaRMA std: {norm_sigma[2]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""All fits for each brain region shown on a single plot along with measurements""")
    return


@app.cell
def _(base_dir, dataset, get_gluc_conc, jnp, os, pl, plt, sb, solution):
    # show fits on same plot
    _colors = sb.color_palette(n_colors=3)
    _shapes = ["o", "s", "^"]
    handles = []
    labels = []

    order = ["SN", "CA1", "CP"]

    #_colors = ["blue", "orange", "green"]
    for _i, _dataset in enumerate(dataset.options.keys()):

        print(_dataset) 
        _raw_df = pl.read_csv(os.path.join(base_dir, _dataset, "source.csv"))
        _gluc_df = get_gluc_conc(_raw_df, 'rma')
        _mean_gluc = _gluc_df.group_by("time").agg([
            pl.col("gluc").mean().alias("mean_gluc"),
            pl.col("gluc").std().alias("std_gluc")
        ])

        _solution = jnp.load(os.path.join(base_dir, _dataset, "solution.npy"))

        line, = plt.plot(solution.ts, _solution, color=_colors[_i])
        plt.errorbar(_mean_gluc["time"], _mean_gluc["mean_gluc"], yerr=_mean_gluc["std_gluc"], fmt=_shapes[_i], color=_colors[_i], alpha=0.5)
        handles.append(line)
        labels.append(_dataset)



    #plt.legend(["CA1", "CP", "SN"], frameon=False)

    order_idx = [labels.index(lbl) for lbl in order]
    handles = [handles[i] for i in order_idx]
    labels = [labels[i] for i in order_idx]

    plt.legend(handles, labels, frameon=False)

    sb.despine()
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fit.svg"))
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
