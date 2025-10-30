import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import ConstitutiveRMA
    from sensitivity import global_sensitivity
    from diffrax import PIDController, SaveAt, Kvaerno5, diffeqsolve
    from jax import (
        numpy as jnp,
        config,
    )

    import marimo as mo
    import seaborn as sb
    import matplotlib.pyplot as plt
    import os

    config.update("jax_enable_x64", True)
    sb.set_theme(context="talk", style="ticks", font="Arial")
    data_dir = os.path.join("notebooks", "data", "aav_rma_timecourse", "avg_sensitivity")

    return (
        ConstitutiveRMA,
        Kvaerno5,
        PIDController,
        SaveAt,
        data_dir,
        diffeqsolve,
        global_sensitivity,
        jnp,
        os,
        plt,
        sb,
    )


@app.cell
def _(
    ConstitutiveRMA,
    Kvaerno5,
    PIDController,
    SaveAt,
    diffeqsolve,
    global_sensitivity,
    jnp,
):
    # sensitivity analysis
    sim_config = {
        "t0": 0,
        "t1": 504,
        "dt0": 0.1,
        "y0": (0,0),
        "saveat": SaveAt(ts=jnp.linspace(0, 504, 504)),
        "stepsize_controller": PIDController(atol=1e-5, rtol=1e-5),
        "solver": Kvaerno5()
    }

    def map_model(params):
        model = ConstitutiveRMA(*params)
        solution = diffeqsolve(model._terms(), **sim_config)
        return solution.ys[1]

    range = jnp.array([-0.5, 0.5])
    #best_params[2] = 0.6 # test fast deg
    #best_params[0] = best_params[0] / 10 # test slow prod
    avg_params = jnp.array([8.06e-3, 6.78e-1, 9.53e-3])
    param_space = {
        "num_vars": 3,
        "names": ["rma_prod_rate", "rma_rt_rate", "rma_deg_rate"],
        "bounds": [p * (1 + range) for p in avg_params],
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
def _(mu_conf, mu_star, plt, sb, time):
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
    plt.gca()

    return linestyles, param_labels


@app.cell
def _(linestyles, param_labels, plt, sb, sigma, time):
    for _i, _label in enumerate(param_labels):
        _sigma = sigma[:, _i]
        plt.plot(time, _sigma, label=_label, linestyle=linestyles[_i])


    plt.xlabel("Time (hr)")
    plt.ylabel("Std. Morris Sensitivity, $\\sigma$")
    plt.legend(frameon=False)

    sb.despine()
    plt.tight_layout()
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
def _(norm_mu_conf, norm_mu_star):
    print(f"norm kRMA mu*: {norm_mu_star[0]} +- {norm_mu_conf[0]}")
    print(f"norm kRT mu*: {norm_mu_star[1]} +- {norm_mu_conf[1]}")
    print(f"norm gammaRMA mu*: {norm_mu_star[2]} +- {norm_mu_conf[2]}")
    return


@app.cell
def _(jnp, mu_conf, mu_star):
    mu_star_6hrs = mu_star[6,:]
    max_mu_star_6hrs = jnp.max(mu_star_6hrs)
    norm_mu_star_6hrs = mu_star_6hrs / max_mu_star_6hrs
    norm_mu_conf_6hrs = mu_conf[6,:] / max_mu_star_6hrs 
    print(f"norm kRMA mu* (6hr): {norm_mu_star_6hrs[0]} +- {norm_mu_conf_6hrs[0]}")
    print(f"norm kRT mu* (6hr): {norm_mu_star_6hrs[1]} +- {norm_mu_conf_6hrs[1]}")
    print(f"norm gammaRMA mu* (6hr): {norm_mu_star_6hrs[2]} +- {norm_mu_conf_6hrs[2]}")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
