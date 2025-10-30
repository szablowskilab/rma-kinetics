import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import TetRMA, DoxPKConfig

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import seaborn as sb
    import os

    from sensitivity import global_sensitivity, plot_mu, plot_sigma
    from diffrax import PIDController, SaveAt

    from jax import config

    config.update("jax_enable_x64", True)
    sb.set_theme("talk", style="ticks", font="Arial")
    data_dir = os.path.join("notebooks", "data", "tetoff")
    return (
        DoxPKConfig,
        PIDController,
        TetRMA,
        data_dir,
        global_sensitivity,
        jnp,
        os,
        plt,
        sb,
    )


@app.cell
def _(DoxPKConfig):
    # example sanity check
    dox_hyclate_percent = 0.87 # dox hyclate ~ 87% dox
    mouse_weight = 0.03 # kg

    dox_pk = DoxPKConfig(
        vehicle_intake_rate=1.875e-4, # mg food / hr - 4.5 mg / day
        bioavailability=0.90,
        dose=625 * dox_hyclate_percent, # mg / kg food
        absorption_rate=0.8, # 1/hr
        elimination_rate=0.2, # 1 / hr
        brain_transport_rate=0.2, # 1/ hr
        plasma_transport_rate=1, # 1 / hr
        t0=0,
        t1=48,
        plasma_vd=0.7*mouse_weight, # L
    )
    return dox_hyclate_percent, dox_pk, mouse_weight


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
def _(PIDController, model):
    sim_t0 = 0
    sim_t1 = 144
    fs = 2

    sol = model.simulate(
        t0=0,
        t1=144,
        dt0=0.1,
        y0=(0, 0, 1, 0, 0),
        sampling_rate=fs,
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
    ax[1,1].set_xlabel("Time (hr)")

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
def _(
    DoxPKConfig,
    PIDController,
    TetRMA,
    data_dir,
    dox_hyclate_percent,
    jnp,
    mouse_weight,
    os,
    plt,
    sb,
):
    _fig, _ax = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(6.4, 6.4), sharex=True)
    colors = sb.color_palette(palette="Greys", n_colors=10)

    def map_model(dose, color):
        dox_pk = DoxPKConfig(
            vehicle_intake_rate=1.875e-4, # mg food / hr - 4.5 mg / day
            bioavailability=0.90,
            dose=dose * dox_hyclate_percent, # mg / kg food
            absorption_rate=0.8, # 1/hr
            elimination_rate=0.2, # 1 / hr
            brain_transport_rate=0.2, # 1/ hr
            plasma_transport_rate=1, # 1 / hr
            t0=0,
            t1=168,
            plasma_vd=0.7*mouse_weight, # L
        )

        model = TetRMA(
            rma_prod_rate=3e-3,
            rma_rt_rate=0.6,
            rma_deg_rate=7e-3,
            dox_model_config=dox_pk,
            dox_kd=10,
            tta_prod_rate=3, # nM/hr
            tta_deg_rate=0.3, # 1 / hr
            tta_kd=1, # nM
            leaky_rma_prod_rate=5e-5
        )

        plasma_dox_ss = dox_pk.absorption_rate / dox_pk.elimination_rate * dox_pk.intake_rate
        brain_dox_ss = dox_pk.plasma_transport_rate / dox_pk.brain_transport_rate * plasma_dox_ss
        plasma_dox_ss = 0
        brain_dox_ss = 0

        solution = model.simulate(
            t0=0,
            t1=504,
            dt0=0.1,
            y0=(0, 0, model.tta_prod_rate/model.tta_deg_rate, brain_dox_ss, plasma_dox_ss),
            sampling_rate=2,
            stepsize_controller=PIDController(atol=1e-6, rtol=1e-6)
        )

        if dose == 0:
            _ax[0].plot(solution.ts, solution.ys[1], label=dose, linestyle="--", color='lightgrey')
            _ax[1].plot(solution.ts, solution.ys[3], linestyle="--", color='lightgrey')
        else:
            _ax[0].plot(solution.ts, solution.ys[1], label=dose, color=color)
            _ax[1].plot(solution.ts, solution.ys[3], color=color)


    doses = jnp.array([0, 10, 40, 160, 625])

    for _i, _dose in enumerate(doses):
        map_model(_dose, colors[_i*2])

    _ax[0].axvline(x=168, color='lightgrey', linestyle=':')
    _ax[0].set_ylabel("Plasma RMA (nM)")
    _ax[0].set_xlabel("Time (hr)")
    _ax[0].legend(frameon=False, title="Dox mg/kg")

    _ax[1].set_ylabel("Dox (µM)")
    _ax[1].set_xlabel("Time (hr)")
    # dox concentrations are in nM, convert to µM for plot
    yticks = _ax[1].get_yticks()
    uM_labels = [y*1e-3 for y in yticks]
    _ax[1].set_yticks(yticks, labels=uM_labels)
    _ax[1].set_ylim([0, 10e3])

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "tetoff_trajectories.svg"))
    plt.gca()
    return


@app.cell
def _(DoxPKConfig, PIDController, TetRMA, jnp, mouse_weight):
    # sensitivity analysis
    def global_map(params):
        dox_pk = DoxPKConfig(
            vehicle_intake_rate=params[3], # mg food / hr - 4.5 mg / day
            bioavailability=params[4],
            dose=params[5], # mg / kg food
            absorption_rate=params[6], # 1/hr
            elimination_rate=params[7], # 1 / hr
            brain_transport_rate=params[8], # 1/ hr
            plasma_transport_rate=1, # 1 / hr
            t0=0,
            t1=168,
            plasma_vd=params[9], # L
        )

        model = TetRMA(
            rma_prod_rate=params[0],
            rma_rt_rate=params[1],
            rma_deg_rate=params[2],
            dox_model_config=dox_pk,
            dox_kd=params[9],
            tta_prod_rate=params[10], # nM/hr
            tta_deg_rate=params[11], # 1 / hr
            tta_kd=params[12], # nM
            leaky_rma_prod_rate=params[13]
        )

        plasma_dox_ss = dox_pk.absorption_rate / dox_pk.elimination_rate * dox_pk.intake_rate
        brain_dox_ss = dox_pk.plasma_transport_rate / dox_pk.brain_transport_rate * plasma_dox_ss

        solution = model.simulate(
            t0=0,
            t1=672,
            dt0=0.1,
            y0=(0, 0, model.tta_prod_rate/model.tta_deg_rate, brain_dox_ss, plasma_dox_ss),
            sampling_rate=2,
            stepsize_controller=PIDController(atol=1e-6, rtol=1e-6)
        )

        return solution.ys[1]

    range = jnp.array([-0.5, 0.5])
    param_space = {
        "num_vars": 15,
        "names": [
            "rma_prod_rate",
            "rma_rt_rate",
            "rma_deg_rate",
            "vehicle_intake_rate",
            "bioavailability",
            "vehicle_dose",
            "absorption_rate",
            "elimination_rate",
            "brain_transport_rate",
            "plasma_vd",
            "dox_kd",
            "tta_prod_rate",
            "tta_deg_rate",
            "tta_kd",
            "leaky_rma_prod_rate"
        ],
        "bounds": [
            [1e-3, 1e-2],
            0.6 * (1 + range),
            7e-3 * (1 + range),
            1.875e-4 * (1 + range),
            [0.75, 0.95],
            [10*0.87, 625*0.87],
            8 * (1 + range),
            0.2 * (1 + range),
            0.2 * (1 + range),
            [0.7*mouse_weight, 1.6*mouse_weight],
            [1, 10],
            [1, 10],
            [0.1, 1],
            [1, 10],
            7e-5 * (1 + range)
        ],
        "outputs": "Y"
    }

    return global_map, param_space


@app.cell
def _(global_map, global_sensitivity, jnp, param_space):
    morris_y, morris_sens = global_sensitivity(global_map, param_space, 250)
    mu_star = jnp.array([s['mu_star'] for s in morris_sens])
    mu_conf = jnp.array([s['mu_star_conf'] for s in morris_sens])
    sigma = jnp.array([s['sigma'] for s in morris_sens])

    return mu_conf, mu_star, sigma


@app.cell
def _(data_dir, jnp, mu_conf, mu_star, os, plt, sb):
    labels = [
        "$k_{RMA}$",
        "$k_{RT}$",
        "$\\gamma_{RMA}$",
        "$f_{intake}$",
        "$F_{Dox}$",
        "$C_{Dox}$",
        "$k_{a_{Dox}}$",
        "$k_{el_{Dox}}$",
        "$k_{PB_{Dox}}$",
        "$V_D$",
        "$K_{D_{Dox}}$",
        "$k_{tTA}$",
        "$\\gamma_{tTA}$",
        "$K_{D_{tTA}}$",
        "$k_{0_{RMA}}$"
    ]
    top_params = [2, 10, 11, 12, 13]
    time = jnp.linspace(0, 672, 1344)

    for _i in top_params:
        _mu_star = mu_star[:,_i]
        _mu_conf = mu_conf[:,_i]
        plt.plot(time, _mu_star, label=labels[_i])
        plt.fill_between(
            time,
            _mu_star - _mu_conf,
            _mu_star + _mu_conf,
            alpha=0.25
        )

    plt.xlabel("Time (hr)")
    plt.ylabel("Mean Morris Sensitivity, $µ^*$")
    plt.legend(frameon=False, bbox_to_anchor=(1,1))
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean.svg"))
    plt.gca()

    return labels, time, top_params


@app.cell
def _(data_dir, labels, os, plt, sb, sigma, time, top_params):
    for _i in top_params:
        plt.plot(time, sigma[:, _i], label=labels[_i])

    plt.xlabel("Time (hr)")
    plt.ylabel("Std. Morris Sensitivity, $\sigma$")
    plt.legend(frameon=False, bbox_to_anchor=(1,1))
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std.svg"))
    plt.gca()

    return


@app.cell
def _(data_dir, jnp, labels, mu_conf, mu_star, os, plt, sb):
    # sensitivity at day 1 post dox
    idx = jnp.linspace(0, 15, 15, dtype=int)
    _fig, _ax = plt.subplots()
    _ax.bar(
        labels, 
        [mu_star[192,i] for i in idx],
        yerr=[mu_conf[192,j] for j in idx],
        color='lightgrey'
    )

    plt.title("Day 1 post dox withdrawal")
    plt.ylabel("Mean Morris Sensitivty, $µ^*$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean_day_1_post_dox.svg"))
    plt.gca()

    return (idx,)


@app.cell
def _(mu_conf, mu_star):
    print(f"RMA deg mu_star: {mu_star[192, 2]} +- {mu_conf[192, 2]}")
    print(f"tTA-TetO Kd: {mu_star[192, 13]} +- {mu_conf[192, 13]}")
    return


@app.cell
def _(data_dir, idx, jnp, labels, mu_conf, mu_star, os, plt, sb):
    # normalize mean
    #idx = jnp.linspace(0, 15, 15, dtype=int)
    max_mu_star = jnp.max(mu_star[192, :])
    _fig, _ax = plt.subplots()
    _ax.bar(
        labels, 
        [mu_star[192,i] / max_mu_star for i in idx],
        yerr=[mu_conf[192,j] / max_mu_star for j in idx],
        color='lightgrey',
    )

    plt.title("Day 1 post dox withdrawal")
    plt.ylabel("Relative Ranking")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_day_1_post_dox.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, idx, labels, mu_conf, mu_star, os, plt, sb):
    _fig, _ax = plt.subplots()
    _ax.bar(
        labels, 
        [mu_star[-1,i] for i in idx],
        yerr=[mu_conf[-1,j] for j in idx],
        color='lightgrey'
    )

    plt.title("Day 21 post dox withdrawal")
    plt.ylabel("Mean Morris Sensitivty, $µ^*$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean_day_21_post_dox.svg"))
    plt.gca()
    return


@app.cell
def _(mu_conf, mu_star):
    print(f"RMA deg mu_star: {mu_star[-1, 2]} +- {mu_conf[-1, 2]}")
    print(f"tTA-TetO Kd: {mu_star[-1, 13]} +- {mu_conf[-1, 13]}")

    print(f"Dox Kd mu_star: {mu_star[-1, 10]} +- {mu_conf[-1, 10]}")
    print(f"tTA prod mu_star: {mu_star[-1, 11]} +- {mu_conf[-1, 11]}")
    print(f"tTA deg mu_star: {mu_star[-1, 12]} +- {mu_conf[-1, 12]}")
    return


@app.cell
def _(data_dir, idx, jnp, labels, mu_conf, mu_star, os, plt, sb):
    # normalized day 28 (3 week time point after dox removal)
    _fig, _ax = plt.subplots()
    _max_mu_star = jnp.max(mu_star[-1, :])
    _ax.bar(
        labels, 
        [mu_star[-1,i] / _max_mu_star for i in idx],
        yerr=[mu_conf[-1,j] / _max_mu_star for j in idx],
        color='lightgrey',
    )

    plt.title("Day 21 post dox withdrawal")
    plt.ylabel("Relative Ranking")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_day_21_post_dox.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, idx, labels, os, plt, sb, sigma):
    # std
    _fig, _ax = plt.subplots()
    _ax.bar(
        labels, 
        [sigma[192,i] for i in idx],
        color='lightgrey'
    )

    plt.title("Day 1 post dox withdrawal")
    plt.ylabel("Std. Morris Sensitivty, $\\sigma$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std_day_1_post_dox.svg"))
    plt.gca()

    return


@app.cell
def _(sigma):
    print(f"RMA deg sigma: {sigma[192, 2]}")
    print(f"tTA-TetO Kd: {sigma[192, 13]}")
    return


@app.cell
def _(data_dir, idx, jnp, labels, os, plt, sb, sigma):
    # normalized std on day 1
    _fig, _ax = plt.subplots()
    _max_sigma = jnp.max(sigma[192, :])
    _ax.bar(
        labels, 
        [sigma[192,i] / _max_sigma for i in idx],
        color='lightgrey'
    )

    plt.title("Day 1 post dox withdrawal")
    plt.ylabel("Relative Nonlinearity or Interaction")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_day_1_post_dox.svg"))
    plt.gca()

    return


@app.cell
def _(data_dir, idx, labels, os, plt, sb, sigma):
    _fig, _ax = plt.subplots()
    _ax.bar(
        labels, 
        [sigma[-1,i] for i in idx],
        color='lightgrey'
    )

    plt.title("Day 21 post dox withdrawal")
    plt.ylabel("Std. Morris Sensitivty, $\\sigma$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std_day_21_post_dox.svg"))
    plt.gca()
    return


@app.cell
def _(sigma):
    print(f"RMA deg sigma: {sigma[-1, 2]}")
    print(f"tTA-TetO Kd: {sigma[-1, 13]}")

    print(f"Dox Kd sigma: {sigma[-1, 10]}")
    print(f"tTA prod sigma: {sigma[-1, 11]}")
    print(f"tTA deg sigma: {sigma[-1, 12]}")
    return


@app.cell
def _(data_dir, idx, jnp, labels, os, plt, sb, sigma):
    # normalized std on day 21
    _fig, _ax = plt.subplots()
    _max_sigma = jnp.max(sigma[-1, :])
    _ax.bar(
        labels, 
        [sigma[-1,i] / _max_sigma for i in idx],
        color='lightgrey'
    )

    plt.title("Day 21 post dox withdrawal")
    plt.ylabel("Relative Nonlinearity or Interaction")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_day_21_post_dox.svg"))
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
