import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import ChemogeneticRMA, DoxPKConfig, CnoPKConfig
    from gluc import get_gluc_conc
    from diffrax import PIDController, SaveAt, Kvaerno5
    from jax import config, numpy as jnp
    from diffopt.multiswarm import ParticleSwarm, get_best_loss_and_params
    from functools import partial

    import matplotlib.pyplot as plt
    import seaborn as sb
    import os
    import marimo as mo
    import polars as pl

    config.update("jax_enable_x64", True)
    sb.set_context('talk')
    data_dir = os.path.join("notebooks", "data", "dreadd_activation")
    return (
        ChemogeneticRMA,
        CnoPKConfig,
        DoxPKConfig,
        Kvaerno5,
        PIDController,
        ParticleSwarm,
        SaveAt,
        data_dir,
        get_best_loss_and_params,
        get_gluc_conc,
        jnp,
        mo,
        os,
        partial,
        pl,
        plt,
        sb,
    )


@app.cell
def _(DoxPKConfig):
    mouse_weight = 0.03 # kg
    dox_hyclate_percent = 0.87
    dox_model_config = DoxPKConfig(
        vehicle_intake_rate=1.875e-4, # mg food / hr - 4.5 mg / day
        bioavailability=0.90,
        vehicle_dose=40 * dox_hyclate_percent, # mg / kg food
        absorption_rate=0.8, # 1/hr
        elimination_rate=0.2, # 1 / hr
        brain_transport_rate=0.2, # 1/ hr
        plasma_transport_rate=1, # 1 / hr
        t0=0,
        t1=0,
        plasma_vd=0.7*mouse_weight # L
    )
    plasma_dox_ss = dox_model_config.absorption_rate/dox_model_config.elimination_rate*dox_model_config.intake_rate
    brain_dox_ss = dox_model_config.brain_transport_rate/dox_model_config.plasma_transport_rate*plasma_dox_ss
    return brain_dox_ss, dox_model_config, mouse_weight, plasma_dox_ss


@app.cell
def _(mo):
    cno_dose = mo.ui.radio(options=["1", "2.5", "5"], value="1", label="CNO Dose (mg/kg)")
    cno_dose
    return (cno_dose,)


@app.cell
def _(CnoPKConfig, cno_dose, jnp, mouse_weight, os):
    # load CNO model params
    cno_model_params = jnp.load(os.path.join("notebooks/", "data", "cno_pk", "params_estimate.npy"))
    cno_model_config = CnoPKConfig(float(cno_dose.value) * mouse_weight, *cno_model_params)
    return cno_model_config, cno_model_params


@app.cell
def _(cno_model_params):
    print(cno_model_params)
    return


@app.cell
def _(cno_model_config):
    print(cno_model_config.cno_nmol)
    return


@app.cell
def _(cno_dose, data_dir, get_gluc_conc, os, pl):
    full_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
    df = full_df.filter(pl.col("cno_dose") == float(cno_dose.value))
    gluc_df = get_gluc_conc(df)
    return full_df, gluc_df


@app.cell
def _(gluc_df):
    gluc_df
    return


@app.cell
def _(cno_dose, gluc_df, plt, sb):
    sb.barplot(gluc_df, x="time", y="gluc", errorbar="sd", color="lightgrey")

    plt.title(f"{cno_dose.value} mg/kg CNO")
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.gca()
    return


@app.cell
def _(
    ChemogeneticRMA,
    Kvaerno5,
    PIDController,
    SaveAt,
    cno_model_config,
    dox_model_config,
    jnp,
):
    def loss(params, args):
        observed, std = args
        t0 = 0
        t1 = 48
        t2 = 96

        rma_model = ChemogeneticRMA(
            rma_prod_rate=params[0],
            rma_rt_rate=params[1],
            rma_deg_rate=params[2],
            dox_model_config=dox_model_config,
            dox_kd=params[3],
            tta_prod_rate=params[4],
            tta_deg_rate=params[5],
            tta_kd=params[6],
            cno_model_config=cno_model_config,
            cno_t0=48.0,
            cno_ec50=params[7],
            clz_ec50=params[8],
            dq_prod_rate=params[9],
            dq_deg_rate=1,
            dq_ec50=params[10],
            leaky_rma_prod_rate=params[11],
            leaky_tta_prod_rate=params[12]
        )

        pid_controller = PIDController(atol=1e-6, rtol=1e-6)

        #stepsize_controller = ClipStepSizeController(pid_controller, step_ts=jnp.array([0.]), store_rejected_steps=100)

        """
        chunk_1 = rma_model.simulate(
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, params[9], 0, 0, 0, 0, 0),
            #stepsize_controller=PIDController(atol=1e-3, rtol=1e-3),
            stepsize_controller=pid_controller,
            saveat=SaveAt(t1=True),
            throw=False,
            solver=Kvaerno5()
        )

        y0 = list(chunk_1.ys)
        y0[6] += cno_model_config.cno_nmol
        #y0[3] = 0
        #y0[4] = 0
        """

        chunk_2 = rma_model.simulate(
            t0=0,
            t1=48,
            dt0=0.1,
            #y0=tuple(y0),
            y0=(0, 0, 0, 0, 0, params[9], cno_model_config.cno_nmol, 0, 0, 0, 0),
            #stepsize_controller=PIDController(atol=1e-3, rtol=1e-3),
            stepsize_controller=pid_controller,
            saveat=SaveAt(ts=jnp.array([0, 24, 48])),
            throw=False,
            solver=Kvaerno5()
        )

        """
        mse = cond(
            solution.result != RESULTS.successful,
            lambda: 1e8,
            lambda: jnp.mean((observed[1:] - solution.ys[1][1:])**2)
        )
        """

        #return mse
        #if solution.result != RESULTS.successful:
           #return 1e8
        #else:
        return jnp.mean((observed - chunk_2.ys[1])**2/std**2)
    return (loss,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button()
    return (run_button,)


@app.cell
def _(gluc_df, pl):
    mean_gluc = gluc_df.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc"),
        pl.col("gluc").std().alias("std_gluc")
    ])
    return (mean_gluc,)


@app.cell
def _(
    ParticleSwarm,
    get_best_loss_and_params,
    loss,
    mean_gluc,
    mo,
    partial,
    run_button,
):
    mo.stop(not run_button.value)
    bounds = [
        (1e-3, 0.1), # RMA production rate
        (0.54, 1), # RMA RT rate
        (4e-3, 1e-2), # RMA deg rate
        (1, 10), # dox Kd
        (5, 10), # tTA prod rate
        (0.1, 1), # tTA deg rate
        (1, 10), #tTA Kd
        (5, 10), # CNO EC50
        (1, 5), # CLZ EC50
        (1e-4, 10), # hM3Dq steady state
        (1, 10), # hM3Dq EC50
        (1e-5, 1e-3), # leaky RMA prod rate
        (5e-5, 5e-3) # leaky tTA prod rate
    ]


    observed = mean_gluc.sort("time")["mean_gluc"].to_jax()
    std = mean_gluc.sort("time")["std_gluc"].to_jax()

    loss_fn = partial(loss, args=(observed, std))
    swarm = ParticleSwarm(
        nparticles=100,
        ndim=13,
        xlow=[b[0] for b in bounds],
        xhigh=[b[1] for b in bounds],
        cognitive_weight=0.3,
        social_weight=0.1
    )
    pso_result = swarm.run_pso(loss_fn)

    best_loss, best_params = get_best_loss_and_params(
        pso_result["swarm_loss_history"],
        pso_result["swarm_x_history"]
    )
    return best_params, observed


@app.cell
def _(ChemogeneticRMA, best_params, cno_model_config, dox_model_config):
    rma_model = ChemogeneticRMA(
        rma_prod_rate=best_params[0],
        rma_rt_rate=best_params[1],
        rma_deg_rate=best_params[2],
        dox_model_config=dox_model_config,
        dox_kd=best_params[3],
        tta_prod_rate=best_params[4],
        tta_deg_rate=best_params[5],
        tta_kd=best_params[6],
        cno_model_config=cno_model_config,
        cno_t0=48.0,
        cno_ec50=best_params[7],
        clz_ec50=best_params[8],
        dq_prod_rate=best_params[9],
        dq_deg_rate=1,
        dq_ec50=best_params[10],
        leaky_rma_prod_rate=best_params[11],
        leaky_tta_prod_rate=best_params[12]
    )
    return (rma_model,)


@app.cell
def _(
    Kvaerno5,
    PIDController,
    SaveAt,
    brain_dox_ss,
    cno_model_config,
    jnp,
    observed,
    plasma_dox_ss,
    rma_model,
):
    #plasma_dox_ss = dox_model_config.absorption_rate / dox_model_config.elimination_rate * dox_model_config.intake_rate
    #brain_dox_ss = dox_model_config.brain_transport_rate / dox_model_config.plasma_transport_rate * plasma_dox_ss
    from sklearn.metrics import r2_score

    def inspect_fit(params):

        dox_withdrawal = rma_model.simulate(
            t0=0,
            t1=48,
            dt0=0.1,
            #y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, 10, 0, 0, 0, 0, 0),
            y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, params[9], 0, 0, 0, 0, 0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            saveat=SaveAt(t1=True),
            throw=True,
            solver=Kvaerno5()
        )


        y0 = list(dox_withdrawal.ys)
        y0[6] += cno_model_config.cno_nmol

        solution = rma_model.simulate(
            t0=0,
            t1=48,
            dt0=0.1,
            #y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, 10, 0, 0, 0, 0, 0),
            y0=tuple(y0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            saveat=SaveAt(ts=jnp.linspace(0, 48, 48*6)),
            throw=False,
            solver=Kvaerno5()
        )

        timepoints = [0, 24, 48]
        predicted = jnp.array([solution.ys[1][t*6] for t in timepoints])
        mse = jnp.mean((observed - predicted)**2)
        r2 = r2_score(observed, predicted)

        print(f"MSE: {mse}")
        print(f"R2: {r2}")

        return solution

        """
        fig, ax = plt.subplots(3, 2, figsize=(12,8))
        sb.despine()

        timepoints = [0, 24, 48]
        predicted = jnp.array([solution.ys[1][t*6] for t in timepoints])
        mse = jnp.mean((observed - predicted)**2)
        r2 = r2_score(observed, predicted)

        print(f"MSE: {mse}")
        print(f"R2: {r2}")

        ax[0,0].plot(solution.ts, solution.ys[1], 'k')
        ax[0,0].errorbar(gluc_df["time"], gluc_df["gluc"], color='k', fmt='o')
        ax[0,0].set_xlabel("Time (hr)")
        ax[0,0].set_ylabel("Plasma RMA (nM)")

        ax[0,1].plot(solution.ts, solution.ys[2], 'k')
        ax[0,1].set_xlabel("Time (hr)")
        ax[0,1].set_ylabel("tTA (nM)")

        ax[1,0].plot(solution.ts, solution.ys[7] / rma_model.cno.cno_brain_vd, 'k')
        ax[1,0].set_xlabel("Time (hr)")
        ax[1,0].set_ylabel("Brain CNO (nM)")

        ax[1,1].plot(solution.ts, solution.ys[9] / rma_model.cno.clz_brain_vd, 'k')
        ax[1,1].set_xlabel("Time (hr)")
        ax[1,1].set_ylabel("Brain CLZ (nM)")

        ax[2,0].plot(solution.ts, solution.ys[3], 'k')
        ax[2,0].set_xlabel("Time (hr)")
        ax[2,0].set_ylabel("Brain Dox (nM)")

        ax[2,1].plot(solution.ts, solution.ys[4], 'k')
        ax[2,1].set_xlabel("Time (hr)")
        ax[2,1].set_ylabel("Plasma Dox (nM)")

        plt.tight_layout()
        """
    return (inspect_fit,)


@app.cell
def _(best_params, cno_dose, data_dir, gluc_df, inspect_fit, jnp, os, plt, sb):
    solution = inspect_fit(best_params)
    jnp.save(os.path.join(data_dir, f"cno_{cno_dose.value}_solution.npy"), solution.ys)

    plt.plot(solution.ts, solution.ys[1], 'k')
    plt.errorbar(gluc_df["time"], gluc_df["gluc"], color='k', fmt='o')
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.tight_layout()

    plt.savefig(os.path.join(data_dir, f"cno_{cno_dose.value}_fit.svg"))
    plt.gca()
    return (solution,)


@app.cell
def _(cno_dose, data_dir, os, plt, rma_model, sb, solution):
    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True)

    ax[0,0].plot(solution.ts, solution.ys[7] / rma_model.cno.cno_brain_vd, 'k')
    ax[0,0].set_ylabel("Brain CNO (nM)")

    ax[0,1].plot(solution.ts, solution.ys[9] / rma_model.cno.clz_brain_vd, 'k')
    ax[0,1].set_ylabel("Brain CLZ (nM)")

    ax[1,0].plot(solution.ts, solution.ys[2], 'k')
    ax[1,0].set_xlabel("Time (hr)")
    ax[1,0].set_ylabel("tTA (nM)")

    ax[1,1].plot(solution.ts, solution.ys[3], 'k')
    ax[1,1].set_xlabel("Time (hr)")
    ax[1,1].set_ylabel("Brain Dox (nM)")

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"cno_{cno_dose.value}_fit_other_species.svg"))
    plt.gca()

    return


@app.cell
def _(best_params):
    best_params
    return


@app.cell
def _(best_params):
    labels = [
        "RMA production",
        "RMA RT",
        "RMA degradation",
        "Dox Kd",
        "tTA production",
        "tTA degradation",
        "tTA Kd",
        "CNO EC50",
        "CLZ EC50",
        "hM3Dq steady state",
        "hM3Dq EC50",
        "Leaky RMA production",
        "Leaky tTA production"
    ]

    for _i, _label in enumerate(labels):
        print(f"{_label}: {best_params[_i]}")
    return


@app.cell
def _(best_params, cno_dose, data_dir, jnp, os):
    jnp.save(os.path.join(data_dir, f"cno_{cno_dose.value}_param_estimates.npy"), best_params)
    return


@app.cell
def _(data_dir, jnp, os):
    # plot fits on single figure
    cno_1mg_kg_fit = jnp.load(os.path.join(data_dir, "cno_1_solution.npy"))
    cno_2mg_kg_fit = jnp.load(os.path.join(data_dir, "cno_2.5_solution.npy"))
    return cno_1mg_kg_fit, cno_2mg_kg_fit


@app.cell
def _(
    cno_1mg_kg_fit,
    cno_2mg_kg_fit,
    data_dir,
    full_df,
    get_gluc_conc,
    jnp,
    os,
    pl,
    plt,
    sb,
):
    _colors = sb.color_palette(n_colors=3)
    _ts = jnp.linspace(0, 48, 288)

    cno_1mg_df = full_df.filter(pl.col("cno_dose") == 1)
    cno_1mg_gluc = get_gluc_conc(cno_1mg_df)
    cno_1mg_mean_gluc = cno_1mg_gluc.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc"),
        pl.col("gluc").std().alias("std_gluc")
    ])

    cno_2mg_df = full_df.filter(pl.col("cno_dose") == 2.5)
    cno_2mg_gluc = get_gluc_conc(cno_2mg_df)
    cno_2mg_mean_gluc = cno_2mg_gluc.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc"),
        pl.col("gluc").std().alias("std_gluc")
    ])

    plt.plot(_ts, cno_1mg_kg_fit[1], color=_colors[0], label="1")
    plt.errorbar(cno_1mg_mean_gluc["time"], cno_1mg_mean_gluc["mean_gluc"], yerr=cno_1mg_mean_gluc["std_gluc"], fmt="o", color=_colors[0], alpha=0.25)

    _colors = sb.color_palette(n_colors=3)
    plt.plot(_ts, cno_2mg_kg_fit[1], color=_colors[1], label="2.5")
    plt.errorbar(cno_2mg_mean_gluc["time"], cno_2mg_mean_gluc["mean_gluc"], yerr=cno_2mg_mean_gluc["std_gluc"], fmt="s", color=_colors[1], alpha=0.25)

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.legend(frameon=False, title="CNO (mg/kg)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "chemogenetic_rma_fit_all.svg"))
    plt.gca()
    return


@app.cell
def _(cno_1mg_kg_fit, cno_2mg_kg_fit, jnp, plt, sb):
    _ts = jnp.linspace(0, 48, 288)
    _colors = sb.color_palette(n_colors=3)
    _fig, _ax = plt.subplots()
    _ax.plot(_ts, cno_1mg_kg_fit[9], color=_colors[0], label="Brain CLZ (1mg/kg CNO)")
    _ax.plot(_ts, cno_2mg_kg_fit[9], color=_colors[1], label="Brain CLZ (2.5mg/kg CNO")

    _ax.plot(_ts, cno_1mg_kg_fit[7], linestyle="--", color=_colors[0], label="Brain CNO (1mg/kg CNO)")
    _ax.plot(_ts, cno_2mg_kg_fit[7], linestyle="--", color=_colors[1], label="Brain CNO (2.5mg/kg CNO)")

    _ax.plot(_ts, cno_1mg_kg_fit[2], linestyle=":")
    _ax.plot(_ts, cno_2mg_kg_fit[2], linestyle=":")
    #_ax[0].legend(frameon=False)
    sb.despine()
    plt.tight_layout()

    plt.gca()
    return


@app.cell
def _(cno_1mg_kg_fit, cno_2mg_kg_fit, jnp, plt):
    _ts = jnp.linspace(0, 48, 288)
    plt.plot(_ts, cno_1mg_kg_fit[2])
    plt.plot(_ts, cno_2mg_kg_fit[2])
    return


@app.cell
def _(cno_1mg_kg_fit, cno_2mg_kg_fit, plt):
    plt.plot(cno_1mg_kg_fit[5])
    plt.plot(cno_2mg_kg_fit[5])
    return


@app.cell
def _(
    ChemogeneticRMA,
    Kvaerno5,
    PIDController,
    SaveAt,
    brain_dox_ss,
    cno_model_config,
    dox_model_config,
    jnp,
    plasma_dox_ss,
):
    # sensitivity analysis
    from sensitivity import global_sensitivity

    def map_model(params):
        rma_model = ChemogeneticRMA(
            rma_prod_rate=params[0],
            rma_rt_rate=params[1],
            rma_deg_rate=params[2],
            dox_model_config=dox_model_config,
            dox_kd=params[3],
            tta_prod_rate=params[4],
            tta_deg_rate=params[5],
            tta_kd=params[6],
            cno_model_config=cno_model_config,
            cno_t0=48.0,
            cno_ec50=params[7],
            clz_ec50=params[8],
            dq_prod_rate=params[9],
            dq_deg_rate=1,
            dq_ec50=params[10],
            leaky_rma_prod_rate=params[11],
            leaky_tta_prod_rate=params[12]
        )

        dox_withdrawal = rma_model.simulate(
            t0=0,
            t1=48,
            dt0=0.1,
            #y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, 10, 0, 0, 0, 0, 0),
            y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, params[9], 0, 0, 0, 0, 0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            saveat=SaveAt(t1=True),
            throw=True,
            solver=Kvaerno5()
        )

        y0 = list(dox_withdrawal.ys)
        y0[6] += cno_model_config.cno_nmol

        solution = rma_model.simulate(
            t0=0,
            t1=48,
            dt0=0.1,
            #y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, 10, 0, 0, 0, 0, 0),
            y0=tuple(y0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            saveat=SaveAt(ts=jnp.linspace(0, 48, 48)),
            throw=True,
            solver=Kvaerno5()
        )
        #breakpoint()

        return solution.ys[1]

    return global_sensitivity, map_model


@app.cell
def _(data_dir, global_sensitivity, jnp, map_model, os):
    range = jnp.array([-0.5, 0.5])
    sa_params = jnp.load(os.path.join(data_dir, "cno_1_param_estimates.npy"))
    print(len(sa_params))
    param_space = {
        "num_vars": 13,
        "names": [
            "rma_prod_rate",
            "rma_rt_rate",
            "rma_deg_rate",
            "dox_kd",
            "tta_prod_rate",
            "tta_deg_rate",
            "tta_kd",
            "cno_ec50",
            "clz_ec50",
            "dq_prod_rate",
            "dq_ec50",
            "leaky_rma_prod_rate",
            "leaky_tta_prod_rate"
        ],
        "bounds": [p * (1 + range) for p in sa_params],
        "outputs": "Y"
    }

    morris_y, morris_sens = global_sensitivity(map_model, param_space, 250)
    time = jnp.linspace(0, 48, 288)
    mu_star = jnp.array([s['mu_star'] for s in morris_sens])
    mu_conf = jnp.array([s['mu_star_conf'] for s in morris_sens])
    sigma = jnp.array([s['sigma'] for s in morris_sens])

    return mu_conf, mu_star, sigma


@app.cell
def _(data_dir, jnp, mu_conf, mu_star, os, plt, sb):
    idx = jnp.linspace(0, 13, 13, dtype=int)
    sa_labels = [
        "$k_{RMA}$",
        "$k_{RT}$",
        "$\\gamma_{RMA}$",
        "$K_{D_{Dox}}$",
        "$k_{tTA}$",
        "$\\gamma_{tTA}$",
        "$K_{D_{tTA}}$",
        "$EC_{50_{CNO}}$",
        "$EC_{50_{CLZ}}$",
        "$[Dq]_{ss}$",
        "$EC_{50_{Dq}}$",
        "$k_{0_{RMA}}$",
        "$k_{0_{tTA}}$",
    ]

    _fig, _ax = plt.subplots()
    _ax.bar(
        sa_labels, 
        [mu_star[-1,i] for i in idx],
        yerr=[mu_conf[-1,j] for j in idx],
        color='lightgrey'
    )

    plt.ylabel("Mean Morris Sensitivty, $µ^*$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean_tf.svg"))
    plt.gca()

    return idx, sa_labels


@app.cell
def _(data_dir, idx, os, plt, sa_labels, sb, sigma):
    _fig, _ax = plt.subplots()
    _ax.bar(
        sa_labels, 
        [sigma[-1,i] for i in idx],
        color='lightgrey'
    )

    plt.ylabel("Std. Morris Sensitivty, $\\sigma$")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std_tf.svg"))
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
