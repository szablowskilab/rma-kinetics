import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import ChemogeneticRMA, DoxPKConfig, CnoPKConfig
    from gluc import get_gluc_conc
    from diffrax import PIDController, Kvaerno5
    from jax import config, numpy as jnp
    from diffopt.multiswarm import ParticleSwarm, get_best_loss_and_params
    from sklearn.metrics import r2_score

    import matplotlib.pyplot as plt
    import seaborn as sb
    import os
    import marimo as mo
    import polars as pl

    config.update("jax_enable_x64", True)
    sb.set_theme("talk", style="ticks", font="Arial")
    data_dir = os.path.join("notebooks", "data", "dreadd_activation")
    return (
        ChemogeneticRMA,
        CnoPKConfig,
        DoxPKConfig,
        Kvaerno5,
        PIDController,
        ParticleSwarm,
        data_dir,
        get_best_loss_and_params,
        get_gluc_conc,
        jnp,
        mo,
        os,
        pl,
        plt,
        r2_score,
        sb,
    )


@app.cell
def _(mo):
    cno_dose = mo.ui.radio(options=["1", "2.5"], value="1", label="CNO Dose (mg/kg)")
    cno_dose
    return (cno_dose,)


@app.cell
def _(cno_dose, data_dir, get_gluc_conc, os, pl):
    # load data
    full_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
    df = full_df.filter(pl.col("cno_dose") == float(cno_dose.value))
    gluc_df = get_gluc_conc(df, "rma")
    return full_df, gluc_df


@app.cell
def _(cno_dose, gluc_df, plt, sb):
    # plot RMA response to CNO administration
    sb.barplot(gluc_df, x="time", y="gluc", errorbar="sd", color="lightgrey")

    plt.title(f"{cno_dose.value} mg/kg CNO")
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(gluc_df, pl):
    mean_gluc = gluc_df.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc"),
        pl.col("gluc").std().alias("std_gluc")
    ])
    return (mean_gluc,)


@app.cell
def _(CnoPKConfig, DoxPKConfig, cno_dose, jnp, os):
    # load CNO and Dox model parameters
    mouse_weight = 0.03 # kg
    cno_model_params = jnp.load(os.path.join("notebooks", "data", "cno_pk", "params_estimate.npy"))
    cno_model_config = CnoPKConfig(float(cno_dose.value) * mouse_weight, *cno_model_params)

    dox_hyclate_percent = 0.87
    dox_model_config = DoxPKConfig(
        vehicle_intake_rate=1.875e-4, # mg food / hr - 4.5 mg / day
        bioavailability=0.90,
        dose=40 * dox_hyclate_percent, # mg / kg food
        absorption_rate=0.8, # 1/hr
        elimination_rate=0.2, # 1 / hr
        brain_transport_rate=0.2, # 1/ hr
        plasma_transport_rate=1, # 1 / hr
        t0=0,
        t1=0,
        plasma_vd=0.7*mouse_weight # L
    )

    return cno_model_config, cno_model_params, dox_model_config, mouse_weight


@app.cell
def _(
    ChemogeneticRMA,
    Kvaerno5,
    PIDController,
    cno_model_config,
    dox_model_config,
    jnp,
    mean_gluc,
):
    observed = mean_gluc.sort("time")["mean_gluc"].to_jax()
    std = mean_gluc.sort("time")["std_gluc"].to_jax()
    t0 = 0
    t1 = 96

    def loss(params):
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

        solution = rma_model.simulate(
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=(0, 0, 0, dox_model_config.brain_dox_ss, dox_model_config.plasma_dox_ss, params[9], 0, 0, 0, 0, 0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            solver=Kvaerno5(),
            sampling_rate=1
        )

        predicted = jnp.array([solution.ys[1][47], solution.ys[1][71], solution.ys[1][95]])
        return jnp.mean((observed - predicted)**2/observed**2)
    return loss, observed, t0, t1


@app.cell
def _(
    ParticleSwarm,
    cno_dose,
    data_dir,
    get_best_loss_and_params,
    jnp,
    loss,
    os,
):
    bounds = [
        (7e-3, 1), # RMA production rate
        (0.54, 1), # RMA RT rate
        (4e-3, 1e-2), # RMA deg rate
        (1, 10), # dox Kd
        (5, 15), # tTA prod rate
        (1e-2, 1e-1), # tTA deg rate
        (1, 10), #tTA Kd
        (5, 10), # CNO EC50
        (1, 5), # CLZ EC50
        (1, 10), # hM3Dq steady state
        (1, 10), # hM3Dq EC50
        (1e-5, 1e-2), # leaky RMA prod rate
        (5e-2, 1.5e-1) # leaky tTA prod rate
    ]

    swarm = ParticleSwarm(
        nparticles=300,
        ndim=13,
        xlow=[b[0] for b in bounds],
        xhigh=[b[1] for b in bounds],
        cognitive_weight=0.3,
        inertial_weight=0.1
    )

    pso_result = swarm.run_pso(loss)

    best_loss, best_params = get_best_loss_and_params(
        pso_result["swarm_loss_history"],
        pso_result["swarm_x_history"]
    )

    jnp.save(os.path.join(data_dir, f"cno_{cno_dose.value}_params.npy"), best_params)
    return (best_params,)


@app.cell
def _(
    ChemogeneticRMA,
    Kvaerno5,
    PIDController,
    best_params,
    cno_model_config,
    dox_model_config,
    gluc_df,
    jnp,
    observed,
    plt,
    r2_score,
    sb,
    t0,
    t1,
):
    def inspect_fit(params, cno_model_config):

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

        solution = rma_model.simulate(
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=(0, 0, 0, dox_model_config.brain_dox_ss, dox_model_config.plasma_dox_ss, params[9], 0, 0, 0, 0, 0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            solver=Kvaerno5(),
            sampling_rate=1
        )

        timepoints = [48, 72, 96] # first 48 hours are where dox is withdrawn (CNO is adminstered at T=48 hrs)
        predicted = jnp.array([solution.ys[1][tp] for tp in timepoints])

        r2 = r2_score(observed, predicted)
        print(f"R2: {r2:.3f}")

        return solution

    solution = inspect_fit(best_params, cno_model_config)

    # visualize fit
    plt.plot(jnp.linspace(0, 47, 47), solution.ys[1][48:], 'k')
    plt.errorbar(gluc_df["time"], gluc_df["gluc"], color='k', fmt='o')
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.tight_layout()
    plt.gca()
    return (solution,)


@app.cell
def _(plt, sb, solution):
    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True)

    ax[0,0].plot(solution.ts, solution.cno)
    ax[0,0].set_ylabel("Brain CNO (nM)")

    ax[0,1].plot(solution.ts, solution.clz)
    ax[0,1].set_ylabel("Brain CLZ (nM)")

    ax[1,0].plot(solution.ts, solution.tta)
    ax[1,0].set_ylabel("Brain tTA (nM)")

    ax[1,1].plot(solution.ts, solution.dox)
    ax[1,1].set_ylabel("Brain Dox (nM)")

    sb.despine()
    plt.tight_layout()
    plt.gca()
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
def _(plt, solution):
    plt.plot(solution.ts, solution.ys[5])
    return


@app.cell
def _(
    ChemogeneticRMA,
    CnoPKConfig,
    Kvaerno5,
    PIDController,
    best_params,
    cno_model_params,
    dox_model_config,
    full_df,
    get_gluc_conc,
    jnp,
    mouse_weight,
    pl,
    r2_score,
    solution,
    t0,
    t1,
):
    # simulate RMA at 2.5mg/kg CNO
    cno_model_config_high_dose = CnoPKConfig(2.5 * mouse_weight, *cno_model_params)
    high_dose_rma_model = ChemogeneticRMA(
        rma_prod_rate=best_params[0],
        rma_rt_rate=best_params[1],
        rma_deg_rate=best_params[2],
        dox_model_config=dox_model_config,
        dox_kd=best_params[3],
        tta_prod_rate=best_params[4],
        tta_deg_rate=best_params[5],
        tta_kd=best_params[6],
        cno_model_config=cno_model_config_high_dose,
        cno_t0=48.0,
        cno_ec50=best_params[7],
        clz_ec50=best_params[8],
        dq_prod_rate=best_params[9],
        dq_deg_rate=1,
        dq_ec50=best_params[10],
        leaky_rma_prod_rate=best_params[11],
        leaky_tta_prod_rate=best_params[12]
    )

    high_dose_solution = high_dose_rma_model.simulate(
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=(0, 0, 0, dox_model_config.brain_dox_ss, dox_model_config.plasma_dox_ss, best_params[9], 0, 0, 0, 0, 0),
        stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
        solver=Kvaerno5(),
        sampling_rate=1
    )

    timepoints = [48, 72, 96]
    high_dose_predicted = jnp.array([solution.ys[1][tp] for tp in timepoints])
    high_dose_df = full_df.filter(pl.col("cno_dose") == 2.5)
    high_dose_gluc_df = get_gluc_conc(high_dose_df, "rma")
    high_dose_mean_gluc = high_dose_gluc_df.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc"),
        pl.col("gluc").std().alias("std_gluc")
    ])
    high_dose_observed = high_dose_mean_gluc.sort("time")["mean_gluc"].to_jax()
    high_dose_std = high_dose_mean_gluc.sort("time")["std_gluc"].to_jax()

    high_dose_r2 = r2_score(high_dose_observed, high_dose_predicted)
    print(f"R2: {high_dose_r2:.3f}")
    return high_dose_gluc_df, high_dose_solution


@app.cell
def _(
    data_dir,
    gluc_df,
    high_dose_gluc_df,
    high_dose_solution,
    jnp,
    os,
    plt,
    sb,
    solution,
):
    # visualize fit
    colors = sb.color_palette(n_colors=2)
    plt.plot(jnp.linspace(0, 48, 48), solution.ys[1][47:], color=colors[0], label="1.0")
    plt.errorbar(gluc_df["time"], gluc_df["gluc"], fmt='o', alpha=0.5, color=colors[0])

    plt.plot(jnp.linspace(0, 48, 48), high_dose_solution.ys[1][47:], color=colors[1], label="2.5")
    plt.errorbar(high_dose_gluc_df["time"], high_dose_gluc_df["gluc"], fmt='s', alpha=0.5, color=colors[1])

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.tight_layout()
    plt.legend(frameon=False, title="CNO (mg/kg)")
    plt.savefig(os.path.join(data_dir, "chemogenetic_rma_fit_all.svg"))
    plt.gca()
    return (colors,)


@app.cell
def _(colors, data_dir, high_dose_solution, os, plt, sb, solution):
    # visualize brain CNO, CLZ, and tTA
    _fig, _ax = plt.subplots(1, 2, figsize=(6.4, 3.5))
    _ax[0].plot(solution.ts[:48], solution.clz[47:], color=colors[0], label="Brain CLZ (1mg/kg CNO)")
    _ax[0].plot(solution.ts[:48], high_dose_solution.clz[47:], color=colors[1], label="Brain CLZ (2.5mg/kg CNO)")

    #_ax[0].plot(solution.ts[:48], solution.cno[47:], color=colors[0], linestyle=":", label="Brain CNO (1mg/kg CNO)")
    #_ax[0].plot(solution.ts[:48], high_dose_solution.cno[47:], color=colors[1], linestyle=":", label="Brain CNO (1mg/kg CNO)")

    _ax[0].set_xlabel("Time (hr)")
    _ax[0].set_ylabel("CLZ (nM)")
    _ax[0].legend(["1.0", "2.5"], title="CNO (mg/kg)", frameon=False)

    _ax[1].plot(solution.ts[:48], solution.tta[47:], color=colors[0], label="Brain tTA (1mg/kg CNO)")
    _ax[1].plot(solution.ts[:48], high_dose_solution.tta[47:], color=colors[1], label="Brain tTA (2.5mg/kg CNO)")
    _ax[1].set_xlabel("Time (hr)")
    _ax[1].set_ylabel("tTA (nM)")

    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "cno_tta_prediction_all.svg"))
    plt.gca()
    return


@app.cell
def _(high_dose_solution, jnp, solution):
    print(f"2.5mg/kg CNO - max tTA: {jnp.max(high_dose_solution.tta)}")
    # subtract 48 to account for dox withdrawal period (Tmax from CNO administration)
    print(f"Tmax tTA 2.5mg/kg CNO: {solution.ts[jnp.argmax(high_dose_solution.tta)] - 48}")

    print(f"1mg/kg CNO - max tTA: {jnp.max(solution.tta)}")
    print(f"Tmax tTA 1mg/kg CNO: {solution.ts[jnp.argmax(solution.tta)] - 48}")
    return


@app.cell
def _(high_dose_solution, jnp, solution):
    print(f"Tmax CNO 1mg/kg CNO: {solution.ts[jnp.argmax(high_dose_solution.cno)] - 48}")
    print(f"Tmax CLZ 1mg/kg CNO: {solution.ts[jnp.argmax(high_dose_solution.clz)] - 48}")
    print(f"Tmax CNO 2.5mg/kg CNO: {solution.ts[jnp.argmax(solution.cno)] - 48}")
    print(f"Tmax CLZ 2.5mg/kg CNO: {solution.ts[jnp.argmax(solution.clz)] - 48}")


    return


@app.cell
def _(high_dose_solution, jnp, solution):
    print(f"1mg/kg CNO max CNO: {jnp.max(solution.cno)}")
    print(f"1mg/kg CNO max CLZ: {jnp.max(solution.clz)}")

    print(f"2.5mg/kg CNO max CNO: {jnp.max(high_dose_solution.cno)}")
    print(f"2.5mg/kg CNO max CLZ: {jnp.max(high_dose_solution.clz)}")

    return


@app.cell
def _(high_dose_solution, jnp, solution):
    print(f"Max RMA (1mg/kg CNO): {jnp.max(solution.plasma_rma)}")
    print(f"Max RMA (2.5mg/kg CNO): {jnp.max(high_dose_solution.plasma_rma)}")
    return


@app.cell
def _(
    ChemogeneticRMA,
    Kvaerno5,
    PIDController,
    cno_model_config,
    dox_model_config,
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
            cno_t0=48.2,
            cno_ec50=params[7],
            clz_ec50=params[8],
            dq_prod_rate=params[9],
            dq_deg_rate=1,
            dq_ec50=params[10],
            leaky_rma_prod_rate=params[11],
            leaky_tta_prod_rate=params[12]
        )

        solution = rma_model.simulate(
            t0=0,
            t1=96,
            dt0=0.1,
            #y0=(0, 0, 0, brain_dox_ss, plasma_dox_ss, 10, 0, 0, 0, 0, 0),
            y0=(0, 0, 0, 0, 0, params[9], 0, 0, 0, 0, 0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            sampling_rate=1,
            throw=True,
            solver=Kvaerno5()
        )

        return solution.ys[1]
    return global_sensitivity, map_model


@app.cell
def _(data_dir, global_sensitivity, jnp, map_model, os):
    range = jnp.array([-0.5, 0.5])
    sa_params = jnp.load(os.path.join(data_dir, "cno_1_params.npy"))

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
            "leaky_tta_prod_rate",
        ],
        "bounds": [p * (1 + range) for p in sa_params],
        "outputs": "Y"
    }

    morris_y, morris_sens = global_sensitivity(map_model, param_space, 250)
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
def _(data_dir, idx, jnp, mu_conf, mu_star, os, plt, sa_labels, sb):
     # normalize mean
    _fig, _ax = plt.subplots()
    _max_mu_star = jnp.max(mu_star[-1, :])
    _ax.bar(
        sa_labels, 
        [mu_star[-1,i] / _max_mu_star for i in idx],
        yerr=[mu_conf[-1,j] / _max_mu_star for j in idx],
        color='lightgrey'
    )

    plt.ylabel("Relative Ranking")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    _fig.set_figheight(5.2)
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_tf.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, jnp, mu_conf, mu_star, os, plt, sa_labels, sb):
    # morris mean
    top_params = mu_star[-1,:].argsort()[-5:][::-1]
    time = jnp.linspace(0, 94, 94)
    _fig, _ax = plt.subplots()

    for _i in top_params:
        _ax.plot(
            time,
            mu_star[:,_i],
            label=sa_labels[_i]
        )

        _ax.fill_between(
            time,
            mu_star[:,_i] - mu_conf[:,_i],
            mu_star[:,_i] + mu_conf[:,_i],
            alpha=0.25
        )


    plt.ylabel("Mean Morris Sensitivty, $µ^*$")
    plt.xlabel("Time (hr)")
    plt.legend(frameon=False)


    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_mean.svg"))
    plt.gca()
    return time, top_params


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


@app.cell
def _(data_dir, idx, jnp, os, plt, sa_labels, sb, sigma):
    # normalize std
    _fig, _ax = plt.subplots()
    _max_sigma = jnp.max(sigma[-1, :])
    _ax.bar(
        sa_labels, 
        [sigma[-1,i] / _max_sigma for i in idx],
        color='lightgrey'
    )

    plt.ylabel("Relative Nonlinearity or Interaction")
    for _label in _ax.get_xticklabels():
        _label.set_rotation(75)

    _fig.set_figheight(5.2)
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_tf.svg"))
    plt.gca()
    return


@app.cell
def _(data_dir, os, plt, sa_labels, sb, sigma, time, top_params):
    _fig, _ax = plt.subplots()
    for _i in top_params:
        _ax.plot(
            time,
            sigma[:,_i],
            label=sa_labels[_i]
        )

    plt.ylabel("Std. Morris Sensitivty, $\sigma$")
    plt.xlabel("Time (hr)")
    #plt.legend(frameon=False)


    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "morris_std.svg"))
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
