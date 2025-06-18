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
    from sklearn.metrics import r2_score

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
        SaveAt,
        data_dir,
        get_gluc_conc,
        jnp,
        os,
        pl,
        plt,
        r2_score,
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
def _(CnoPKConfig, jnp, mouse_weight, os):
    cno_model_params = jnp.load(os.path.join("notebooks/", "data", "cno_pk", "params_estimate.npy"))
    cno_model_config = CnoPKConfig(2.5 * mouse_weight, *cno_model_params)
    return (cno_model_config,)


@app.cell
def _(data_dir, get_gluc_conc, os, pl):
    full_df = pl.read_csv(os.path.join(data_dir, "source.csv"))
    df = full_df.filter(pl.col("cno_dose") == 2.5)
    gluc_df = get_gluc_conc(df)
    return full_df, gluc_df


@app.cell
def _(data_dir, jnp, os):
    params = jnp.load(os.path.join(data_dir, "cno_1_param_estimates.npy"))
    return (params,)


@app.cell
def _(
    ChemogeneticRMA,
    Kvaerno5,
    PIDController,
    SaveAt,
    brain_dox_ss,
    cno_model_config,
    dox_model_config,
    gluc_df,
    jnp,
    params,
    pl,
    plasma_dox_ss,
    r2_score,
):
    mean_gluc = gluc_df.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc"),
        pl.col("gluc").std().alias("std_gluc")
    ])

    observed = mean_gluc.sort("time")["mean_gluc"].to_jax()
    std = mean_gluc.sort("time")["std_gluc"].to_jax()

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


    return (solution,)


@app.cell
def _(gluc_df, plt, sb, solution):

    #jnp.save(os.path.join(data_dir, f"cno_{cno_dose.value}_solution.npy"), solution.ys)

    plt.plot(solution.ts, solution.ys[1], 'k')
    plt.errorbar(gluc_df["time"], gluc_df["gluc"], color='k', fmt='o')
    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.tight_layout()

    #plt.savefig(os.path.join(data_dir, f"cno_{cno_dose.value}_fit.svg"))
    plt.gca()

    return


@app.cell
def _(data_dir, jnp, os):
    # load 1mg/kg CNO fit
    cno_1mg_fit = jnp.load(os.path.join(data_dir, "cno_1_solution.npy"))
    cno_2mg_fit = jnp.load(os.path.join(data_dir, "cno_2.5_solution.npy"))
    return (cno_1mg_fit,)


@app.cell
def _(
    cno_1mg_fit,
    data_dir,
    full_df,
    get_gluc_conc,
    os,
    pl,
    plt,
    sb,
    solution,
):
    _colors = sb.color_palette(n_colors=3)
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

    plt.plot(solution.ts, cno_1mg_fit[1], color=_colors[0], label="1.0")
    plt.errorbar(cno_1mg_gluc["time"], cno_1mg_gluc["gluc"], fmt="o", color=_colors[0], alpha=0.25)

    # 2.5 mg/kg CNO
    plt.plot(solution.ts, solution.ys[1], color=_colors[1], label="2.5")
    plt.errorbar(cno_2mg_gluc["time"], cno_2mg_gluc["gluc"], fmt="s", color=_colors[1], alpha=0.25)

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.legend(frameon=False, title="CNO (mg/kg)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "chemogenetic_rma_fit_all.svg"))
    plt.gca()

    return


@app.cell
def _(cno_1mg_fit, data_dir, os, plt, sb, solution):
    _colors = sb.color_palette(n_colors=3)
    _fig, _ax = plt.subplots(1, 2, figsize=(6.4, 3.5))
    _ax[0].plot(solution.ts, cno_1mg_fit[9], color=_colors[0], label="Brain CLZ (1mg/kg CNO)")
    _ax[0].plot(solution.ts, solution.ys[9], color=_colors[1], label="Brain CLZ (2.5mg/kg CNO")

    _ax[0].plot(solution.ts, cno_1mg_fit[7], linestyle=":", color=_colors[0], label="Brain CNO (1mg/kg CNO)")
    _ax[0].plot(solution.ts, solution.ys[7], linestyle=":", color=_colors[1], label="Brain CNO (2.5mg/kg CNO)")
    _ax[0].set_xlabel("Time (hr)")
    _ax[0].set_ylabel("CNO or CLZ (nM)")
    _ax[0].legend(["CNO", "CLZ"], frameon=False)

    _ax[1].plot(solution.ts, cno_1mg_fit[2])
    _ax[1].plot(solution.ts, solution.ys[2])
    _ax[1].set_xlabel("Time (hr)")
    _ax[1].set_ylabel("Brain tTA (nM)")
    sb.despine()
    plt.tight_layout()

    plt.savefig(os.path.join(data_dir, "cno_tta_prediction_all.svg"))
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
