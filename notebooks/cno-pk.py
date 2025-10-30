import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import CnoPK, CnoPKConfig

    from diffrax import PIDController, SaveAt, diffeqsolve, Kvaerno3, ODETerm
    from jax import numpy as jnp
    from jax import config, vmap
    from functools import partial
    from diffopt.multiswarm import ParticleSwarm, get_best_loss_and_params
    from sklearn.metrics import r2_score

    import seaborn as sb
    import matplotlib.pyplot as plt
    import polars as pl
    import os

    config.update("jax_enable_x64", True)
    sb.set_context("talk")
    return (
        CnoPK,
        CnoPKConfig,
        Kvaerno3,
        ODETerm,
        PIDController,
        ParticleSwarm,
        SaveAt,
        diffeqsolve,
        get_best_loss_and_params,
        jnp,
        os,
        partial,
        pl,
        plt,
        r2_score,
        sb,
        vmap,
    )


@app.cell
def _(pl):
    plasma_cno = pl.read_csv("notebooks/data/cno_pk/plasma_cno.csv")
    brain_cno = pl.read_csv("notebooks/data/cno_pk/brain_cno.csv")
    plasma_clz = pl.read_csv("notebooks/data/cno_pk/plasma_clz.csv")
    brain_clz = pl.read_csv("notebooks/data/cno_pk/brain_clz.csv")
    data_agg = [plasma_cno, plasma_clz, brain_cno, brain_clz]
    return brain_clz, brain_cno, data_agg, plasma_clz, plasma_cno


@app.cell
def _(brain_clz, brain_cno, pl, plasma_clz, plasma_cno, plt, sb):
    def plot_concentration(df: pl.DataFrame, ax: 'np.ndarray', loc: str, drug: str) -> None:
        sb.boxplot(df, x="time(min)", y="concentration(nM)", color='lightgrey', ax=ax, showfliers=False)
        sb.stripplot(x="time(min)", y="concentration(nM)", data=df, ax=ax, size=8, color='k')
        ax.set(
            xlabel="Time (min)",
            ylabel=f"{loc} {drug} (nM)"
        )

    fig1, ax1 = plt.subplots(2, 2, figsize=(12,8))

    plot_concentration(plasma_cno, ax1[0,0], "Plasma", "CNO")
    plot_concentration(plasma_clz, ax1[0,1], "Plasma", "CLZ")
    plot_concentration(brain_cno, ax1[1,0], "Brain", "CNO")
    plot_concentration(brain_clz, ax1[1,1], "Brain", "CLZ")

    sb.despine()
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(data_agg, jnp, pl):
    def get_mean_concentrations(
        df: pl.DataFrame,
        x: str = "time(min)",
        y: str = "concentration(nM)"
    ) -> jnp.ndarray:
        """Get mean concentrations at each timepoint in the dataframe"""
        observed = df.group_by(x).agg([
            pl.col(y).mean().fill_null(1e-6)
        ])

        return observed.sort(x, descending=False)[y].to_jax()

    observed = jnp.array([get_mean_concentrations(df) for df in data_agg])
    return (observed,)


@app.cell
def _(observed):

    observed
    return


@app.cell
def _(
    CnoPK,
    CnoPKConfig,
    Kvaerno3,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
    jnp,
    plt,
    vmap,
):
    # sanity check

    _model = CnoPK(CnoPKConfig(
        3.5 * 0.03,
        0.38,
        9e-4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.25,
        0.25,
        0.25,
        0.25,
    ))

    solution = diffeqsolve(
        ODETerm(_model._model),
        solver=Kvaerno3(),
        t0=0,
        t1=60,
        dt0=0.1,
        y0=(3.5*0.03 / 342.8 * 1e6,0,0,0,0),
        saveat=SaveAt(dense=True),
        stepsize_controller=PIDController(atol=1e-5, rtol=1e-5)
    )

    time = jnp.linspace(0, 60, 60)
    result = vmap(solution.evaluate)(time)

    plt.plot(time, result[1])
    return


@app.cell
def _(
    CnoPK,
    CnoPKConfig,
    Kvaerno3,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
    jnp,
):
    cno_dose = 3.5 # mg/kg
    mouse_weight = 0.03 # kg

    def loss(params, args):
        observed, weights = args
        # working in minutes for now
        t0 = 0
        t1 = 60

        model = CnoPK(CnoPKConfig(
            cno_dose * mouse_weight,
            *params,
        ))

        predicted_nmol = diffeqsolve(
            ODETerm(model._model),
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=(cno_dose*mouse_weight / 342.8 * 1e6,0,0,0,0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            saveat=SaveAt(ts=jnp.array([15, 30, 60])),
            solver=Kvaerno3(),
        )

        predicted = jnp.array([
            predicted_nmol.ys[1] / model.cno_plasma_vd,
            predicted_nmol.ys[3] / model.clz_plasma_vd,
            predicted_nmol.ys[2] / model.cno_brain_vd,
            predicted_nmol.ys[4] / model.clz_brain_vd,
        ])


        #plasma_cno_res = jnp.sum((observed[0] - predicted[0])**2)
        plasma_cno_res = jnp.sum((observed[0] - predicted[0])**2) * weights[0]
        plasma_clz_res = jnp.sum((observed[1] - predicted[1])**2) * weights[1]
        brain_cno_res = jnp.sum((observed[2] - predicted[2])**2) * weights[2]
        brain_clz_res = jnp.sum((observed[3] - predicted[3])**2) * weights[3]

        wssr = plasma_cno_res + brain_cno_res + plasma_clz_res + brain_clz_res
        return wssr
    return cno_dose, loss, mouse_weight


@app.cell
def _(ParticleSwarm, jnp, loss, observed, partial):
    weights = jnp.array([1, 2.5, 1.25, 10])

    bounds = [
        (0.3, 0.5), # cno absorption
        (1e-5, 1e-3), # cno elimination
        (0.005, 0.5), # cno reverse metabolism
        (0.005, 0.5), # clz metabolism
        (0.03, 0.04), # cno brain transport
        (0.5, 2), # cno plasma transport
        (0.05, 0.6), # clz brain transport
        (0.5, 0.7), # clz plasma transport
        (1e-3, 0.1), # clz elimination
        (0.04, 0.25), # cno plasma Vd
        (0.20, 0.25), # cno brain Vd
        (0.04, 0.25), # clz plasma Vd
        (0.04, 0.25) # clz brain Vd
    ]

    loss_partial = partial(loss, args=(observed, weights))
    swarm = ParticleSwarm(
        nparticles=300,
        ndim=13,
        xlow=[b[0] for b in bounds],
        xhigh=[b[1] for b in bounds],
        cognitive_weight=0.3,
        social_weight=0.1
    )
    pso_result = swarm.run_pso(loss_partial)
    return (pso_result,)


@app.cell
def _(get_best_loss_and_params, pso_result):
    best_loss, best_params = get_best_loss_and_params(
        pso_result["swarm_loss_history"],
        pso_result["swarm_x_history"]
    )
    return (best_params,)


@app.cell
def _(best_params):
    labels = [
        "CNO absorption",
        "CNO elimination",
        "CNO reverse metabolism",
        "CLZ metabolism",
        "CNO brain transport",
        "CNO plasma transport",
        "CLZ brain transport",
        "CLZ plasma transport",
        "CLZ elimination",
        "CNO plasma Vd",
        "CNO brain Vd",
        "CLZ plasma Vd",
        "CLZ brain Vd"
    ]

    for _i, _label in enumerate(labels):
        print(f"{_label}: {best_params[_i]}")
    return


@app.cell
def _(
    CnoPK,
    CnoPKConfig,
    Kvaerno3,
    ODETerm,
    PIDController,
    SaveAt,
    brain_clz,
    brain_cno,
    cno_dose,
    diffeqsolve,
    jnp,
    mouse_weight,
    observed,
    plasma_clz,
    plasma_cno,
    plt,
    r2_score,
):
    # visual inspection
    def inspect_fit(params):
        t0 = 0
        t1 = 60
        fs = 2

        model = CnoPK(CnoPKConfig(
            cno_dose * mouse_weight,
            *params,
        ))

        predicted_nmol = diffeqsolve(
            ODETerm(model._model),
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=(cno_dose * mouse_weight / 342.8 * 1e6,0,0,0,0),
            stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
            saveat=SaveAt(ts=jnp.linspace(0, t1, t1*fs)),
            solver=Kvaerno3()
        )


        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        plasma_cno_conc = predicted_nmol.ys[1] / model.cno_plasma_vd
        brain_cno_conc = predicted_nmol.ys[2] / model.cno_brain_vd
        plasma_clz_conc = predicted_nmol.ys[3] / model.clz_plasma_vd
        brain_clz_conc = predicted_nmol.ys[4] / model.clz_brain_vd

        observed_timepoints = [15, 30, 60]

        predicted_plasma_cno = jnp.array([plasma_cno_conc[t*fs] for t in observed_timepoints])
        predicted_brain_cno = jnp.array([brain_cno_conc[t*fs] for t in observed_timepoints])
        predicted_plasma_clz = jnp.array([plasma_clz_conc[t*fs] for t in observed_timepoints])
        predicted_brain_clz = jnp.array([brain_clz_conc[t*fs] for t in observed_timepoints])

        print(predicted_plasma_cno)
        print(observed[0])

        plasma_cno_rmse = jnp.sqrt(jnp.sum((observed[0] - predicted_plasma_cno)**2) / len(observed[0]))
        plasma_clz_rmse = jnp.sqrt(jnp.sum((observed[1] - predicted_plasma_clz)**2) / len(observed[1]))
        brain_cno_rmse = jnp.sqrt(jnp.sum((observed[2][:-1] - predicted_brain_cno[:-1])**2) / len(observed[2][:-1]))
        brain_clz_rmse = jnp.sqrt(jnp.sum((observed[3] - predicted_brain_clz)**2) / len(observed[3]))

        print(f"Plasma CNO RMSE: {plasma_cno_rmse}")
        print(f"Brain CNO RMSE: {brain_cno_rmse}")
        print(f"Plasma CLZ RMSE: {plasma_clz_rmse}")
        print(f"Brain CLZ RMSE: {brain_clz_rmse}")

        plasma_cno_r2 = r2_score(observed[0], predicted_plasma_cno)
        plasma_clz_r2 = r2_score(observed[1], predicted_plasma_clz)
        brain_cno_r2 = r2_score(observed[2], predicted_brain_cno)
        brain_clz_r2 = r2_score(observed[3], predicted_brain_clz)

        print("-------------------------------")
        print(f"Plasma CNO R2: {plasma_cno_r2}")
        print(f"Brain CNO R2: {brain_cno_r2}")
        print(f"Plasma CLZ R2: {plasma_clz_r2}")
        print(f"Brain CLZ R2: {brain_clz_r2}")

        ax[0,0].plot(predicted_nmol.ts, plasma_cno_conc, color='k')
        ax[0,0].errorbar(plasma_cno["time(min)"], plasma_cno["concentration(nM)"], color="k", fmt="o")
        ax[0,0].set_ylabel("Plasma CNO (nM)")

        ax[0,1].plot(predicted_nmol.ts, plasma_clz_conc, color='k')
        ax[0,1].errorbar(plasma_clz["time(min)"], plasma_clz["concentration(nM)"], color="k", fmt="o")
        ax[0,1].set_ylabel("Plasma CLZ (nM)")

        ax[1,0].plot(predicted_nmol.ts, brain_cno_conc, color='k')
        ax[1,0].errorbar(brain_cno["time(min)"], brain_cno["concentration(nM)"], color="k", fmt="o")
        ax[1,0].set_ylabel("Brain CNO (nM)")
        ax[1,0].set_xlabel("Time (min)")

        ax[1,1].plot(predicted_nmol.ts, brain_clz_conc, color='k')
        ax[1,1].errorbar(brain_clz["time(min)"], brain_clz["concentration(nM)"], color="k", fmt="o")
        ax[1,1].set_ylabel("Brain CLZ (nM)")
        ax[1,1].set_xlabel("Time (min)")

        plt.tight_layout()
    return (inspect_fit,)


@app.cell
def _(best_params, inspect_fit, plt):
    inspect_fit(best_params)
    plt.gca()

    return


@app.cell
def _(best_params, jnp):
    param_estimates = jnp.concatenate([best_params[0:9]*60, best_params[9:]])
    return (param_estimates,)


@app.cell
def _(
    CnoPK,
    CnoPKConfig,
    Kvaerno3,
    ODETerm,
    PIDController,
    SaveAt,
    data_dir,
    diffeqsolve,
    jnp,
    mouse_weight,
    os,
    param_estimates,
    plt,
):
    model = CnoPK(CnoPKConfig(
        1 * mouse_weight,
        *param_estimates,
    ))

    predicted_nmol = diffeqsolve(
        ODETerm(model._model),
        t0=0,
        t1=72,
        dt0=0.1,
        y0=(3.5 * mouse_weight / 342.8 * 1e6,0,0,0,0),
        stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
        saveat=SaveAt(ts=jnp.linspace(0, 72, 72*15)),
        solver=Kvaerno3()
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    ax[0,0].plot(predicted_nmol.ts, predicted_nmol.ys[1] / model.cno_plasma_vd)
    ax[0,0].set_ylabel('plasma CNO (nM)')

    ax[0,1].plot(predicted_nmol.ts, predicted_nmol.ys[3] / model.clz_plasma_vd)
    ax[0,1].set_ylabel('plasma CLZ (nM)')

    ax[1,0].plot(predicted_nmol.ts, predicted_nmol.ys[2] / model.cno_brain_vd, 'k')
    ax[1,0].set_ylabel('Brain CNO (nM)')
    ax[1,0].set_xlabel('Time (hr)')

    ax[1,1].plot(predicted_nmol.ts, predicted_nmol.ys[4] / model.clz_brain_vd, 'k')
    ax[1,1].set_ylabel('Brain CLZ (nM)')
    ax[1,1].set_xlabel('Time (hr)')

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "cno_pk_fit_hrs.svg"))
    plt.gca()
    return (model,)


@app.cell
def _(model):
    print(f"CNO Vd: {model.cno_brain_vd}")
    print(f"CLZ Vd: {model.clz_brain_vd}")
    return


@app.cell
def _(jnp, os, param_estimates):
    data_dir = os.path.join("notebooks", "data", "cno_pk")
    jnp.save(os.path.join(data_dir, "params_estimate.npy"), param_estimates)
    return (data_dir,)


if __name__ == "__main__":
    app.run()
