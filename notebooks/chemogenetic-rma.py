import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models import ChemogeneticRMA, DoxPK, DoxPKConfig, CnoPK, CnoPKConfig
    from gluc import get_gluc_conc
    from diffrax import PIDController, ClipStepSizeController, StepTo, SaveAt, RESULTS, Kvaerno5
    from jax import config, numpy as jnp
    from jax.lax import cond
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
    return (gluc_df,)


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
def _(ParticleSwarm, get_best_loss_and_params, gluc_df, loss, partial, pl):
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

    mean_gluc = gluc_df.group_by("time").agg([
        pl.col("gluc").mean().alias("mean_gluc"),
        pl.col("gluc").std().alias("std_gluc")
    ])

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


app._unparsable_cell(
    r"""
    - good for tracking dynamics and optimising experiments
    - optimisation of temporal resolution
    - what are the most important contributors to the kinetics (i.e, sensitivy of params)

    - serum markers are useful (limited so synthetic is better)
    - slow kinetics (long half lives)
    - developed synthetic serum markers that allow studying gene expression
    - to develop better tools, we wanted to explore which parameters are most important
    - explored transcytosis, overall kinetics, production
    - in order to make better tools, more applicable, we evaluated what are the limits of the temporal resolution
        - in different context like const or drug induced
    - overall, we will be able to build better RMA based tools
    - inform parameters that are necessary for optimal detection

    - these results can be applicable to RMAs other markers, and Focused ultrasound liquid biopsy (Joon)
    - can adapt these models can be adapted to other scenarios allowing to direct engineering efforts

    discussion
    - how this model can improve the design (i.e., design stronger promoters for RMAs)
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
