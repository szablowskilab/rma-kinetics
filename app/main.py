import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", app_title="RMA Kinetics")


@app.cell
def _():
    from rma_kinetics.models import (
        ConstitutiveRMA,
        TetRMA,
        ChemogeneticRMA,
        DoxPKConfig,
        CnoPKConfig
    )
    from rma_kinetics.units import Time, Concentration
    import defaults
    import validators

    from jax import config as jax_config, numpy as jnp
    from diffrax import SaveAt, PIDController

    import marimo as mo
    import seaborn as sb
    import matplotlib.pyplot as plt

    jax_config.update("jax_enable_x64", True)
    sb.set_theme("talk", "ticks", font="Arial")
    return (
        ConstitutiveRMA,
        DoxPKConfig,
        PIDController,
        SaveAt,
        TetRMA,
        defaults,
        jnp,
        mo,
        plt,
        validators,
    )


@app.cell
def _(mo):
    # common parameters - needed by rates, so we have to keep them in a separate cell
    concentration_units = mo.ui.dropdown(
            options=["nM", "µM", "mM"],
            label="Concentration units",
            value="nM"
    )

    time_units = mo.ui.dropdown(
            options=["sec", "min", "hr"],
            label="Time units",
            value="hr"
    )
    return concentration_units, time_units


@app.cell
def _(concentration_units, defaults, mo, time_units):
    # run config
    start_time = mo.ui.number(
        value=defaults.T0,
        label=f"Start time ({time_units.value})"
    )

    stop_time = mo.ui.number(
        value=defaults.T1,
        label=f"Stop time ({time_units.value})"
    )

    sampling_rate = mo.ui.number(
        value=defaults.FS,
        label=f"Sampling rate (1/{time_units.value})"
    )

    dt0 = mo.ui.number(
        value=defaults.DT0,
        label="Initial step size"
    )

    init_tissue_rma = mo.ui.number(
        value=defaults.INIT_BRAIN_RMA,
        label=f"Initial brain RMA ({concentration_units.value})"
    )

    init_blood_rma = mo.ui.number(
        value=defaults.INIT_PLASMA_RMA,
        label=f"Initial plasma RMA ({concentration_units.value})"
    )

    init_tta = mo.ui.number(
        value=10,
        label=f"Initial tTA ({concentration_units.value})"
    )

    # RMA rates

    rma_prod_rate = mo.ui.number(
        value=defaults.RMA_PROD_RATE,
        label=f"Production rate ({concentration_units.value}/{time_units.value})"
    )

    rma_rt_rate = mo.ui.number(
        value=defaults.RMA_RT_RATE,
        label=f"Reverse transcytosis rate (1/{time_units.value})"
    )

    rma_deg_rate = mo.ui.number(
        value=defaults.RMA_DEG_RATE,
        label=f"Degradation rate (1/{time_units.value})"
    )

    # Tet-Off params
    dox_vehicle_intake_rate = mo.ui.number(
        value=defaults.DOX_VEHICLE_INTAKE_RATE,
        label=f"Dox vehicle intake rate (1/{time_units.value})"
    )

    dox_bioavailability = mo.ui.number(
        value=defaults.DOX_BIOAVAILABILITY,
        label="Dox bioavailability (i.e., number between [0, 1])"
    )

    dox_vehicle_dose = mo.ui.number(
        value=defaults.DOX_VEHICLE_DOSE,
        label="Dox vehicle dose (mg dox per unit food/water)"
    )

    dox_absorption_rate = mo.ui.number(
        value=defaults.DOX_ABSORPTION_RATE,
        label=f"Dox absorption rate (1/{time_units.value})"
    )

    dox_elimination_rate = mo.ui.number(
        value=defaults.DOX_ELIMINATION_RATE,
        label=f"Dox elimination rate (1/{time_units.value})"
    )

    dox_brain_transport_rate = mo.ui.number(
        value=defaults.DOX_BRAIN_TRANSPORT_RATE,
        label=f"Dox plasma to brain transport rate (1/{time_units.value})"
    )

    dox_plasma_transport_rate = mo.ui.number(
        value=defaults.DOX_PLASMA_TRANSPORT_RATE,
        label=f"Dox plasma to brain transport rate (1/{time_units.value})"
    )

    dox_plasma_vd = mo.ui.number(
        value=defaults.DOX_PLASMA_VD,
        label=f"Plasma dox volume of distribution (L)"
    )

    dox_kd = mo.ui.number(
        value=defaults.DOX_KD,
        label=f"Dox-tTA dissocation constant ({concentration_units.value})"
    )

    dox_t0 = mo.ui.number(
        value=0,
        label=f"Dox administration start time ({time_units.value})"
    )

    dox_t1 = mo.ui.number(
        value=48,
        label=f"Dox administration start time ({time_units.value})"
    )

    tta_prod_rate = mo.ui.number(
        value=defaults.TTA_PROD_RATE,
        label=f"tTA production rate ({concentration_units.value}/{time_units.value})"
    )

    tta_deg_rate = mo.ui.number(
        value=defaults.TTA_DEG_RATE,
        label=f"tTA degradation rate (1/{time_units.value})"
    )

    tta_kd = mo.ui.number(
        value=defaults.TTA_KD,
        label=f"tTA-TetO dissocation constant ({concentration_units.value})"
    )

    leaky_rma_prod_rate = mo.ui.number(
        value=defaults.LEAKY_RMA_PROD_RATE,
        label=f"Leaky RMA production rate ({concentration_units.value}/{time_units.value}"
    )

    init_brain_dox = mo.ui.number(
        value=10,
        label=f"Initial brain dox ({concentration_units.value})"
    )

    init_plasma_dox = mo.ui.number(
        value=10,
        label=f"Initial plasma dox ({concentration_units.value})"
    )



    return (
        dox_absorption_rate,
        dox_bioavailability,
        dox_brain_transport_rate,
        dox_elimination_rate,
        dox_kd,
        dox_plasma_transport_rate,
        dox_plasma_vd,
        dox_t0,
        dox_t1,
        dox_vehicle_dose,
        dox_vehicle_intake_rate,
        dt0,
        init_blood_rma,
        init_brain_dox,
        init_plasma_dox,
        init_tissue_rma,
        init_tta,
        leaky_rma_prod_rate,
        rma_deg_rate,
        rma_prod_rate,
        rma_rt_rate,
        sampling_rate,
        start_time,
        stop_time,
        tta_deg_rate,
        tta_kd,
        tta_prod_rate,
    )


@app.cell
def _(
    concentration_units,
    dox_absorption_rate,
    dox_bioavailability,
    dox_brain_transport_rate,
    dox_elimination_rate,
    dox_plasma_transport_rate,
    dox_plasma_vd,
    dox_t0,
    dox_t1,
    dox_vehicle_dose,
    dox_vehicle_intake_rate,
    dt0,
    init_blood_rma,
    init_brain_dox,
    init_plasma_dox,
    init_tissue_rma,
    init_tta,
    mo,
    rma_deg_rate,
    rma_prod_rate,
    rma_rt_rate,
    sampling_rate,
    start_time,
    stop_time,
    time_units,
    tta_deg_rate,
    tta_kd,
    tta_prod_rate,
):
    base_params_ui = mo.vstack([
        mo.md("## Simulation Configuration"),
        concentration_units,
        time_units,
        start_time,
        stop_time,
        mo.accordion({
            "More options": mo.vstack([
                sampling_rate,
                dt0
            ])
        }),
        mo.md("## RMA Parameters"),
        rma_prod_rate,
        rma_rt_rate,
        rma_deg_rate,
        mo.accordion({
            "More options": mo.vstack([
                init_tissue_rma,
                init_blood_rma,
            ])
        }),
    ])

    tet_params_ui = mo.vstack([
        base_params_ui,
        mo.md("## Tet-Off Parameters"),
        dox_vehicle_dose,
        dox_t0,
        dox_t1,
        mo.accordion({
            "More options": mo.vstack([
                dox_vehicle_intake_rate,
                dox_bioavailability,
                dox_absorption_rate,
                dox_elimination_rate,
                dox_brain_transport_rate,
                dox_plasma_transport_rate,
                dox_plasma_vd,
                tta_prod_rate,
                tta_deg_rate,
                tta_kd,
                init_tta,
                init_brain_dox,
                init_plasma_dox
            ])
        })
    ])

    return base_params_ui, tet_params_ui


@app.cell
def _(
    dox_absorption_rate,
    dox_bioavailability,
    dox_brain_transport_rate,
    dox_elimination_rate,
    dox_kd,
    dox_plasma_transport_rate,
    dox_plasma_vd,
    dox_t0,
    dox_t1,
    dox_vehicle_dose,
    dox_vehicle_intake_rate,
    leaky_rma_prod_rate,
    rma_deg_rate,
    rma_prod_rate,
    rma_rt_rate,
    tta_deg_rate,
    tta_kd,
    tta_prod_rate,
):
    constitutive_params = {
        "rma_prod_rate": rma_prod_rate.value,
        "rma_rt_rate": rma_rt_rate.value,
        "rma_deg_rate": rma_deg_rate.value
    }

    dox_pk_params = {
        "vehicle_intake_rate": dox_vehicle_intake_rate.value,
        "bioavailability": dox_bioavailability.value,
        "dose": dox_vehicle_dose.value,
        "absorption_rate": dox_absorption_rate.value,
        "elimination_rate": dox_elimination_rate.value,
        "brain_transport_rate": dox_brain_transport_rate.value,
        "plasma_transport_rate": dox_plasma_transport_rate.value,
        "t0": dox_t0.value,
        "t1": dox_t1.value,
        "plasma_vd": dox_plasma_vd.value
    }

    tet_params = constitutive_params | {
        "dox_kd": dox_kd.value,
        "tta_prod_rate": tta_prod_rate.value,
        "tta_deg_rate": tta_deg_rate.value,
        "tta_kd": tta_kd.value,
        "leaky_rma_prod_rate": leaky_rma_prod_rate.value
    }
    return constitutive_params, dox_pk_params, tet_params


@app.cell
def _(base_params_ui, mo, tet_params_ui):
    model_selection = mo.ui.tabs({
        "Constitutive": base_params_ui,
        "Tet-Off": tet_params_ui,
        "Chemogenetic": mo.ui.checkbox()
    })
    return (model_selection,)


@app.cell
def _(mo):
    header = mo.hstack([
        mo.md("[Szablowski Lab](https://szablowskilab.org)"),
        mo.md("[GitHub](https://github.com/szablowskilab)"),
        mo.md("[Help](https://github.com/szablowskilab/rma-kinetics/docs)")
    ], justify="start", gap=2.5)
    return (header,)


@app.cell
def _(header, mo, model_selection):
    update_button = mo.ui.button(label="Update Parameters")
    run_button = mo.ui.run_button(label="Run", kind="success")
    mo.vstack([
        mo.md("# RMA Kinetics Simulator"),
        header,
        model_selection,
        mo.hstack([update_button, run_button], justify="start", gap=2.5),
        mo.md("If you found this useful, consider citing our paper: [Buitrago et al., 2025]()")
    ], gap=2.5)
    return (run_button,)


@app.cell
def _(
    ConstitutiveRMA,
    DoxPKConfig,
    PIDController,
    SaveAt,
    TetRMA,
    constitutive_params,
    dox_pk_params,
    dt0,
    init_blood_rma,
    init_brain_dox,
    init_plasma_dox,
    init_tissue_rma,
    init_tta,
    jnp,
    mo,
    model_selection,
    run_button,
    sampling_rate,
    start_time,
    stop_time,
    tet_params,
    validators,
):
    mo.stop(not run_button.value)
    def run():
        sol = None
        species_options = ["Brain RMA", "Plasma RMA"]
        with mo.status.spinner(title="Running...") as spinner:
            try:
                # validate inputs
                validators.time(start_time.value, stop_time.value)
                #params = [validate_num(key, scalars[key]) for key in scalars]
    
                # run simulation
                sim_config = {
                    "t0": start_time.value,
                    "t1": stop_time.value,
                    "dt0": dt0.value,
                    "saveat": SaveAt(ts=jnp.linspace(
                        start_time.value,
                        stop_time.value,
                        stop_time.value*sampling_rate.value)),
                    "stepsize_controller": PIDController(rtol=1e-5, atol=1e-5)
                }
    
                if model_selection.value == "Constitutive":
                    sim_config["y0"] = (init_tissue_rma.value, init_blood_rma.value)
                    model = ConstitutiveRMA(**constitutive_params)
                
                elif model_selection.value == "Tet-Off":
                    species_options += ["tTA", "Brain Dox", "Plasma Dox"]
                
                    sim_config["y0"] = (
                        init_tissue_rma.value,
                        init_blood_rma.value,
                        init_tta.value,
                        init_brain_dox.value,
                        init_plasma_dox.value
                    )
                
                    dox_model_config = DoxPKConfig(**dox_pk_params)
                    model = TetRMA(**tet_params, dox_model_config=dox_model_config)
    
                sol = model.simulate(**sim_config)
                return sol, species_options
    
                # plot results
                #sol._plot_species(species_selector.value)
                #sb.despine()
                #plt.tight_layout()
                #output = plt.gcf()
    
            except Exception as e:
                mo.stop(True, mo.callout(kind="danger", value=str(e)))
    
    sol, species_options = run()
    return sol, species_options


@app.cell
def _(mo, species_options):
    species_selector = mo.ui.dropdown(
        options=species_options,
        value="Plasma RMA",
        label="view",
        searchable=True,
    )
    return (species_selector,)


@app.cell
def _(mo, plt, sol, species_selector):
    if sol is not None:
        sol._plot_species(species_selector.value)
        plt.tight_layout()
        fig = plt.gcf()
    
    mo.hstack([
        mo.mpl.interactive(fig),
        species_selector
    ], widths=[0.85, 0.25])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
