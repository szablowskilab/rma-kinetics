import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from rma_kinetics.models import ConstitutiveRMA, TetRMA, ChemogeneticRMA, DoxPKConfig, CnoPKConfig
    from rma_kinetics.units import Time, Concentration
    from diffrax import SaveAt
    from jax import config as jax_config, numpy as jnp

    import matplotlib.pyplot as plt
    import seaborn as sb

    jax_config.update("jax_enable_x64", True)
    sb.set_theme(context="talk", style="ticks")
    return (
        ChemogeneticRMA,
        CnoPKConfig,
        Concentration,
        ConstitutiveRMA,
        DoxPKConfig,
        TetRMA,
        Time,
        mo,
        plt,
        sb,
    )


@app.cell
def _(Concentration, Time, mo):
    # units for formatting - used in rates so we have to put in a separate cell
    concentration_units = mo.ui.dropdown(
            options=["nM", "µM", "mM"],
            label="Concentration units",
            value="nM"
    )

    conc_map = {
        "nM": Concentration.nanomolar,
        "µM": Concentration.micromolar,
        "mM": Concentration.millimolar
    }

    time_units = mo.ui.dropdown(
            options=["sec", "min", "hr"],
            label="Time units",
            value="hr"
    )

    time_map = {
        "sec": Time.seconds,
        "min": Time.minutes,
        "hr": Time.hours
    }
    return conc_map, concentration_units, time_map, time_units


@app.cell
def _(concentration_units, mo, time_units):
    # Individual inputs for ui elements
    # simulation parameters
    start_time = mo.ui.number(
        value=0,
        label=f"Start time ({time_units.value})"
    )

    stop_time = mo.ui.number(
        value=504,
        label=f"Stop time ({time_units.value})"
    )

    sampling_rate = mo.ui.number(
        value=1,
        label=f"Sampling rate (1/{time_units.value})"
    )

    dt0 = mo.ui.number(
        value=0.1,
        label="Initial step size"
    )

    simulation_params = mo.vstack([
        mo.md("## Simulation Parameters"),
        mo.hstack([
            time_units,
            concentration_units
        ], justify="start"),
        start_time,
        stop_time,
        mo.accordion({
            "More options": mo.vstack([
                sampling_rate,
                dt0
            ])
        })
    ])

    # RMA rates
    rma_prod_rate = mo.ui.number(
        value=5e-3,
        label=f"Production rate ({concentration_units.value}/{time_units.value})"
    )

    rma_rt_rate = mo.ui.number(
        value=0.6,
        label=f"Reverse transcytosis rate (1/{time_units.value})"
    )

    rma_deg_rate = mo.ui.number(
        value=7e-3,
        label=f"Degradation rate (1/{time_units.value})"
    )

    leaky_rma_prod_rate = mo.ui.number(
        value=5e-5,
        label=f"Leaky production rate ({concentration_units.value}/{time_units.value})"
    )

    rma_params = mo.vstack([
        mo.md("## RMA Rates"),
        rma_prod_rate,
        rma_rt_rate,
        rma_deg_rate
    ])

    extended_rma_params = mo.vstack([
        mo.md("## RMA Rates"),
        rma_prod_rate,
        leaky_rma_prod_rate,
        rma_rt_rate,
        rma_deg_rate,
    ])

    # dox parameters
    dox_dose = mo.ui.number(
        value=40,
        label="Dose (mg/kg food)"
    )

    dox_t0 = mo.ui.number(
        value=0,
        label=f"Dox administration start time ({time_units.value})"
    )

    dox_t1 = mo.ui.number(
        value=96,
        label=f"Dox administration stop time ({time_units.value})"
    )

    dox_vehicle_intake_rate = mo.ui.number(
        value=1.875e-4,
        label= f"Food intake rate (kg/{time_units.value})"
    )

    dox_bioavailability = mo.ui.number(
        value=0.9,
        label="Dox Bioavailability (0, 1]"
    )

    dox_absorption_rate = mo.ui.number(
        value=0.8,
        label=f"Dox absorption rate (1/{time_units.value})"
    )

    dox_elimination_rate = mo.ui.number(
        value=0.2,
        label=f"Dox elimination rate (1/{time_units.value})"
    )

    dox_brain_transport_rate = mo.ui.number(
        value=0.2,
        label=f"Dox brain transport rate (1/{time_units.value})"
    )

    dox_plasma_transport_rate = mo.ui.number(
        value=1,
        label=f"Dox plasma transport rate (1/{time_units.value})"
    )

    dox_plasma_vd = mo.ui.number(
        value=0.21,
        label="Dox plasma volume of distribution (L)"
    )

    dox_kd = mo.ui.number(
        value=10,
        label=f"tTA-Dox Kd ({concentration_units.value})"
    )

    # tTA rates
    tta_prod_rate = mo.ui.number(
        value=10,
        label=f"tTA production rate ({concentration_units.value}/{time_units.value})"
    )

    leaky_tta_prod_rate = mo.ui.number(
        value=3e-3,
        label=f"Leaky tTA production rate ({concentration_units.value}/{time_units.value})"
    )

    tta_deg_rate = mo.ui.number(
        value=1,
        label=f"tTA degradation rate (1/{time_units.value})"
    )

    inducible_tta_deg_rate = mo.ui.number(
        value=0.1,
        label=f"tTA degradation rate (1/{time_units.value})"
    )

    tta_kd = mo.ui.number(
        value=10,
        label=f"tTA-TetO operator Kd ({concentration_units.value})"
    )

    tta_coop = mo.ui.number(
        value=2,
        label="tTA cooperativity (Hill coefficient)"
    )

    dox_params = mo.vstack([
        mo.md("## Doxycycline Administration"),
        dox_dose,
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
                dox_kd,
            ])
        })
    ])

    tta_params = mo.vstack([
        mo.md("## tTA"),
        tta_prod_rate,
        tta_deg_rate,
        tta_kd,
        tta_coop
    ])

    extended_tta_params = mo.vstack([
        mo.md("## tTA"),
        tta_prod_rate,
        leaky_tta_prod_rate,
        inducible_tta_deg_rate,
        tta_kd,
        tta_coop
    ])

    # cno parameters
    cno_dose = mo.ui.number(
        value=0.075,
        label="Clozapine-N-oxide (CNO) dose (mg)"
    )

    cno_t0 = mo.ui.number(
        value=48,
        label=f"CNO administration start time ({time_units.value})"
    )

    cno_absorption_rate = mo.ui.number(
        value=23.94,
        label=f"CNO absorption rate (1/{time_units.value})"
    )

    cno_elimination_rate = mo.ui.number(
        value=5.51e-2,
        label=f"CNO elimination rate (1/{time_units.value})"
    )

    cno_revmet_rate = mo.ui.number(
        value=1.44,
        label=f"CNO reverse metabolism rate (1/{time_units.value})"
    )

    clz_met_rate = mo.ui.number(
        value=3e-1,
        label=f"CLZ metabolism rate (1/{time_units.value})"
    )

    cno_brain_transport_rate = mo.ui.number(
        value=2.33,
        label=f"CNO brain transport rate (1/{time_units.value})"
    )

    cno_plasma_transport_rate = mo.ui.number(
        value=71.85,
        label=f"CNO plasma transport rate (1/{time_units.value})"
    )

    clz_brain_transport_rate = mo.ui.number(
        value=35.62,
        label=f"CLZ brain transport rate (1/{time_units.value})"
    )

    clz_plasma_transport_rate = mo.ui.number(
        value=34.07,
        label=f"CLZ plasma transport rate (1/{time_units.value})"
    )

    clz_elimination_rate = mo.ui.number(
        value=3.94,
        label=f"CLZ elimination rate (1/{time_units.value})"
    )

    cno_plasma_vd = mo.ui.number(
        value=3.99e-2,
        label="CNO plasma volume of distribution (L)"
    )

    cno_brain_vd = mo.ui.number(
        value=0.21,
        label="CNO brain volume of distribution (L)"
    )

    clz_plasma_vd = mo.ui.number(
        value=0.24,
        label="CLZ plasma volume of distribution (L)"
    )

    clz_brain_vd = mo.ui.number(
        value=8.87e-2,
        label="CLZ brain volume of distribution (L)"
    )

    cno_ec50 = mo.ui.number(
        value=7,
        label=f"CNO EC50 ({concentration_units.value})"
    )

    clz_ec50 = mo.ui.number(
        value=3,
        label=f"CLZ EC50 ({concentration_units.value})"
    )

    cno_coop = mo.ui.number(
        value=1,
        label="CNO cooperativity (Hill coefficient)"
    )

    clz_coop = mo.ui.number(
        value=1,
        label="CLZ cooperativity (Hill coefficient)"
    )

    # hM3Dq parameters
    hm3dq_prod_rate = mo.ui.number(
        value=10,
        label=f"HM3Dq production rate ({concentration_units.value}/{time_units.value})"
    )

    hm3dq_deg_rate = mo.ui.number(
        value=1,
        label=f"HM3Dq degradation rate (1/{time_units.value})"
    )

    hm3dq_ec50 = mo.ui.number(
        value=3,
        label=f"HM3Dq EC50 ({concentration_units.value})"
    )

    hm3dq_coop = mo.ui.number(
        value=1,
        label="HM3Dq cooperativity (Hill coefficient)"
    )

    cno_params = mo.vstack([
        mo.md("## CNO Administration"),
        cno_dose,
        cno_t0,
        mo.accordion({
            "More options": mo.vstack([
                cno_absorption_rate,
                cno_elimination_rate,
                cno_revmet_rate,
                clz_met_rate,
                cno_brain_transport_rate,
                cno_plasma_transport_rate,
                clz_brain_transport_rate,
                clz_plasma_transport_rate,
                clz_elimination_rate,
                cno_plasma_vd,
                cno_brain_vd,
                clz_plasma_vd,
                clz_brain_vd,
                cno_ec50,
                clz_ec50,
                cno_coop,
                clz_coop
            ])
        })
    ])

    hm3dq_params = mo.vstack([
        mo.md("## hM3Dq"),
        hm3dq_prod_rate,
        hm3dq_deg_rate,
        mo.accordion({
            "More options": mo.vstack([
                hm3dq_ec50,
                hm3dq_coop
            ])
        })
    ])

    return (
        clz_brain_transport_rate,
        clz_brain_vd,
        clz_coop,
        clz_ec50,
        clz_elimination_rate,
        clz_met_rate,
        clz_plasma_transport_rate,
        clz_plasma_vd,
        cno_absorption_rate,
        cno_brain_transport_rate,
        cno_brain_vd,
        cno_coop,
        cno_dose,
        cno_ec50,
        cno_elimination_rate,
        cno_params,
        cno_plasma_transport_rate,
        cno_plasma_vd,
        cno_revmet_rate,
        cno_t0,
        dox_absorption_rate,
        dox_bioavailability,
        dox_brain_transport_rate,
        dox_dose,
        dox_elimination_rate,
        dox_kd,
        dox_params,
        dox_plasma_transport_rate,
        dox_plasma_vd,
        dox_t0,
        dox_t1,
        dox_vehicle_intake_rate,
        dt0,
        extended_rma_params,
        extended_tta_params,
        hm3dq_coop,
        hm3dq_deg_rate,
        hm3dq_ec50,
        hm3dq_params,
        hm3dq_prod_rate,
        leaky_rma_prod_rate,
        leaky_tta_prod_rate,
        rma_deg_rate,
        rma_params,
        rma_prod_rate,
        rma_rt_rate,
        sampling_rate,
        simulation_params,
        start_time,
        stop_time,
        tta_coop,
        tta_deg_rate,
        tta_kd,
        tta_params,
        tta_prod_rate,
    )


@app.cell
def _(
    concentration_units,
    hm3dq_deg_rate,
    hm3dq_prod_rate,
    mo,
    tta_deg_rate,
    tta_prod_rate,
):
    # initial conditions
    init_brain_rma = mo.ui.number(
        value=0,
        label=f"Brain RMA ({concentration_units.value})"
    )

    init_plasma_rma = mo.ui.number(
        value=0,
        label=f"Plasma RMA ({concentration_units.value})"
    )

    init_constitutive_tta = mo.ui.number(
        value=tta_prod_rate.value / tta_deg_rate.value,
        label=f"tTA ({concentration_units.value})"
    )

    init_inducible_tta = mo.ui.number(
        value=0,
        label=f"tTA ({concentration_units.value})"
    )

    init_brain_dox = mo.ui.number(
        value=0,
        label=f"Brain Dox ({concentration_units.value})"
    )

    init_plasma_dox = mo.ui.number(
        value=0,
        label=f"Plasma Dox ({concentration_units.value})"
    )

    init_hm3dq = mo.ui.number(
        value=hm3dq_prod_rate.value / hm3dq_deg_rate.value,
        label=f"HM3Dq ({concentration_units.value})"
    )

    init_peritoneal_cno = mo.ui.number(
        value=0,
        label=f"Peritoneal CNO ({concentration_units.value})"
    )

    init_brain_cno = mo.ui.number(
        value=0,
        label=f"Brain CNO ({concentration_units.value})"
    )

    init_plasma_cno = mo.ui.number(
        value=0,
        label=f"Plasma CNO ({concentration_units.value})"
    )

    init_brain_clz = mo.ui.number(
        value=0,
        label=f"Brain CLZ ({concentration_units.value})"
    )

    init_plasma_clz = mo.ui.number(
        value=0,
        label=f"Plasma CLZ ({concentration_units.value})"
    )
    return (
        init_brain_clz,
        init_brain_cno,
        init_brain_dox,
        init_brain_rma,
        init_constitutive_tta,
        init_hm3dq,
        init_inducible_tta,
        init_peritoneal_cno,
        init_plasma_clz,
        init_plasma_cno,
        init_plasma_dox,
        init_plasma_rma,
    )


@app.cell
def _(
    cno_params,
    dox_params,
    extended_rma_params,
    extended_tta_params,
    hm3dq_params,
    init_brain_clz,
    init_brain_cno,
    init_brain_dox,
    init_brain_rma,
    init_constitutive_tta,
    init_hm3dq,
    init_inducible_tta,
    init_peritoneal_cno,
    init_plasma_clz,
    init_plasma_cno,
    init_plasma_dox,
    init_plasma_rma,
    mo,
    rma_params,
    simulation_params,
    tta_params,
):
    # form elements
    base_params = mo.vstack([
        mo.vstack([mo.md("")]),
        mo.vstack([
            simulation_params,
            rma_params,
            mo.accordion({
                "Initial Conditions": mo.vstack([
                    init_brain_rma,
                    init_plasma_rma,
                ])
            })
        ], gap=1.25),
    ], gap=1.25)

    tetoff_params = mo.vstack([
        mo.vstack([mo.md("")]),
        mo.vstack([
            mo.hstack([
                mo.vstack([
                    simulation_params,
                    extended_rma_params,
                ]),
                mo.vstack([
                    dox_params,
                    tta_params,
                ])
            ]),
            mo.accordion({
                "Initial Conditions": mo.vstack([
                    init_brain_rma,
                    init_plasma_rma,
                    init_constitutive_tta,
                    init_brain_dox,
                    init_plasma_dox
                ])
            })
        ], gap=1.25),
    ], gap=1.25)

    chemogenetic_params = mo.vstack([
        mo.vstack([mo.md("")]),
        mo.vstack([
            mo.hstack([
                mo.vstack([
                    simulation_params,
                    extended_rma_params,
                    hm3dq_params
                ]),
                mo.vstack([
                    dox_params,
                    extended_tta_params,
                    cno_params
                ])
            ]),
            mo.accordion({
                "Initial Conditions": mo.vstack([
                    init_brain_rma,
                    init_plasma_rma,
                    init_inducible_tta,
                    init_brain_dox,
                    init_plasma_dox,
                    init_hm3dq,
                    init_peritoneal_cno,
                    init_brain_cno,
                    init_plasma_cno,
                    init_brain_clz,
                    init_plasma_clz
                ])
            })
        ], gap=1.25),
    ], gap=1.25)



    model_selection = mo.ui.tabs({
        "Constitutive": base_params,
        "Tet-Off": tetoff_params,
        "Chemogenetic": chemogenetic_params,
    })
    return (model_selection,)


@app.cell
def _(
    CnoPKConfig,
    DoxPKConfig,
    clz_brain_transport_rate,
    clz_brain_vd,
    clz_elimination_rate,
    clz_met_rate,
    clz_plasma_transport_rate,
    clz_plasma_vd,
    cno_absorption_rate,
    cno_brain_transport_rate,
    cno_brain_vd,
    cno_dose,
    cno_elimination_rate,
    cno_plasma_transport_rate,
    cno_plasma_vd,
    cno_revmet_rate,
    dox_absorption_rate,
    dox_bioavailability,
    dox_brain_transport_rate,
    dox_dose,
    dox_elimination_rate,
    dox_plasma_transport_rate,
    dox_plasma_vd,
    dox_t0,
    dox_t1,
    dox_vehicle_intake_rate,
):
    # dox and cno models
    dox_model_config = DoxPKConfig(
        dox_dose.value,
        dox_t0.value,
        dox_t1.value,
        dox_vehicle_intake_rate.value,
        dox_bioavailability.value,
        dox_absorption_rate.value,
        dox_elimination_rate.value,
        dox_brain_transport_rate.value,
        dox_plasma_transport_rate.value,
        dox_plasma_vd.value
    )

    cno_model_config = CnoPKConfig(
        cno_dose=cno_dose.value,
        cno_absorption_rate=cno_absorption_rate.value,
        cno_elimination_rate=cno_elimination_rate.value,
        cno_reverse_metabolism_rate=cno_revmet_rate.value,
        clz_metabolism_rate=clz_met_rate.value,
        cno_brain_transport_rate=cno_brain_transport_rate.value,
        cno_plasma_transport_rate=cno_plasma_transport_rate.value,
        clz_brain_transport_rate=clz_brain_transport_rate.value,
        clz_plasma_transport_rate=clz_plasma_transport_rate.value,
        clz_elimination_rate=clz_elimination_rate.value,
        cno_plasma_vd=cno_plasma_vd.value,
        cno_brain_vd=cno_brain_vd.value,
        clz_plasma_vd=clz_plasma_vd.value,
        clz_brain_vd=clz_brain_vd.value
    )

    return cno_model_config, dox_model_config


@app.cell
def _(
    ChemogeneticRMA,
    ConstitutiveRMA,
    TetRMA,
    clz_coop,
    clz_ec50,
    cno_coop,
    cno_ec50,
    cno_model_config,
    cno_t0,
    conc_map,
    concentration_units,
    dox_kd,
    dox_model_config,
    dt0,
    hm3dq_coop,
    hm3dq_deg_rate,
    hm3dq_ec50,
    hm3dq_prod_rate,
    init_brain_clz,
    init_brain_cno,
    init_brain_dox,
    init_brain_rma,
    init_constitutive_tta,
    init_hm3dq,
    init_inducible_tta,
    init_peritoneal_cno,
    init_plasma_clz,
    init_plasma_cno,
    init_plasma_dox,
    init_plasma_rma,
    leaky_rma_prod_rate,
    leaky_tta_prod_rate,
    mo,
    rma_deg_rate,
    rma_prod_rate,
    rma_rt_rate,
    sampling_rate,
    start_time,
    stop_time,
    time_map,
    time_units,
    tta_coop,
    tta_deg_rate,
    tta_kd,
    tta_prod_rate,
):
    def run_simulation(model_selection):
        with mo.status.spinner(title="Running..."):
            species = ["Brain RMA", "Plasma RMA"]
            match model_selection.value:
                case "Constitutive":
                    model = ConstitutiveRMA(
                        rma_prod_rate.value,
                        rma_rt_rate.value,
                        rma_deg_rate.value,
                        time_map[time_units.value],
                        conc_map[concentration_units.value]
                    )
                    y0 = (init_brain_rma.value, init_plasma_rma.value)

                case "Tet-Off":
                    species += ["tTA", "Dox"]

                    model = TetRMA(
                        rma_prod_rate.value,
                        rma_rt_rate.value,
                        rma_deg_rate.value,
                        dox_model_config=dox_model_config,
                        dox_kd=dox_kd.value,
                        tta_prod_rate=tta_prod_rate.value,
                        tta_deg_rate=tta_deg_rate.value,
                        tta_kd=tta_kd.value,
                        leaky_rma_prod_rate=leaky_rma_prod_rate.value,
                        tta_coop=tta_coop.value,
                        time_units=time_map[time_units.value],
                        conc_units=conc_map[concentration_units.value]
                    )

                    y0 = (
                        init_brain_rma.value,
                        init_plasma_rma.value,
                        init_constitutive_tta.value,
                        init_brain_dox.value,
                        init_plasma_dox.value
                    )

                case "Chemogenetic":
                    species += ["tTA", "Dox", "hM3Dq", "CNO", "CLZ"]
                    model = ChemogeneticRMA(
                        rma_prod_rate.value,
                        rma_rt_rate.value,
                        rma_deg_rate.value,
                        dox_model_config=dox_model_config,
                        dox_kd=dox_kd.value,
                        tta_prod_rate=tta_prod_rate.value,
                        tta_deg_rate=tta_deg_rate.value,
                        tta_kd=tta_kd.value,
                        cno_model_config=cno_model_config,
                        cno_t0=cno_t0.value,
                        cno_ec50=cno_ec50.value,
                        clz_ec50=clz_ec50.value,
                        dq_prod_rate=hm3dq_prod_rate.value,
                        dq_deg_rate=hm3dq_deg_rate.value,
                        dq_ec50=hm3dq_ec50.value,
                        leaky_rma_prod_rate=leaky_rma_prod_rate.value,
                        leaky_tta_prod_rate=leaky_tta_prod_rate.value,
                        tta_coop=tta_coop.value,
                        cno_coop=cno_coop.value,
                        clz_coop=clz_coop.value,
                        dq_coop=hm3dq_coop.value,
                        time_units=time_map[time_units.value],
                        conc_units=conc_map[concentration_units.value]
                    )
                    y0=(
                        init_brain_rma.value, init_plasma_rma.value, init_inducible_tta.value,
                        init_brain_dox.value, init_plasma_dox.value, init_hm3dq.value,
                        init_peritoneal_cno.value, init_brain_cno.value, init_plasma_cno.value,
                        init_brain_clz.value, init_plasma_clz.value
                    )

            solution = model.simulate(
                t0=start_time.value,
                t1=stop_time.value,
                dt0=dt0.value,
                y0=y0,
                sampling_rate=sampling_rate.value
            )

        return solution, species

    return (run_simulation,)


@app.cell
def _(mo, model_selection):
    # app view
    run_button = mo.ui.run_button(
        kind="success",
        label="Run Simulation",
    )

    mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("# RMA Kinetics Simulator"),
                mo.md('''
                From "Modeling synthetic serum marker dynamics for monitoring deep tissue gene expression."
                '''),
            ]),
            mo.nav_menu(
                {
                    "https://szablowskilab.org": f"{mo.icon("lucide:brain")} Szablowski Lab",
                    "https://github.com/szablowskilab/rma-kinetics": f"{mo.icon("lucide:github")} GitHub",
                    "https://szablowskilab.github.io/rma-kinetics/docs": f"{mo.icon('lucide:info')} Help",
                }
            )
        ]),
        mo.vstack([
            mo.md("## Select model"),
            model_selection
        ]),
        run_button
    ], gap=2)
    return (run_button,)


@app.cell
def _(mo, model_selection, run_button, run_simulation):
    mo.stop(not run_button.value)
    solution, species = run_simulation(model_selection)
    species_selector = mo.ui.dropdown(
        options=species,
        value="Plasma RMA",
    )
    return solution, species_selector


@app.cell
def _(mo, plt, sb, solution, species_selector):
    fig = None
    if solution is not None:
        solution._plot_species(species_selector.value)

        sb.despine()
        fig = plt.gcf()

    mo.vstack([
        mo.mpl.interactive(fig),
        species_selector,
        #mo.md("If you found this tool useful, consider citing [Buitrago et al., 2025]()")
    ])

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
