from rma_kinetics.models import ChemogeneticRMA, DoxPKConfig, CnoPKConfig
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # add dox feeding at 30 mg/kg food from time 0 to 48 hours
    dox_model_config = DoxPKConfig(
        dose=30,
        t0=0,
        t1=48
    )

    # administer 1mg/kg CNO for 30g mouse at 96 hrs (cno_t0 in model)
    cno_model_config = CnoPKConfig(cno_dose=0.03) # 1mg/kg for 30g mouse

    model = ChemogeneticRMA(
        rma_prod_rate=7e-3,
        rma_rt_rate=0.6,
        rma_deg_rate=7e-3,
        dox_model_config=dox_model_config,
        dox_kd=10,
        tta_prod_rate=8e-3,
        tta_deg_rate=8e-3,
        tta_kd=1,
        cno_model_config=cno_model_config,
        cno_t0=96, # adminster at 96 hrs
        cno_ec50=7,
        clz_ec50=3,
        dq_prod_rate=10,
        dq_deg_rate=1,
        dq_ec50=1
    )

    # simulate 0 - 144 hours
    t0 = 0; t1 = 144
    # we'll assume brain and plasma dox are at steady state
    brain_dox_ss = dox_model_config.brain_dox_ss
    plasma_dox_ss = dox_model_config.plasma_dox_ss
    hm3dq_ss = model.dq_prod_rate / model.dq_deg_rate

    # initial conditions
    y0 = (0, 0, 1, brain_dox_ss, plasma_dox_ss, hm3dq_ss, 0, 0, 0, 0, 0)
    solution = model.simulate(t0, t1, y0)

    # check final RMA concentration and plot
    print(f"Plasma RMA at {t1} hours: {solution.plasma_rma[-1]:.3f} nM")
    solution.plot_plasma_rma()
    plt.show()
