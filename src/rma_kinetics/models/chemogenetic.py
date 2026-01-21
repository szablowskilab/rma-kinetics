from .cno import CnoPK, CnoPKConfig
from .dox import DoxPKConfig
from .tet_induced import TetRMA
from ..units import Time, Concentration
from .abstract import Solution

from equinox import field as eqx_field

from jaxtyping import PyTree
from jax import numpy as jnp
from diffrax import (
    AbstractPath,
    AbstractProgressMeter,
    NoProgressMeter,
    diffeqsolve,
    SaveAt,
    AbstractStepSizeController,
    PIDController,
    Kvaerno3,
    AbstractSolver,
    RecursiveCheckpointAdjoint,
    AbstractAdjoint,
    Solution as DiffSol,
    AbstractPath
)


class ChemogeneticRMA(TetRMA):
    """
    Chemogenetic activated RMA expression model.

    Attributes:
        rma_prod_rate (float): RMA production rate (concentration/time).
        rma_rt_rate (float): RMA reverse transcytosis rate (1/time).
        rma_deg_rate (float): RMA degradation rate (1/time).
        dox_model_config (DoxPKConfig): Dox PK model configuration.
        dox_kd (float): Dox dissocation constant.
        tta_prod_rate (float): tTA production rate.
        tta_deg_rate (float): tTA degradation rate.
        tta_kd (float): tTA-TetO operator dissocation constant.
        cno_model_config (CnoPKConfig): CNO PK model configuration.
        cno_t0 (float): CNO administration time.
        cno_ec50 (float): CNO EC50.
        clz_ec50 (float): CLZ EC50.
        dq_prod_rate (float): hM3Dq production rate.
        dq_deg_rate (float): hM3Dq degradation rate.
        dq_ec50 (float): hM3Dq EC50.
        leaky_rma_prod_rate (float): Leaky RMA production rate (Default = 0.0).
        leaky_tta_prod_rate (float): Leaky tTA production rate (Default = 0.0).
        tta_coop (int): tTA cooperativity (Default = 2).
        cno_coop (int): CNO cooperativity (Default = 1).
        clz_coop (int): CLZ cooperativity (Default = 1).
        dq_coop (int): hM3Dq cooperativity (Default = 1).
        time_units (Time): Time units (Default = Time.hours).
        conc_units (Concentration): Concentration units (Default = Concentration.nanomolar).
    """
    cno: CnoPK
    cno_t0: float = eqx_field(static=True)
    cno_ec50: float
    clz_ec50: float
    cno_coop: int
    clz_coop: int
    dq_prod_rate: float
    dq_deg_rate: float
    dq_ec50: float
    dq_coop: int
    leaky_tta_prod_rate: float

    def __init__(
        self,
        rma_prod_rate: float,
        rma_rt_rate: float,
        rma_deg_rate: float,
        dox_model_config: DoxPKConfig,
        dox_kd: float,
        tta_prod_rate: float,
        tta_deg_rate: float,
        tta_kd: float,
        cno_model_config: CnoPKConfig,
        cno_t0: float,
        cno_ec50: float,
        clz_ec50: float,
        dq_prod_rate: float,
        dq_deg_rate: float,
        dq_ec50: float,
        leaky_rma_prod_rate: float = 0.0,
        leaky_tta_prod_rate: float = 0.0,
        tta_coop: int = 2,
        cno_coop: int = 1,
        clz_coop: int = 1,
        dq_coop: int = 1,
        time_units: Time = Time.hours,
        conc_units: Concentration = Concentration.nanomolar
    ):
        super().__init__(
            rma_prod_rate,
            rma_rt_rate,
            rma_deg_rate,
            dox_model_config,
            dox_kd,
            tta_prod_rate,
            tta_deg_rate,
            tta_kd,
            leaky_rma_prod_rate,
            tta_coop,
            time_units,
            conc_units
        )

        self.cno = CnoPK(cno_model_config)
        self.cno_t0 = cno_t0
        self.cno_ec50 = cno_ec50
        self.clz_ec50 = clz_ec50
        self.cno_coop = cno_coop
        self.clz_coop = clz_coop
        self.dq_prod_rate = dq_prod_rate
        self.dq_deg_rate = dq_deg_rate
        self.dq_ec50 = dq_ec50
        self.dq_coop = dq_coop
        self.leaky_tta_prod_rate = leaky_tta_prod_rate

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        r"""
        Full ODE model implementation.

        Info: Model Equations
            Note that this model assumes constitutive expression of hM3Dq.

            $$\begin{align}
            \dot{[Dq]} &= k_{Dq} - \gamma_{Dq}[Dq] \tag{1} \\
            [Dq]_{SS} &= \frac{k_{Dq}}{\gamma_{Dq}} \tag{2}
            \end{align}
            $$

            The designer receptir can be activated by CNO (or CLZ which is
            produced from reverse-metabolism of CNO).

            $$\begin{align}
            \theta_{L} &= \frac{\frac{[CNO]}{EC_{50_{CNO}}} + \frac{[CLZ]}{EC_{50_{CLZ}}}}{1 + \frac{[CNO]}{EC_{50_{CNO}}} + \frac{[CLZ]}{EC_{50_{CLZ}}}} \tag{3}
            \end{align}
            $$

            Production of the tetracycline-transcriptional activator (tTA) is then
            dependent on the level of active hM3Dq and modified for leaky expression,

            $$\begin{align}
            \alpha_{Dq} &= \left(\frac{\theta_{L}[Dq]}{EC_{50_{Dq}}}\right)^{n_{Dq}} \tag{4} \\
            \dot{[tTA]} &= \frac{k_{0_{tTA}} + k_{tTA}\alpha_{Dq}}{1 + \alpha_{Dq}} - \gamma_{TA}[TA] \tag{5}
            \end{align}
            $$

            The remaining equations for doxycycline and RMA dynamics are the same
            as the TetRMA model.

            |Parameters|Description|Units (Example)|
            |----------|-----------|-----|
            |$k_{Dq}$|hM3Dq production rate|Concentration/Time (nM/hr)|
            |$\gamma_{Dq}$|hM3Dq degradation rate|1/Time (1/hr)|
            |$EC_{50_{CNO}}$|CNO EC50|Concentration (nM)|
            |$EC_{50_{CLZ}}$|CLZ EC50|Concentration (nM)|
            |$EC_{50_{Dq}}$|hM3Dq EC50|Concentration (nM)|
            |$n_{Dq}$|hM3Dq cooperativity|Unitless|
            |$k_{tTA}$|tTA production rate|Concentration/Time (nM/hr)|
            |$\gamma_{tTA}$|tTA degradation rate|1/Time (1/hr)|
            |$[CNO]$|Brain CNO concentration|Concentration (nM)|
            |$[CLZ]$|Brain CLZ concentration|Concentration (nM)|
            |$[Dq]$|Brain hM3Dq concentration|Concentration (nM)|
            |$[tTA]$|Brain tTA concentration|Concentration (nM)|



        Args:
            t (float): Time point.
            y (PyTree[float]): Concentration of brain/plasma RMA (along with all other species).

        Returns:
            dydt (PyTree[float]): Change in brain/plasma RMA (along with all other species).
        """

        brain_rma, plasma_rma, ta, brain_dox, plasma_dox, dreadd, peritoneal_cno, brain_cno, plasma_cno, brain_clz, plasma_clz = y

        dbrain_rma, dplasma_rma, dbrain_dox, dplasma_dox = self._tet_rma_model(t, (brain_rma, plasma_rma, ta, brain_dox, plasma_dox))

        dperitoneal_cno, dplasma_cno, dbrain_cno, dplasma_clz, dbrain_clz = self.cno._model(
            t,
            (peritoneal_cno, plasma_cno, brain_cno, plasma_clz, brain_clz)
        )

        # CNO+CLZ/DREADD induced TA expression
        cno_ec50_hill = (brain_cno / self.cno.cno_brain_vd / self.cno_ec50)**self.cno_coop
        clz_ec50_hill = (brain_clz / self.cno.clz_brain_vd / self.clz_ec50)**self.clz_coop
        active_dreadd_frac = (cno_ec50_hill + clz_ec50_hill) / (1 + cno_ec50_hill + clz_ec50_hill)
        dreadd_modulator = (active_dreadd_frac * dreadd / self.dq_ec50)**self.dq_coop
        dta = ((self.leaky_tta_prod_rate + (self.tta_prod_rate * dreadd_modulator)) / (1 + dreadd_modulator)) - (self.tta_deg_rate * ta)

        ddreadd = self.dq_prod_rate - (self.dq_deg_rate * dreadd) # constitutive DREADD expression

        return (
            dbrain_rma,
            dplasma_rma,
            dta,
            dbrain_dox,
            dplasma_dox,
            ddreadd,
            dperitoneal_cno,
            dbrain_cno,
            dplasma_cno,
            dbrain_clz,
            dplasma_clz
        )

    def simulate(
            self,
            t0: float,
            t1: float,
            y0: PyTree[float],
            dt0: float | None=None,
            sampling_rate: float=1,
            stepsize_controller: AbstractStepSizeController = PIDController(rtol=1e-5, atol=1e-5),
            max_steps: int = 4096,
            solver: AbstractSolver = Kvaerno3(),
            adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
            throw: bool = True,
            progress_meter: AbstractProgressMeter = NoProgressMeter()
    ):
        """
        Simulate chemogenetic activated RMA.

        This method differs from other models by splitting the integration
        in up to two parts if needed depending on the time of CNO administration.

        Args:
            t0 (float): Start time of integration.
            t1 (float): Stop time of integration.
            y0 (PyTree[float]): Tuple of initial conditions.
            dt0 (float | None`): Initial step size if using adaptive
                step sizes, or size of all steps if using constant stepsize.
            sampling_rate (float): Sampling rate for saving solution.
            stepsize_controller (AbstractStepSizeController`): Determines
                how to change step size during integration.
            max_steps (int): Max number of steps before stopping.
            solver (AbstractSolver): Differential equation solver.
            adjoint (AbstractAdjoint): How to differentiate.
            throw (bool): Raise an exception if integration fails.
            progress_meter (AbstractProgressMeter): Progress meter.

        Returns:
            solution (Solution): A solution object (parent of diffrax.Solution) with added plotting methods.
        """
        if self.cno_t0 == 0:
            # adminster CNO at time 0 and run until t1
            y0_0 = list(y0)
            y0_0[6] += self.cno.cno_nmol
            y0_0 = tuple(y0_0)
            t1_0 = t1
        else:
            y0_0 = y0
            t1_0 = self.cno_t0

        # simulate first segment pre CNO
        saveat_0 = SaveAt(ts=jnp.linspace(t0, t1_0, int((t1_0 - t0)*sampling_rate)))
        solution_0 = diffeqsolve(
            self._terms(),
            solver,
            t0,
            t1_0,
            dt0,
            y0_0,
            saveat=saveat_0,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            progress_meter=progress_meter
        )

        if solution_0.ys is None or solution_0.ts is None:
            raise ValueError("Solution or saved time arrays are empty, but the solver didn't throw an error!")

        # if CNO is adminstered at time 0, we return early with the solution.
        elif self.cno_t0 == 0:
            # convert plasma/brain CNO/CLZ to concentrations
            _ys = list(solution_0.ys)
            _ys[7] /= self.cno.cno_brain_vd
            _ys[8] /= self.cno.cno_plasma_vd
            _ys[9] /= self.cno.clz_brain_vd
            _ys[10] /= self.cno.clz_brain_vd
            solution_0 = DiffSol(
                t0=self.cno_t0,
                t1=t1,
                ts=solution_0.ts,
                ys=tuple(_ys),
                interpolation=solution_0.interpolation,
                stats=solution_0.stats,
                result=solution_0.result,
                solver_state=solution_0.solver_state,
                controller_state=solution_0.controller_state,
                made_jump=solution_0.made_jump,
                event_mask=solution_0.event_mask
            )
            return Solution(solution_0, self.time_units, self.conc_units)

        saveat_1 = SaveAt(ts=jnp.linspace(self.cno_t0, t1, int((t1-self.cno_t0)*sampling_rate)))

        # adjust initial condition for peritoneal CNO based on injection amount
        y0_1 = [y[-1] for y in solution_0.ys]
        y0_1[6] += self.cno.cno_nmol

        # simulate second segment after administering CNO
        solution_1 = diffeqsolve(
            self._terms(),
            solver,
            self.cno_t0,
            t1,
            dt0,
            tuple(y0_1),
            saveat=saveat_1,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            progress_meter=progress_meter
        )

        if solution_1.ys is None or solution_1.ts is None:
            raise ValueError("Solution or saved time arrays are empty, but the solver didn't throw an error!")

        ts_full = jnp.concatenate([solution_0.ts[:-1], solution_1.ts])
        ys_full = [jnp.concatenate([ys_0[:-1], ys_1]) for ys_0, ys_1 in zip(solution_0.ys, solution_1.ys)]
        # convert plasma/brain CNO/CLZ to concentrations
        ys_full[7] /= self.cno.cno_brain_vd
        ys_full[8] /= self.cno.cno_plasma_vd
        ys_full[9] /= self.cno.clz_brain_vd
        ys_full[10] /= self.cno.clz_plasma_vd

        diffsol = DiffSol(
            t0=t0,
            t1=t1,
            ts=ts_full,
            ys=tuple(ys_full),
            interpolation=solution_1.interpolation,
            stats=solution_1.stats,
            result=solution_1.result,
            solver_state=solution_1.solver_state,
            controller_state=solution_1.controller_state,
            made_jump=solution_1.made_jump,
            event_mask=solution_1.event_mask
        )

        return Solution(diffsol, self.time_units, self.conc_units)
