from .cno import CnoPK, CnoPKConfig
from .dox import DoxPKConfig
from .tet_induced import TetRMA
from ..units import Time, Concentration
from .abstract import Solution

from jaxtyping import PyTree
from jax import numpy as jnp
from diffrax import (
    diffeqsolve,
    SaveAt,
    AbstractStepSizeController,
    PIDController,
    Kvaerno3,
    AbstractSolver,
    RecursiveCheckpointAdjoint,
    AbstractAdjoint,
    Solution as DiffSol
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
    cno_t0: float
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

        brain_rma, plasma_rma, ta, brain_dox, plasma_dox, dreadd, peritoneal_cno, brain_cno, plasma_cno, brain_clz, plasma_clz = y

        dbrain_rma, dplasma_rma, dbrain_dox, dplasma_dox = self._tet_rma_model(t, (brain_rma, plasma_rma, ta, brain_dox, plasma_dox))

        dperitoneal_cno, dplasma_cno, dbrain_cno, dplasma_clz, dbrain_clz = self.cno._model(
            t,
            (peritoneal_cno, plasma_cno, brain_cno, plasma_clz, brain_clz)
        )

        # CNO+CLZ/DREADD induced TA expression
        cno_ec50_hill = (brain_cno / self.cno.cno_brain_vd)**self.cno_coop
        clz_ec50_hill = (brain_clz / self.cno.clz_brain_vd)**self.clz_coop
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
            dt0: float | None=None,
            sampling_rate: float=1,
            y0: PyTree[float]=(0,0),
            stepsize_controller: AbstractStepSizeController = PIDController(rtol=1e-5, atol=1e-5),
            max_steps: int = 4096,
            solver: AbstractSolver = Kvaerno3(),
            adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
            throw: bool = True,
    ):
        """
        """
        # just assume that save at has ts
        if self.cno_t0 == 0:
            saveat = SaveAt(ts=jnp.linspace(t0, t1, int(t1*sampling_rate)))
            solution = diffeqsolve(
                self._terms(),
                solver,
                t0,
                t1,
                dt0,
                y0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
                adjoint=adjoint,
                throw=throw
            )

            return solution

        pre_cno_saveat = SaveAt(ts=jnp.linspace(t0, self.cno_t0, int(self.cno_t0*sampling_rate)))
        pre_cno = diffeqsolve(
            self._terms(),
            solver,
            t0,
            self.cno_t0,
            dt0,
            y0,
            saveat=pre_cno_saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw
        )

        if pre_cno.ys is None or pre_cno.ts is None:
            raise ValueError("Solution or saved time arrays are empty, but the solver didn't throw an error!")

        y1 = [y[-1] for y in pre_cno.ys]
        y1[6] += self.cno.cno_nmol

        post_cno_saveat = SaveAt(ts=jnp.linspace(self.cno_t0, t1, int(t1*sampling_rate)))
        post_cno = diffeqsolve(
            self._terms(),
            solver,
            self.cno_t0,
            t1,
            dt0,
            tuple(y1),
            saveat=post_cno_saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw
        )

        if pre_cno.ys is None or pre_cno.ts is None:
            raise ValueError("Solution or saved time arrays are empty, but the solver didn't throw an error!")

        ts_full = jnp.concatenate([pre_cno.ts[:-1], post_cno.ts])
        ys_full = [jnp.concatenate([pre_ys[:-1], post_ys]) for pre_ys, post_ys in zip(pre_cno.ys, post_cno.ys)]
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
            interpolation=post_cno.interpolation,
            stats=post_cno.stats,
            result=post_cno.result,
            solver_state=post_cno.solver_state,
            controller_state=post_cno.controller_state,
            made_jump=post_cno.made_jump,
            event_mask=post_cno.event_mask
        )

        return Solution(diffsol, self.time_units, self.conc_units)
