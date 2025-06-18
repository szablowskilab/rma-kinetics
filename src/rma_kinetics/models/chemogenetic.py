from .cno import CnoPK, CnoPKConfig
from .dox import DoxPKConfig
from .tet_induced import TetRMA
from .abstract import Solution

from diffrax import ClipStepSizeController, SaveAt, Kvaerno3, AbstractSolver, AbstractAdjoint, RecursiveCheckpointAdjoint, diffeqsolve, AbstractStepSizeController, ConstantStepSize
from jaxtyping import PyTree
from jax import numpy as jnp
from jax.lax import cond as jcond
#import jax


class ChemogeneticRMA(TetRMA):
    """
    Chemogenetic activated RMA expression model.

    Attributes:
        rma_prod_rate (`float`): RMA production rate (concentration/time).
        rma_rt_rate (`float`): RMA reverse transcytosis rate (1/time).
        rma_deg_rate (`float`): RMA degradation rate (1/time).
        dox_model_config (`DoxPKConfig`): Dox PK model configuration.
        dox_kd (`float`): Dox dissocation constant.
        tta_prod_rate (`float`): tTA production rate.
        tta_deg_rate (`float`): tTA degradation rate.
        tta_kd (`float`): tTA-TetO operator dissocation constant.
        cno_model_config (`CnoPKConfig`): CNO PK model configuration.
        cno_t0 (`float`): CNO administration time.
        cno_ec50 (`float`): CNO EC50.
        clz_ec50 (`float`): CLZ EC50.
        dq_prod_rate (`float`): hM3Dq production rate.
        dq_deg_rate (`float`): hM3Dq degradation rate.
        dq_ec50 (`float`): hM3Dq EC50.
        leaky_rma_prod_rate (`float`): Leaky RMA production rate (Default = `0.0`).
        leaky_tta_prod_rate (`float`): Leaky tTA production rate (Default = `0.0`).
        tta_coop (`int`): tTA cooperativity (Default = `2`).
        cno_coop (`int`): CNO cooperativity (Default = `1`).
        clz_coop (`int`): CLZ cooperativity (Default = `1`).
        dq_coop (`int`): hM3Dq cooperativity (Default = `1`).
        time_units (`Time`): Time units (Default = `Time.hours`).
        conc_units (`Concentration`): Concentration units (Default = `Concentration.nanomolar`).
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

    def _cno_inj_control(self, t, y, args=None):
        control = jnp.zeros_like(y)
        return control.at[6].set(1.0)

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:

        brain_rma, plasma_rma, ta, brain_dox, plasma_dox, dreadd, peritoneal_cno, brain_cno, plasma_cno, brain_clz, plasma_clz = y

        # CNO injection
        # peritoneal_cno = jcond(
        #     t == self.cno_t0,
        #     lambda: peritoneal_cno + self.cno.cno_nmol,
        #     lambda: peritoneal_cno
        # )

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
