from .cno import CnoPK, CnoPKConfig
from .dox import DoxPKConfig
from .tet_induced import TetRMA

from diffrax import SaveAt, Kvaerno3, AbstractSolver, AbstractAdjoint, RecursiveCheckpointAdjoint, diffeqsolve, AbstractStepSizeController, ConstantStepSize
from jaxtyping import PyTree
from jax import numpy as jnp
import jax
from http.cookiejar import debug

class ChemogeneticRMA(TetRMA):
    """
    Model of chemogenetic activated RMA expression.
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

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:

        brain_rma, plasma_rma, ta, brain_dox, plasma_dox, dreadd, peritoneal_cno, brain_cno, plasma_cno, brain_clz, plasma_clz = y

        # CNO injection
        # peritoneal_cno = cond(
        #     jnp.abs(t - self.cno_t0) < 0.005,
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

    def _simulate(
        self,
        t0: float,
        t1: float,
        dt0: float | None=None,
        y0: PyTree[float]=(0,0),
        saveat: SaveAt = SaveAt(dense=True),
        stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
        max_steps: int = 4096,
        solver: AbstractSolver = Kvaerno3(),
        adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
        throw: bool = True,
    ):

        chunk_1_saveat = SaveAt(ts=jnp.linspace(t0, self.cno_t0, self.cno_t0))
        chunk_1 = diffeqsolve(
            self._terms(),
            solver,
            t0,
            self.cno_t0,
            dt0,
            y0,
            saveat=chunk_1_saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw
        )

        chunk_2_saveat = SaveAt(ts=jnp.linspace(self.cno_t0, t1, t1 - self.cno_t0))
        y0_2 = (
            chunk_1.ys[0][-1],
            chunk_1.ys[1][-1],
            chunk_1.ys[2][-1],
            chunk_1.ys[3][-1],
            chunk_1.ys[4][-1],
            chunk_1.ys[5][-1],
            chunk_1.ys[6][-1] + self.cno.cno_nmol,
            chunk_1.ys[7][-1],
            chunk_1.ys[8][-1],
            chunk_1.ys[9][-1],
            chunk_1.ys[10][-1],
        )

        chunk_2 = diffeqsolve(
            self._terms(),
            solver,
            self.cno_t0,
            t1,
            dt0,
            y0_2,
            saveat=chunk_2_saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw
        )


        full_ts = jnp.concatenate([chunk_1.ts[:-1], chunk_2.ts])
        ys_1_array = jnp.array(list(chunk_1.ys))
        ys_2_array = jnp.array(list(chunk_2.ys))
        #jax.debug.breakpoint()
        full_ys = jnp.concatenate([ys_1_array[:, :-1], ys_2_array], axis=1)

        return {"ts": full_ts, "ys": full_ys}
