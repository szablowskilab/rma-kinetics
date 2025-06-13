from .abstract import AbstractModel
from .dox import DoxPK, DoxPKConfig
from jaxtyping import PyTree
from diffrax import ODETerm


class TetRMA(AbstractModel):
    dox: DoxPK
    dox_kd: float
    tta_prod_rate: float
    tta_deg_rate: float
    tta_kd: float
    leaky_rma_prod_rate: float
    tta_coop: int

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
        leaky_rma_prod_rate: float = 0.0,
        tta_coop: int = 2,
    ):

        super().__init__(rma_prod_rate, rma_rt_rate, rma_deg_rate)

        self.dox = DoxPK(dox_model_config)
        self.dox_kd = dox_kd

        self.tta_prod_rate = tta_prod_rate
        self.tta_deg_rate = tta_deg_rate
        self.tta_kd = tta_kd
        self.leaky_rma_prod_rate = leaky_rma_prod_rate
        self.tta_coop = tta_coop

    def _tet_rma_model(self, t: float, y: PyTree[float]) -> PyTree[float]:
        """
        Tet induced RMA expression.

        :param t: time (float
        :param y: tuple containing concentrations of plasma/brain RMA, transcriptional activator, plasma/brain dox.
        :return:
        """
        # brain and plasma dox are given as amounts here.
        brain_rma, plasma_rma, ta, brain_dox, plasma_dox = y

        dplasma_dox, dbrain_dox = self.dox._model(t, (plasma_dox, brain_dox))

        active_ta_frac = 1 / (1 + (brain_dox/self.dox_kd))
        ta_modulator = (active_ta_frac * ta / self.tta_kd)**self.tta_coop
        brain_rma_outflux = self.rma_rt_rate * brain_rma

        dbrain_rma = ((self.leaky_rma_prod_rate + (self.rma_prod_rate * ta_modulator)) / (1 + ta_modulator)) - brain_rma_outflux
        dplasma_rma = brain_rma_outflux - (self.rma_deg_rate * plasma_rma)

        return dbrain_rma, dplasma_rma, dbrain_dox, dplasma_dox

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        brain_rma, plasma_rma, ta, brain_dox, plasma_dox = y
        dbrain_rma, dplasma_rma, dbrain_dox, dplasma_dox = self._tet_rma_model(t, (brain_rma, plasma_rma, ta, brain_dox, plasma_dox))

        dta = self.tta_prod_rate - (self.tta_deg_rate * ta) # constitutive TA expression

        return dbrain_rma, dplasma_rma, dta, dbrain_dox, dplasma_dox

    def _terms(self):
        return ODETerm(self._model)
