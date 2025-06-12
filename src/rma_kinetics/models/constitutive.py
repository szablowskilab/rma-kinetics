from .abstract import AbstractModel
from diffrax import ODETerm, AbstractTerm
from jaxtyping import PyTree


class ConstitutiveRMA(AbstractModel):
    """
    Model of constitutive RMA production.

    Attributes:
        rma_prod_rate (float): RMA production rate (concentration/time).
        rma_rt_rate (float): RMA reverse transcytosis rate (1/time).
        rma_deg_rate (float): RMA degradation rate (1/time).
    """

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        """
        ODE model implementation.
        """
        brain_rma, plasma_rma = y
        plasma_rma_transport_flux = self.rma_rt_rate * brain_rma
        dbrain_rma = self.rma_prod_rate - plasma_rma_transport_flux
        dplasma_rma = plasma_rma_transport_flux - (self.rma_deg_rate * plasma_rma)

        return dbrain_rma, dplasma_rma

    def _terms(self) -> AbstractTerm:
        return ODETerm(self._model)
