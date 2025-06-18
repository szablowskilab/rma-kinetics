from .abstract import AbstractModel
from ..units import Time, Concentration
from diffrax import ODETerm, AbstractTerm
from jaxtyping import PyTree


class ConstitutiveRMA(AbstractModel):
    """
    Model of constitutive RMA production.

    Attributes:
        rma_prod_rate (float): RMA production rate (concentration/time).
        rma_rt_rate (float): RMA reverse transcytosis rate (1/time).
        rma_deg_rate (float): RMA degradation rate (1/time).
        time_units: (Time): time units (Default = Time.hours).
        conc_units: (Concentration): concentration units (Default = Concentration.nanomolar).
    """

    def __init__(
        self,
        rma_prod_rate: float,
        rma_rt_rate: float,
        rma_deg_rate: float,
        time_units: Time = Time.hours,
        conc_units: Concentration = Concentration.nanomolar
    ):
        super().__init__(
            rma_prod_rate,
            rma_rt_rate,
            rma_deg_rate,
            time_units=time_units,
            conc_units=conc_units
        )

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        """
        ODE model implementation.

        Arguments:
            t (float): Time point.
            y (PyTree[float]): Brain and plasma RMA concentrations.

        Returns:
            dydt (PyTree[float]): Change in brain and plasma RMA concentrations.
        """
        brain_rma, plasma_rma = y
        plasma_rma_transport_flux = self.rma_rt_rate * brain_rma
        dbrain_rma = self.rma_prod_rate - plasma_rma_transport_flux
        dplasma_rma = plasma_rma_transport_flux - (self.rma_deg_rate * plasma_rma)

        return dbrain_rma, dplasma_rma

    def _terms(self) -> AbstractTerm:
        return ODETerm(self._model)
