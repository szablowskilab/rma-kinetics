from equinox import Module as EqxModule
from jaxtyping import PyTree
from jax import numpy as jnp
from jax.lax import cond as jax_cond


DOX_MW = 444.4 # g/mol

class DoxPKConfig(EqxModule):
    """
    Dox PK model configuration.

    Attributes:
        dose (float): Dox amount in chow/water (i.e., mg dox / kg chow).
        t0 (float): Start time of dox administration.
        t1 (float): Stop time of dox administration.
        vehicle_intake_rate (float): Dox chow/water intake rate (Default: 1.875e-4).
        bioavailability (float): Dox bioavailability as a float between 0 and 1 (Default = 0.9).
        absorption_rate (float): Dox absorption rate into the plasma (Default = 0.8).
        elimination_rate (float): Elimination rate from plasma (Default = 0.2).
        brain_transport_rate (float): Plasma to brain transport rate (Default = 0.2).
        plasma_transport_rate (float): Brain to plasma transport rate (Default = 1).
        plasma_vd (float): Plasma dox volume of distribution (Default = 0.21).
    """
    dose: float
    t0: float
    t1: float
    vehicle_intake_rate: float = 1.875e-4
    bioavailability: float = 0.9
    absorption_rate: float = 0.8
    elimination_rate: float = 0.2
    brain_transport_rate: float = 0.2
    plasma_transport_rate: float = 1
    plasma_vd: float = 0.21

    @property
    def intake_rate(self) -> float:
        return self.vehicle_intake_rate * self.bioavailability * self.dose / (DOX_MW * self.plasma_vd) * 1e6

    @property
    def plasma_dox_ss(self) -> float:
        return self.absorption_rate * self.intake_rate / self.elimination_rate

    @property
    def brain_dox_ss(self) -> float:
        return self.brain_transport_rate * self.plasma_dox_ss / self.plasma_transport_rate


class DoxPK(EqxModule):
    """
    Dox PK model for tet-induced and chemogenetic RMA models.

    Attributes:
        config (DoxPKConfig): Model configuration.
    """
    config: DoxPKConfig

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)

        raise AttributeError(name)


    def _intake(self, t: float) -> float:
        """
        Time dependent dox intake.

        Arguments:
            t (float): Time point.

        Returns:
            intake_rate (float): Time-dependent dox intake rate.
        """

        return jax_cond(
            jnp.logical_and(t >= self.t0, t < self.t1),
            lambda: self.intake_rate,
            lambda: 0.0
        )

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        """
        Two compartment dox kinetic model.

        Arguments:
            t (float): time point.
            y (PyTree[float]): Plasma and brain dox amounts.

        Returns:
            dydt (PyTree[float]): Change in plasma and brain dox concentrations.
        """

        plasma_dox, brain_dox = y
        plasma_outflux = self.brain_transport_rate * plasma_dox
        brain_outflux = self.plasma_transport_rate * brain_dox
        dplasma_dox = (self.absorption_rate * self._intake(t)) - (self.elimination_rate * plasma_dox) - plasma_outflux + brain_outflux
        dbrain_dox =  plasma_outflux - brain_outflux

        return dplasma_dox, dbrain_dox
