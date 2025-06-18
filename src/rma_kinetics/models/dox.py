from dataclasses import field
from equinox import Module as EqxModule
from jaxtyping import PyTree
from jax import numpy as jnp
from jax.lax import cond as jax_cond


DOX_MW = 444.4 # g/mol

class DoxPKConfig(EqxModule):
    """
    Dox PK model configuration.

    Attributes:
        vehicle_intake_rate (float): Dox chow/water intake rate.
        bioavailability (float): Dox bioavailability as a float between 0 and 1.
        vehicle_dose (float): Dox amount in chow/water (i.e., mg dox / kg chow).
        absorption_rate (float): Dox absorption rate into the plasma.
        elimination_rate (float): Elimination rate from plasma.
        brain_transport_rate (float): Plasma to brain transport rate.
        plasma_transport_rate (float): Brain to plasma transport rate.
        t0 (float): Start time of dox administration.
        t1 (float): Stop time of dox administration.
        plasma_vd (float): Plasma dox volume of distribution.
    """
    vehicle_intake_rate: float
    bioavailability: float
    vehicle_dose: float
    absorption_rate: float
    elimination_rate: float
    brain_transport_rate: float
    plasma_transport_rate: float
    t0: float
    t1: float
    plasma_vd: float
    intake_rate: float = field(init=False)

    def __post_init__(self):
        self.intake_rate = self.vehicle_intake_rate * self.bioavailability * self.vehicle_dose / (DOX_MW * self.plasma_vd)* 1e6


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
            jnp.logical_and(t > self.t0, t < self.t1),
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
