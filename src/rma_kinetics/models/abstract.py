from abc import abstractmethod
from equinox import Module as EqxModule
from jax import numpy as jnp
from ..units import Time, Concentration
from diffrax import (
    AbstractProgressMeter,
    NoProgressMeter,
    Solution as DiffSol,
    diffeqsolve,
    SaveAt,
    AbstractStepSizeController,
    PIDController,
    Kvaerno3,
    AbstractSolver,
    RecursiveCheckpointAdjoint,
    AbstractAdjoint,
    AbstractTerm,
)
from jaxtyping import PyTree
import matplotlib.pyplot as plt


SPECIES_MAP = {
    "Brain RMA": 0,
    "Plasma RMA": 1,
    "tTA": 2,
    # for brain species
    # (ignoring plasma concentrations of these, but still accessible in Solution._diffsol.ys)
    "Dox": 3,
    "hM3Dq": 5,
    "CNO": 7,
    "CLZ": 9
}

class Solution(EqxModule):
    """
    Solution returned from simulation.

    Attributes:
        _diffsol (diffrax.Solution): Diffrax solution to wrap (returned by `diffeqsolve`).
        time_units (Time): Time enum used to format axis in plots.
        conc_units (Concentration): Concentration enum used to format axis in plots.
    """
    _diffsol: DiffSol
    time_units: Time
    conc_units: Concentration

    def __getattr__(self, name):
        return getattr(self._diffsol, name)

    @property
    def brain_rma(self):
        """Get Brain RMA trajectory as a jax array."""
        return self._get_species("Brain RMA")

    @property
    def plasma_rma(self):
        """Get Plasma RMA trajectory as a jax array."""
        return self._get_species("Plasma RMA")

    @property
    def tta(self):
        """
        Get tTA trajectory as a jax array.

        Available in solutions from `TetRMA` or `ChemogeneticRMA`.
        """
        return self._get_species("tTA")

    @property
    def dox(self):
        """
        Get dox trajectory as a jax array.

        Available in solutions from `TetRMA` or `ChemogeneticRMA`.
        """
        return self._get_species("Dox")

    @property
    def hm3dq(self):
        """
        Get hM3Dq trajectory as a jax array.

        Available in solutions from `ChemogeneticRMA`.
        """
        return self._get_species("hM3Dq")

    @property
    def cno(self):
        """
        Get CNO trajectory as a jax array.

        Available in solutions from `ChemogeneticRMA`.
        """
        return self._get_species("CNO")

    @property
    def clz(self):
        """
        Get CLZ trajectory as a jax array.

        Available in solutions from `ChemogeneticRMA`.
        """
        return self._get_species("CLZ")

    def plot_plasma_rma(self):
        """Plot plasma RMA simulation."""
        self._plot_species("Plasma RMA")

    def plot_brain_rma(self):
        """Plot brain RMA simulation."""
        self._plot_species("Brain RMA")

    def plot_tta(self):
        """
        Plot tTA simulation.

        Available in solutions from `TetRMA` or `ChemogeneticRMA`.
        """
        self._plot_species("tTA")

    def plot_dox(self):
        """
        Plot dox simulation.

        Available in solutions from `TetRMA` or `ChemogeneticRMA`.
        """
        self._plot_species("Dox")

    def plot_hm3dq(self):
        """
        Plot hM3Dq simulation.

        Available in solutions from `ChemogeneticRMA`.
        """

    def plot_cno(self):
        """
        Plot CNO simulation.

        Available in solutions from `ChemogeneticRMA`.
        """
        self._plot_species("CNO")

    def plot_clz(self):
        """
        Plot CLZ simulation.

        Available in solutions from `ChemogeneticRMA`.
        """
        self._plot_species("CLZ")

    def _get_species(self, label: str):
        idx = SPECIES_MAP[label]
        if self._diffsol.ys is not None and len(self._diffsol.ys) >= idx:
            return self._diffsol.ys[idx]
        else:
            raise ValueError("Solution is empty")

    def _plot_species(self, label: str, vd: float | None = None):
        spc_idx = SPECIES_MAP[label]
        if self._diffsol.ys is not None and len(self._diffsol.ys) >= spc_idx:
            plt.plot(self._diffsol.ts, self._diffsol.ys[spc_idx], 'k')
            plt.xlabel(f"Time ({Time[self.time_units]})")
            plt.ylabel(f"{label} ({Concentration[self.conc_units]})")
            plt.tight_layout()
        else:
            raise ValueError("Solution is empty")


class AbstractModel(EqxModule):
    """
    Abstract RMA model.

    Attributes:
        rma_prod_rate (float): RMA production rate (concentration/time).
        rma_rt_rate (float): RMA reverse transcytosis rate (1/time).
        rma_deg_rate (float): RMA degradation rate (1/time).
        time_units (Time): Time units (Default = Time.hours).
        conc_units (Concentration): Concentration units (Default = Concentration.nanomolar).
    """
    rma_prod_rate: float
    rma_rt_rate: float
    rma_deg_rate: float
    time_units: Time
    conc_units: Concentration

    def __init__(
        self,
        rma_prod_rate: float,
        rma_rt_rate: float,
        rma_deg_rate: float,
        time_units: Time = Time.hours,
        conc_units: Concentration = Concentration.nanomolar,
    ):
        self.rma_prod_rate = rma_prod_rate
        self.rma_rt_rate = rma_rt_rate
        self.rma_deg_rate = rma_deg_rate
        self.time_units = time_units
        self.conc_units = conc_units

    @abstractmethod
    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        """
        Final ODE/SDE model (implemented by child class)

        Args:
            t (float): Time point.
            y (PyTree[float]): Brain and plasma RMA concentrations

        Returns:
            dydt (PyTree[float]): Change in brain and plasma RMA concentrations
                (along with any other additional species).
        """
        pass

    @abstractmethod
    def _terms(self) -> AbstractTerm:
        """
        Wraps model in `AbstractTerm` for use with the differential
        equation solver (implemented by child class).

        Returns:
            term (AbstractTerm): Terms for use with `diffrax.diffeqsolve`.
        """
        pass

    def simulate(
            self,
            t0: float,
            t1: float,
            y0: PyTree[float],
            dt0: float | None = None,
            sampling_rate: float = 1,
            stepsize_controller: AbstractStepSizeController = PIDController(rtol=1e-5, atol=1e-5),
            max_steps: int = 4096,
            solver: AbstractSolver = Kvaerno3(),
            adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
            throw: bool = True,
            progress_meter: AbstractProgressMeter = NoProgressMeter()
    ):
        """
        Simulates model within the given time interval.

        Wraps `diffrax.diffeqsolve` with specific defaults for RMA model simulation.

        Arguments:
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

        Returns:
            solution (Solution): A solution object (parent of diffrax.Solution) with added plotting methods.
        """
        saveat = SaveAt(ts=jnp.linspace(t0, t1, int(t1*sampling_rate)))
        diffsol = diffeqsolve(
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
            throw=throw,
            progress_meter=progress_meter
        )

        return Solution(diffsol, self.time_units, self.conc_units)
