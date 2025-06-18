from abc import abstractmethod
from equinox import Module as EqxModule, field
from ..units import Time, Concentration
from diffrax import (
    Solution as DiffSol,
    diffeqsolve,
    SaveAt,
    AbstractStepSizeController,
    ConstantStepSize,
    Kvaerno3,
    AbstractSolver,
    RecursiveCheckpointAdjoint,
    AbstractAdjoint,
    AbstractTerm
)
from jaxtyping import PyTree
import matplotlib.pyplot as plt


class Solution(EqxModule):
    _diffsol: DiffSol
    time_units: Time
    conc_units: Concentration

    def __getattr__(self, name):
        return getattr(self._diffsol, name)

    def plot_plasma_rma(self):
        if self._diffsol.ys is not None:
            plt.plot(self._diffsol.ts, self._diffsol.ys[1], 'k')
            plt.xlabel(f"Time ({Time[self.time_units]})")
            plt.ylabel(f"Plasma RMA ({Concentration[self.conc_units]})")
        else:
            raise ValueError("Solution is empty")

    def _plot_species(self, i: int, label: str):
        if self._diffsol.ys is not None and len(self._diffsol.ys) >= i:
            plt.plot(self._diffsol.ts, self._diffsol.ys[i], 'k')
            plt.xlabel(f"Time ({Time[self.time_units]}")
            plt.ylabel(f"{label} ({Concentration[self.conc_units]})")


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
            dt0: float | None=None,
            y0: PyTree[float]=(0,0),
            saveat: SaveAt = SaveAt(dense=True),
            stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
            max_steps: int = 4096,
            solver: AbstractSolver = Kvaerno3(),
            adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
            throw: bool = True,
    ):
        """
        Wraps `diffrax.diffeqsolve` with specific defaults for RMA model simulation.

        Arguments:
            t0 (float): Start time of integration.
            t1 (float): Stop time of integration.
            dt0 (float | None`): Initial step size if using adaptive
                step sizes, or size of all steps if using constant stepsize
                (Default = None).
            y0 (PyTree[float]): Initial conditions (Default = (0, 0)).
            saveat (SaveAt): Times to save solution (Default = SaveAt(dense=True)).
            stepsize_controller (AbstractStepSizeController`): Determines
                how to change step size during integration (Default = ConstantStepSize()).
            max_steps (int): Max number of steps before stopping (Default = 4096).
            solver (AbstractSolver): Differential equation solver (Default = Kvaerno3()).
            adjoint (AbstractAdjoint): How to differentiate (Default = RecursiveCheckpointAdjoint()).
            throw (bool): Raise an exception if integration fails (Default = True).

        Returns:
            solution (Solution): A solution object (parent of diffrax.Solution) with added plotting methods.
        """
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
            throw=throw
        )

        return Solution(diffsol, self.time_units, self.conc_units)
