from abc import ABC, abstractmethod
from collections.abc import Callable
from collections import deque
import numpy as np
from scipy.integrate import solve_ivp

class Solution:
    def __init__(
        self,
        time: np.ndarray,
        tissue_rma: np.ndarray,
        blood_rma: np.ndarray,
        run_params: dict = {},
    ):
        self.time = time
        self.tissue_rma = tissue_rma
        self.blood_rma = blood_rma
        self.run_params = run_params

    def compute_psd(self):
        pass

    def show_results(self):
        pass


class Model(ABC):
    """
    Abstract base class for RMA kinetic models.
    """

    def __init__(
        self,
        prod_rate: float | Callable[[float], float],
        rt_rate: float,
        deg_rate: float,
    ):
        self.prod_rate = prod_rate
        self.rt_rate = rt_rate
        self.deg_rate = deg_rate

    @abstractmethod
    def run(
        self,
        t_span: tuple[float, float],
        n_steps: int | None = None,
        init_tissue_rma: float = 0.,
        init_blood_rma: float = 0.,
    ) -> Solution:
        pass


class RMA(Model):
    """
    Deterministic kinetic model of RMA expression and transport.

    Parameters
    ----------

    prod_rate : float | Callable[[float], float]
        Tissue RMA production rate constant or time-dependent function.
    rt_rate : float
        Tissue RMA reverse transcytosis rate constant.
    deg_rate : float
        Blood RMA degradation rate constant.
    """

    def _ode_model(self, t: float, y: tuple[float, float]) -> list[float]:
        """
        RMA expression model. Used by the ODE solver.

        Parameters
        ----------

        t : float
            Time to evaluate the model.
        y : tuple[float, float]
            Tuple containing concentrations of each species.

        Returns
        -------

        yi : list[float]
            Change of concentration of each species w.r.t. time
        """
        # if the prod_rate is callable (function like), we pass it t (time).
        # Otherwise, prod_rate is treated as a constant.
        k_exp = self.prod_rate(t) if isinstance(self.prod_rate, Callable) else self.prod_rate

        tissue_rma, blood_rma = y
        dtissue_dt = k_exp - (self.rt_rate * tissue_rma)
        dblood_dt = (self.rt_rate * tissue_rma) - (self.deg_rate * blood_rma)

        return [dtissue_dt, dblood_dt]

    def run(
        self,
        t_span: tuple[float, float],
        n_steps: int | None = None,
        init_tissue_rma: float = 0.,
        init_blood_rma: float = 0.,
    ):
        """
        Run simulation

        Parameters
        ----------
        t_span : tuple[float, float]
            Interval of integration (T0, Tf) starting at T=T0 until T=Tf.
        n_steps : int | None
            Number of equally spaced steps to evaluate during run (default: None).
            If None, evaluate at time steps chosen by the solver.
        init_tissue_rma : float
            Initial tissue RMA concentration (nM).
        init_blood_rma : float
            Initial blood RMA concentration (nM).

        Returns
        -------
            Result of run (OdeResult)
        """

        t_eval = None
        if n_steps:
            start_time, stop_time = t_span
            t_eval = np.linspace(start_time, stop_time, n_steps)

        solution = solve_ivp(
            self._ode_model,
            t_span,
            [init_tissue_rma, init_blood_rma],
            t_eval=t_eval
        )

        return Solution(solution.t, solution.y[0], solution.y[1])


class NoisyRMA_SDE(Model):
    """
    RMA kinetic model with added protein diffusion noise and (optional)
    experimental measurement noise.

    Parameters
    ----------
        k_exp : float | Callable[[float], float]
            Tissue RMA expression rate constant or time-dependent function.
        k_rt : float
            Tissue RMA reverse transcytosis rate constant.
        gamma : float
            Blood RMA degradation rate constant.
        diff_noise : float
            RMA effective diffusion coefficient.
        exp_noise : float
            Experimental error range (0, 1). For example, exp_noise = 0.1 will
           add an additional 10% random noise to the blood RMA time series.
    """
    def __init__(self, prod_rate: float | Callable[[float], float], rt_rate: float, deg_rate: float, tau_dist: Callable[[int], np.ndarray], exp_noise: float = 0.):
        super().__init__(prod_rate, rt_rate, deg_rate)
        self.tau_dist = tau_dist
        self.exp_noise = exp_noise

    def run(self,
        t_span: tuple[float, float],
        n_steps: int | None = None,
        init_tissue_rma: float = 0.,
        init_blood_rma: float = 0.,
    ):
        if n_steps:
            time = np.linspace(t_span[0], t_span[1], n_steps)
        else:
            time = np.linspace(t_span[0], t_span[1])
            n_steps = len(time)

        dt = time[1] - time[0]

        protein_tissue = np.zeros(n_steps)
        protein_blood = np.zeros(n_steps)
        protein_tissue[0] = init_tissue_rma
        protein_blood[0] = init_blood_rma

        wiener = np.random.normal(0, np.sqrt(dt), n_steps-1)
        tau = self.tau_dist(n_steps - 1)

        if isinstance(self.prod_rate, Callable):
            prod_rate = [self.prod_rate(t) for t in time]
        else:
            prod_rate = [self.prod_rate] * n_steps

        for i in range(n_steps - 1):
            dtissue = (prod_rate[i] - self.rt_rate * protein_tissue[i]) * dt + (protein_tissue[i] * wiener[i] / np.sqrt(tau[i]))
            dblood = (self.rt_rate * protein_tissue[i] - self.deg_rate * protein_blood[i]) * dt # (protein_blood[i] *  wiener[i] / np.sqrt(tau[i]))

            protein_tissue[i+1] = protein_tissue[i] + dtissue
            protein_blood[i+1] = protein_blood[i] + dblood

        return Solution(time, protein_tissue, protein_blood,
            {
                "t_span": t_span,
                "n_steps": n_steps,
                "init_tissue_rma": init_tissue_rma,
                "init_blood_rma": init_blood_rma,
            }
        )


class NoisyRMA_DDE(Model):
    """
    Stochastic kinetic model of RMA expression and transport.

    Parameters
    ----------

    prod_rate : float | Callable[[float], float]
        Tissue RMA production rate constant or time-dependent function.
    rt_rate : float
        Tissue RMA reverse transcytosis rate constant.
    deg_rate : float
        Blood RMA degradation rate constant.
    tau_dist : Callable[[int], np.ndarray]
        Function to generate delay times. The function should accept an integer
        n and return an array of n delay times. For example:
            tau_dist = lambda n: np.random.uniform(0, 1, n)
    """

    def __init__(
        self,
        prod_rate: float | Callable[[float], float],
        rt_rate: float,
        deg_rate: float,
        tau_dist: Callable[[int], np.ndarray], # np.random.uniform(Dl^2/2D, Dh^2/2D)
    ):
        super().__init__(prod_rate, rt_rate, deg_rate)
        self.tau_dist = tau_dist

    def run(
        self,
        t_span: tuple[float, float],
        n_steps: int | None = None,
        init_tissue_rma: float = 0.,
        init_blood_rma: float = 0.,
    ) -> Solution:

        if t_span[0] >= t_span[1]:
            raise ValueError("Invalid time span: t_span[0] must be less than t_span[1].")
        elif t_span[0] < 0:
            raise ValueError("Invalid time span: t_span[0] must be greater than or equal to 0.")

        if n_steps:
            time = np.linspace(t_span[0], t_span[1], n_steps)
        else:
            time = np.linspace(t_span[0], t_span[1])
            n_steps = len(time)

        dt = time[1] - time[0]
        tau = self.tau_dist(n_steps - 1)

        delayed_tissue_rma = deque()

        # setup initial concentrations
        tissue_rma = np.zeros(n_steps)
        blood_rma = np.zeros(n_steps)
        tissue_rma[0] = init_tissue_rma
        blood_rma[0] = init_blood_rma

        if isinstance(self.prod_rate, Callable):
            prod_rate = [self.prod_rate(t) for t in time]
        else:
            prod_rate = [self.prod_rate] * n_steps

        for i, t in enumerate(time[:-1]):
            dtissue_rma = 0
            dblood_rma = 0
            # new RMA is produced in the tissue and as a delay time tau[i]
            tissue_rma_prod = prod_rate[i] * dt
            time_delay = tau[i]
            # we add the newly produced RMA to a queue with it's'
            # arrival time (i.e., time it will be available for transport)
            dtissue_rma += tissue_rma_prod
            delayed_tissue_rma.append((tissue_rma_prod, t + time_delay))

            # we subsequently remove RMA from the tissue as it is transported
            rma_transport = 0
            while delayed_tissue_rma and delayed_tissue_rma[0][1] <= t:
                delayed_rma = delayed_tissue_rma.popleft()
                rma_transport += delayed_rma[0]

            rt_flux = rma_transport * self.rt_rate * dt
            dtissue_rma -= rt_flux
            dblood_rma += rt_flux - (self.deg_rate * blood_rma[i] * dt)
            # update the concentrations
            tissue_rma[i+1] = tissue_rma[i] + dtissue_rma
            blood_rma[i+1] = blood_rma[i] + dblood_rma

        return Solution(time, tissue_rma, blood_rma)
