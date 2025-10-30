from .abstract import AbstractModel
from ..units import Time, Concentration
from .dox import DoxPK, DoxPKConfig
from jaxtyping import PyTree
from diffrax import ODETerm, AbstractTerm


class TetRMA(AbstractModel):
    r"""
    Tetracycline transcriptional activator (tTA) induced RMA expression model.

    Attributes:
        rma_prod_rate (float): RMA production rate, $k_{RMA}$.
        rma_rt_rate (float): RMA reverse transcytosis rate, $k_{RT}$.
        rma_deg_rate (float): RMA degradation rate, $\gamma_{RMA}$.
        dox_model_config (DoxPKConfig): Dox PK model configuration.
        dox_kd (float): Dox dissocation constant, $K_{D_{Dox}}$.
        tta_prod_rate (float): tTA production rate, $k_{tTA}$.
        tta_deg_rate (float): tTA degradation rate, $\gamma_{tTA}$.
        tta_kd (float): tTA-TetO operator dissocation constant, $K_{D_{tTA}}$.
        leaky_rma_prod_rate (float): Leaky RMA production rate, $k_{0_{RMA}}$ (Default = 0.0).
        tta_coop (int): tTA cooperativity, $n_{tTA}$ (Default = 2).
        time_units (Time): Time units (Default = Time.hours).
        conc_units (Concentration): Concentration units (Default = Concentration.nanomolar).
    """
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
        time_units: Time = Time.hours,
        conc_units: Concentration = Concentration.nanomolar
    ):

        super().__init__(rma_prod_rate, rma_rt_rate, rma_deg_rate, time_units=time_units, conc_units=conc_units)

        self.dox = DoxPK(dox_model_config)
        self.dox_kd = dox_kd

        self.tta_prod_rate = tta_prod_rate
        self.tta_deg_rate = tta_deg_rate
        self.tta_kd = tta_kd
        self.leaky_rma_prod_rate = leaky_rma_prod_rate
        self.tta_coop = tta_coop

    def _tet_rma_model(self, t: float, y: PyTree[float]) -> PyTree[float]:
        """
        Tet induced RMA expression compartments.

        Arguments:
            t (float): Time points.
            y (PyTree[float]): Concentrations of plasma/brain RMA, transcriptional activator,
                plasma/brain dox.

        Returns:
            dydt (PyTree[float]): Brain/plasma RMA and dox concentrations.
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
        r"""
        Full ODE model implementation.

        Info: Model Equations
            Note that this model assumes constitutive expression of tTA.

            $$\begin{align}
            \dot{[TA]} &= k_{TA} - \gamma_{TA}[TA] \tag{1} \\
            [TA]_{SS} &= \frac{k_{TA}}{\gamma_{TA}} \tag{2}
            \end{align}
            $$

            Doxycycline is the preferred inhibitor (although tetracycline or other
            derivatives may be used by updating the [DoxPKConfig](https://szablowskilab/rma-kinetics/docs/api/dox/config).
            The fraction of the transcriptional activator available for inducing RMA
            expression is then modeled with a Hill function and modified for leaky expression,


            $$\begin{align}
            \theta_{tTA} &= \frac{1}{1 + \frac{[Dox]}{K_{D_{Dox}}}} \tag{3} \\
            \dot{[RMA_{B}]} &= \frac{k_{0_{RMA}} + k_{RMA}\left(\frac{\theta_{TA}[TA]}{K_{D_{TA}}}\right)^{n_{tTA}}}{1 + \left(\frac{\theta_{TA}[TA]}{K_{D_{TA}}}\right)^{n_{tTA}}} - k_{RT}[RMA_{B}] \tag{4} \\
            \dot{[RMA_{P}]} &= k_{RT}[RMA_{B}] - \gamma_{RMA}[RMA_{P}] \tag{5}
            \end{align}
            $$

            |Parameters|Description|Units (Example)|
            |----------|-----------|-----|
            |$k_{RMA}$|RMA production rate|Concentration/Time (nM/hr)|
            |$k_{0_{RMA}}$|Leaky RMA production rate|Concentration/Time (nM/hr)|
            |$k_{RT}$|RMA reverse transcytosis rate|1/Time (1/hr)|
            |$\gamma_{RMA}$|RMA degradation rate|1/Time (1/hr)|
            |$k_{tTA}$|tTA production rate|Concentration/Time (nM/hr)|
            |$\gamma_{tTA}$|tTA degradation rate|1/Time (1/hr)|
            |$K_{D_{Dox}}$|Dox-tTA binding dissocation constant|Concentration (nM)|
            |$K_{D_{tTA}}$|tTA-TetO binding dissocation constant|Concentration (nM)|
            |$n_{tTA}$|tTA cooperativity|Unitless|
            |$[RMA_B]$|Brain RMA concentration|Concentration (nM)|
            |$[RMA_P]$|Plasma RMA concentration|Concentration (nM)|
            |$[tTA]$|Brain tTA concentration|Concentration (nM)|
            |$[Dox]$|Brain Dox concentration|Concentration (nM)|
        """
        brain_rma, plasma_rma, ta, brain_dox, plasma_dox = y
        dbrain_rma, dplasma_rma, dbrain_dox, dplasma_dox = self._tet_rma_model(t, (brain_rma, plasma_rma, ta, brain_dox, plasma_dox))

        dta = self.tta_prod_rate - (self.tta_deg_rate * ta) # constitutive TA expression

        return dbrain_rma, dplasma_rma, dta, dbrain_dox, dplasma_dox

    def _terms(self) -> AbstractTerm:
        return ODETerm(self._model)
