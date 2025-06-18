from .abstract import AbstractModel
from ..units import Time, Concentration
from diffrax import ODETerm, AbstractTerm

from jax import (
    numpy as jnp,
    random as jrand
)

from jax.scipy.signal import welch
from jax.lax import fori_loop, cond

from jaxtyping import PyTree, Array
from typing import Any

class ForceRMA(AbstractModel):
    rma_prod_rate: float
    rma_rt_rate: float
    rma_deg_rate: float
    freq: float
    """
    Model of rapidly changing RMA expression.

    Attributes:
        rma_prod_rate (float): RMA production rate (concentration/time).
        rma_rt_rate (float): RMA reverse transcytosis rate (1/time).
        rma_deg_rate (float): RMA degradation rate (1/time).
        freq (float): Frequency of oscillations (1/time).
    """

    def __init__(
        self,
        rma_prod_rate: float,
        rma_rt_rate: float,
        rma_deg_rate: float,
        freq: float,
        time_units: Time = Time.hours,
        conc_units: Concentration = Concentration.nanomolar
    ):
        super().__init__(rma_prod_rate, rma_rt_rate, rma_deg_rate, time_units, conc_units)
        self.freq = freq

    def _force_rma_prod_rate(self, t: float) -> Array:
        return self.rma_prod_rate * (1 + jnp.sin(2 * jnp.pi * self.freq * t))

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        brain_rma, plasma_rma = y
        plasma_rma_transport_flux = self.rma_rt_rate * brain_rma
        dbrain_rma = self._force_rma_prod_rate(t) - plasma_rma_transport_flux
        dplasma_rma = plasma_rma_transport_flux - (self.rma_deg_rate * plasma_rma)

        return dbrain_rma, dplasma_rma

    def _terms(self) -> AbstractTerm:
        return ODETerm(self._model)

    def _apply_noise(self, solution: Array, std: float, prng_key: Array) -> Array:
        noise = std * jrand.normal(prng_key, shape=(len(solution),))
        return jnp.clip(solution * (1 + noise), min=0)

    def max_temporal_resolution(
        self,
        simulation_config: dict[str, Any],
        noise_std: float,
        prng_key: Array,
        target_freq: float,
        n_iter: int,
        fs: float,
        rtol: float,
        min_snr: float
    ):
        resolution = 0
        keys = jrand.split(prng_key, n_iter)

        def body_fn(i, resolution):
            solution = self.simulate(**simulation_config)
            plasma_rma = cond(
                noise_std > 0,
                lambda: self._apply_noise(solution.ys[1], noise_std, keys[i]),
                lambda: solution.ys[1]
            )

            norm_plasma_rma = plasma_rma / (self.rma_prod_rate/self.rma_deg_rate)

            freq, psd = welch(norm_plasma_rma, fs=fs)
            fpeak = freq[jnp.argmax(psd)]
            freq_match = jnp.isclose(fpeak, target_freq, rtol=rtol, atol=0)
            psd_noise = jnp.where(~jnp.isclose(freq, target_freq, rtol=rtol), psd, jnp.nan)
            snr = fpeak / jnp.nanmean(psd_noise)
            increment = cond(
                jnp.logical_and(freq_match, snr >= min_snr),
                lambda: resolution + 1,
                lambda: resolution
            )

            return increment

        resolution = fori_loop(0, n_iter, body_fn, resolution)
        return resolution / n_iter
