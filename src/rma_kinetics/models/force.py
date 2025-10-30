from .abstract import AbstractModel
from ..units import Time, Concentration
from diffrax import ODETerm, AbstractTerm, diffeqsolve

from jax import (
    numpy as jnp,
    random as jrand
)

from jax.scipy.signal import welch, csd
from jax.lax import fori_loop, cond
from scipy.signal import coherence

from jaxtyping import PyTree, Array
from typing import Any

import jax.debug as debug

def bandpower(
    power: jnp.ndarray,
    f: jnp.ndarray,
    frange: tuple[float, float],
    logical_and: bool = True
) -> jnp.ndarray:
    """
    Compute the average band power of a signal from its power spectrum.

    Arguments:
        power (jnp.ndarray): Power spectral estimate of the signal.
        f (jnp.ndarray): Frequency vector of the power spectral estimate.
        frange (tuple[float, float]): Frequency range over which to compute the band power.
        logical_and (bool): If True, compute the band power in the frequency range; if False, compute outside the range.

    Returns:
        avg_band_power (jnp.ndarray): Average band power in the specified frequency range.
    """
    idx = cond(
        logical_and,
        lambda: jnp.logical_and(f >= frange[0], f <= frange[1]),
        lambda: ~jnp.logical_and(f >= frange[0], f <= frange[1]),
    )
    dx = f[1] - f[0]
    return jnp.where(idx, power, 0.0).sum() * dx


def total_power(power, dx):
    return jnp.sum(power) * dx


class ForceRMA(AbstractModel):
    """
    Model of rapidly changing RMA expression.

    Attributes:
        rma_prod_rate (float): RMA production rate (concentration/time).
        rma_rt_rate (float): RMA reverse transcytosis rate (1/time).
        rma_deg_rate (float): RMA degradation rate (1/time).
        freq (float): Frequency of oscillations (1/time).
    """
    rma_prod_rate: float
    rma_rt_rate: float
    rma_deg_rate: float
    freq: float

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
        """
        Sinusoid oscillating RMA production rate.

        Args:
            t (float): Time point.

        Returns:
            rma_prod_rate (jax.Array): RMA production rate evaluated at time `t`.
        """
        return self.rma_prod_rate * (1 + jnp.sin(2 * jnp.pi * self.freq * t))

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        """
        ODE model implementation. See the model equations section for more details.

        Arguments:
            t (float): Time point.
            y (PyTree[float]): Brain and plasma RMA concentrations.

        Returns:
            dydt (PyTree[float]): Change in brain and plasma RMA concentrations.
        """
        brain_rma, plasma_rma = y
        plasma_rma_transport_flux = self.rma_rt_rate * brain_rma
        dbrain_rma = self._force_rma_prod_rate(t) - plasma_rma_transport_flux
        dplasma_rma = plasma_rma_transport_flux - (self.rma_deg_rate * plasma_rma)

        return dbrain_rma, dplasma_rma

    def _terms(self) -> AbstractTerm:
        return ODETerm(self._model)

    def _apply_noise(self, solution: Array, std: float, prng_key: Array) -> Array:
        """
        Apply Gaussian noise to a given trajectory.

        Args:
            solution (jax.Array): Solution/trajectory to apply noise to.
            std (float): Noise standard deviation.
            prng_key (jax.Array): Jax PRNG key.

        Returns:
            noisy_solution (jax.Array): Solution with applied Gaussian noise.
        """
        noise = std * jrand.normal(prng_key, shape=(len(solution),))
        return jnp.clip(solution * (1 + noise), min=0)

    def freq_recovery(
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
        """
        Calculate temporal resolution of model at a given noise level
        and target frequency.

        Args:
            simulation_config (dict[str, Any]): `simulate` method params as a dictionary.
            noise_std (float): Gaussian noise standard deviation.
            prng_key (jax.Array): Jax PRNG key.
            target_freq (float): Target frequency of true RMA oscillations.
            n_iter (int): Number of iterations to run for bootstrapping.
            fs (float): Sampling frequency.
            rtol (float): Relative tolerance for matching target and recovered frequencies.
            min_snr (float): Minimum signal-to-noise ratio to consider recovered frequencies.

        Returns:
            resolution (float): Percent recovery of target frequency.
        """
        resolution = 0
        keys = jrand.split(prng_key, n_iter)

        def body_fn(i, resolution):
            # solution = diffeqsolve(
            #     self._terms(),
            #     **simulation_config
            # )
            solution = self.simulate(**simulation_config)
            plasma_rma = cond(
                noise_std > 0,
                lambda: self._apply_noise(solution.ys[1], noise_std, keys[i]),
                lambda: solution.ys[1]
            )

            norm_plasma_rma = plasma_rma / (self.rma_prod_rate/self.rma_deg_rate)
            nperseg = len(norm_plasma_rma) // 2

            freq, psd = welch(norm_plasma_rma - jnp.mean(norm_plasma_rma), fs=fs, nperseg=nperseg)
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

    def coherence_cutoff(
        self,
        input_signal: Array,
        input_psd: Array,
        simulation_config: dict[str, Any],
        noise_std: float,
        prng_key: Array,
        target_freq: float,
        n_iter: int,
        fs: float,
    ):
        """
        Calculate coherence based frequency cutoff of model at a given noise level
        and target frequency.

        Args:
            input_signal (Array): Input sine wave used to drive RMA production.
            input_psd (Array): Input signal power spectral density.
            simulation_config (dict[str, Any]): `simulate` method params as a dictionary.
            noise_std (float): Gaussian noise standard deviation.
            prng_key (jax.Array): Jax PRNG key.
            target_freq (float): Target frequency of true RMA oscillations.
            n_iter (int): Number of iterations to run for bootstrapping.
            fs (float): Sampling frequency.

        Returns:
            coherence (float): Average magnitude-squared coherence at a given target input frequency.
        """
        sum_coh = 0
        keys = jrand.split(prng_key, n_iter)

        def body_fn(i, sum_coh):
            solution = diffeqsolve(
                self._terms(),
                **simulation_config
            )
            plasma_rma = cond(
                noise_std > 0,
                lambda: self._apply_noise(solution.ys[1], noise_std, keys[i]),
                lambda: solution.ys[1]
            )

            norm_plasma_rma = plasma_rma / (self.rma_prod_rate/self.rma_deg_rate)
            # freq, pyy = welch(norm_plasma_rma, fs=fs)
            # _, pxy = csd(input_signal, norm_plasma_rma, fs=fs)
            # coherence = jnp.abs(pxy)**2 / pyy / input_psd
            freq, coh = coherence(input_signal, norm_plasma_rma, fs=fs)
            # return coherence near the target frequency
            target_idx = jnp.argmin(jnp.abs(freq - target_freq))
            return sum_coh + coh[target_idx]

        sum_coh = fori_loop(0, n_iter, body_fn, sum_coh)
        return sum_coh / n_iter

    def power_cutoff(
        self,
        simulation_config: dict[str, Any],
        noise_std: float,
        prng_key: Array,
        target_freq: float,
        n_iter: int,
        fs: float,
        rtol: float
    ):
        """
        Calculate power based frequency cutoff of model at a given noise level
        and target frequency.

        Args:
            input_signal (Array): Input sine wave used to drive RMA production.
            input_psd (Array): Input signal power spectral density.
            simulation_config (dict[str, Any]): `simulate` method params as a dictionary.
            noise_std (float): Gaussian noise standard deviation.
            prng_key (jax.Array): Jax PRNG key.
            target_freq (float): Target frequency of true RMA oscillations.
            n_iter (int): Number of iterations to run for bootstrapping.
            fs (float): Sampling frequency.

        Returns:
            coherence (float): Average magnitude-squared coherence at a given target input frequency.
        """
        sum_power_ratio = 0
        keys = jrand.split(prng_key, n_iter)

        def body_fn(i, sum_power_ratio):
            solution = self.simulate(**simulation_config)
            plasma_rma = cond(
                noise_std > 0,
                lambda: self._apply_noise(solution.ys[1], noise_std, keys[i]),
                lambda: solution.ys[1]
            )

            norm_plasma_rma = plasma_rma / (self.rma_prod_rate/self.rma_deg_rate)
            nperseg = len(norm_plasma_rma) // 2

            freq, psd = welch(norm_plasma_rma - jnp.mean(norm_plasma_rma), fs=fs, nperseg=nperseg)

            bands = (target_freq - rtol*target_freq, target_freq + rtol*target_freq)
            avg_target_bandpower = bandpower(psd, freq, bands)
            # let's try to use the total bandpower instead
            #total_bandpower = bandpower(psd, freq, (float(freq[0]), float(freq[-1])))
            total_bandpower = jnp.sum(psd) * (freq[1] - freq[0])
            #avg_noise_bandpower = bandpower(psd, freq, bands, logical_and=False)
            power_ratio = avg_target_bandpower / total_bandpower
            return power_ratio + sum_power_ratio

        sum_power_ratio = fori_loop(0, n_iter, body_fn, sum_power_ratio)
        return sum_power_ratio / n_iter

    def coh_cutoff(
        self,
        simulation_config: dict[str, Any],
        input_signal: jnp.ndarray,
        input_psd: jnp.ndarray,
        noise_std: float,
        prng_key: Array,
        target_freq: float,
        n_iter: int,
        fs: float,
    ):
        """
        Calculate power based frequency cutoff of model at a given noise level
        and target frequency.

        Args:
            input_signal (Array): Input sine wave used to drive RMA production.
            input_psd (Array): Input signal power spectral density.
            simulation_config (dict[str, Any]): `simulate` method params as a dictionary.
            noise_std (float): Gaussian noise standard deviation.
            prng_key (jax.Array): Jax PRNG key.
            target_freq (float): Target frequency of true RMA oscillations.
            n_iter (int): Number of iterations to run for bootstrapping.
            fs (float): Sampling frequency.

        Returns:
            coherence (float): Average magnitude-squared coherence at a given target input frequency.
        """
        sum_coh = 0
        keys = jrand.split(prng_key, n_iter)

        def body_fn(i, sum_coh):
            solution = self.simulate(**simulation_config)
            plasma_rma = cond(
                noise_std > 0,
                lambda: self._apply_noise(solution.ys[1], noise_std, keys[i]),
                lambda: solution.ys[1]
            )

            norm_plasma_rma = plasma_rma / (self.rma_prod_rate/self.rma_deg_rate)

            #freq, coh = coherence(input_signal, norm_plasma_rma, fs=fs)
            nperseg = len(norm_plasma_rma) // 2
            freq, psdy = welch(norm_plasma_rma, fs=fs, nperseg=len(norm_plasma_rma)//2)
            _, psdxy = csd(input_signal, norm_plasma_rma, fs=fs, nperseg=nperseg)
            cxy = jnp.abs(psdxy)**2 / input_psd / psdy

            # return coherence near the target frequency
            target_idx = jnp.argmin(jnp.abs(freq - target_freq))
            return sum_coh + cxy[target_idx]

        sum_coh = fori_loop(0, n_iter, body_fn, sum_coh)
        return sum_coh / n_iter
