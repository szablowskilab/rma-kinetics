import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from rma_kinetics.models import ForceRMA
    from jax import config as jax_config, numpy as jnp, random, vmap
    from jax.lax import cond
    from jax.scipy.signal import welch
    from SALib.sample import latin
    import seaborn as sb
    import matplotlib.pyplot as plt
    import polars as pl
    from matplotlib.collections import LineCollection
    from itertools import product
    import os
    from random import randint

    data_dir = os.path.join("notebooks", "data", "temporal_resolution")
    jax_config.update("jax_enable_x64", True)
    sb.set_theme("talk", "ticks", font="Arial")
    sb.set_palette("crest_r")


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Temporal Resolution of RMA Dynamics

    1. Modified RMA dynamics with forcing function (with varying serum half-life)
    2. Dynamic range vs. oscillation frequency and half-life
    3.
    """
    )
    return


@app.cell
def _():
    max_rma_prod_rate = 7e-3 # nM/hr
    rma_rt_rate = 0.6 # 1/hr
    rma_half_lives = [100, 50, 25, 12.5] # hrs
    oscillation_freq = 1/72 # 1/hr
    return max_rma_prod_rate, oscillation_freq, rma_half_lives, rma_rt_rate


@app.function
def plot_deg_rate_sweep(half_lives: list[float], rma_params: list[float], sim_params: dict[str, float]):
    for half_life in half_lives:
        deg_rate = jnp.log(2) / half_life
        sim_params["y0"] = (rma_params[0]/rma_params[1], rma_params[0]/deg_rate)
        model = ForceRMA(rma_params[0], rma_params[1], deg_rate, rma_params[2])
        solution = model.simulate(**sim_params)

        plt.plot(solution.ts, solution.ys[1]/(rma_params[0]/deg_rate), label=f"{half_life}")


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 1. ForceRMA with varying serum half life

    Constitutive model is updated to use an oscillating force function in the production term.
    """
    )
    return


@app.cell
def _(max_rma_prod_rate, oscillation_freq, rma_half_lives, rma_rt_rate):
    sweep_sim_config = {
        "t0": 0,
        "t1": 504,
        "dt0": 0.1,
    }

    plot_deg_rate_sweep(rma_half_lives, [max_rma_prod_rate, rma_rt_rate, oscillation_freq], sweep_sim_config)
    plt.tight_layout()
    sb.despine()
    plt.xlabel("Time (hr)")
    plt.ylabel("Normalized Plasma RMA (A.U.)")
    plt.savefig(os.path.join(data_dir, "normalized_rma_varying_deg_rate.svg"))
    plt.gca()

    return (sweep_sim_config,)


@app.function
def plot_freq_sweep(freqs: list[float], half_lives: list[float], rma_params: list[float], sim_params: dict[str, float]):
    markers = ["s", "^", "d", "o"]
    for half_life, marker in zip(half_lives, markers):
        dyn_range = []
        for freq in freqs:
            deg_rate = jnp.log(2) / half_life
            sim_params["y0"] = (rma_params[0]/rma_params[1], rma_params[0]/deg_rate)
            model = ForceRMA(rma_params[0], rma_params[1], deg_rate, freq)
            solution = model.simulate(**sim_params)
            norm_plasma_rma = solution.plasma_rma / (rma_params[0]/deg_rate)

            dyn_range.append(jnp.max(norm_plasma_rma) - jnp.min(norm_plasma_rma))

        plt.plot(freqs, dyn_range, label=f"{half_life}", marker=marker)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## 2. Dynamic range vs frequency and half-life""")
    return


@app.cell
def _(max_rma_prod_rate, rma_half_lives, rma_rt_rate, sweep_sim_config):
    freqs = [1/72, 1/48, 1/24, 1/12, 1/6]
    plot_freq_sweep(freqs, rma_half_lives, [max_rma_prod_rate, rma_rt_rate], sweep_sim_config)
    plt.tight_layout()
    sb.despine()
    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Dynamic Range (A.U.)")
    plt.legend(frameon=False, title="RMA Half-Life")
    plt.savefig(os.path.join(data_dir, "dynamic_range_varying_deg_and_freq.svg"))
    plt.gca()
    return


@app.function
def plot_prod_rate_sweep(rma_prod_rates: list[float], half_lives: list[float], rma_params: list[float], sim_params: dict[str, float]):
    markers = ["s", "^", "d", "o"]
    for half_life, marker in zip(half_lives, markers):
        max_rma = []
        deg_rate = jnp.log(2) / half_life
        for prod_rate in rma_prod_rates:
            sim_params["y0"] = (prod_rate/rma_params[0], prod_rate/deg_rate)
            model = ForceRMA(prod_rate, rma_params[0], deg_rate, rma_params[1])
            solution = model.simulate(**sim_params)

            max_rma.append(jnp.max(solution.plasma_rma))

        plt.plot(rma_prod_rates, max_rma, label=f"{half_life}", marker=marker)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## 3. Max plasma RMA vs max production rate and half-life""")
    return


@app.cell
def _(oscillation_freq, rma_half_lives, rma_rt_rate, sweep_sim_config):
    plot_prod_rate_sweep(
        [7e-3, 1e-2, 4e-2, 7e-2],
        rma_half_lives,
        [rma_rt_rate, oscillation_freq],
        sweep_sim_config
    )

    plt.tight_layout()
    sb.despine()
    plt.xlabel("RMA Production Rate (nM/hr)")
    plt.ylabel("Max Plasma RMA (nM)")
    plt.legend(frameon=False, title="RMA Half-Life")
    plt.savefig(os.path.join(data_dir, "max_intensity_varying_deg_and_prod.svg"))
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 4. Max plasma RMA vs dynamic range

    For a fixed oscillation frequency and production/reverse transcytosis rate (i.e., only varying serum half-life).
    """
    )
    return


@app.cell
def _(max_rma_prod_rate, oscillation_freq, rma_rt_rate, sweep_sim_config):
    def max_v_range_map(half_life: float):
        max_rma_conc = []
        dyn_range = []
        deg_rate = jnp.log(2)/half_life
        model = ForceRMA(max_rma_prod_rate, rma_rt_rate, deg_rate, oscillation_freq)
        sweep_sim_config["y0"] = (max_rma_prod_rate/rma_rt_rate, max_rma_prod_rate/deg_rate)
        solution = model.simulate(**sweep_sim_config)

        max_rma = jnp.max(solution.plasma_rma)
        norm_plasma_rma = solution.plasma_rma / (max_rma_prod_rate/deg_rate)
        dyn_range = jnp.max(norm_plasma_rma) - jnp.min(norm_plasma_rma)

        return max_rma, dyn_range

    return (max_v_range_map,)


@app.cell
def _(max_v_range_map):
    half_life_space = {
        "num_vars": 1,
        "names": ["rma_half_life"],
        "bounds": [[12.5, 100]]
    }

    half_life_vector = latin.sample(half_life_space, 1000)

    max_rma, dyn_range = vmap(max_v_range_map)(half_life_vector)
    plt.scatter(dyn_range, max_rma, marker='o', s=12, c=[p[0] for p in half_life_vector], cmap='crest')
    cbar = plt.colorbar()
    cbar.set_label('RMA Half-Life (hr)', rotation=270, labelpad=25)

    plt.ylabel("Max Concentration (nM)")
    plt.xlabel("Dynamic Range (A.U.)")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "max_intensity_v_dyn_range_varying_half_life.svg"))
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Adding Gaussian noise to the solution of the plasma RMA trajectory""")
    return


@app.function
def plot_noisy_rma_example(rma_params: list[float], noise_std: float, sim_config: dict[str, float], prng_key: jnp.ndarray):
    sim_config["y0"] = (0,0)
    deg_rate = jnp.log(2)/rma_params[-2]
    rma_params[-2] = deg_rate
    model = ForceRMA(*rma_params)
    solution = model.simulate(**sim_config)
    noisy_rma = model._apply_noise(solution.plasma_rma, noise_std, prng_key)

    plt.plot(solution.ts, noisy_rma, color='lightgrey', label=rf"$\sigma = {noise_std}$")
    plt.plot(solution.ts, solution.plasma_rma, label="Deterministic", color='black')


@app.cell
def _(max_rma_prod_rate, rma_half_lives, rma_rt_rate, sweep_sim_config):

    noisy_rma_example_oscillation_freq = 1/100
    plot_noisy_rma_example([max_rma_prod_rate, rma_rt_rate, rma_half_lives[0], noisy_rma_example_oscillation_freq], noise_std=0.2, sim_config=sweep_sim_config, prng_key=random.key(111))

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    sb.despine()
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "example_noisy_plasma_rma.svg"))
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 5. Calculate frequency recovery for given $f_i$ and $\sigma$.

    Testing single iteration.
    """
    )
    return


@app.cell
def _(max_rma_prod_rate, rma_half_lives, rma_rt_rate):
    def freq_recovery_single_iter():
        deg_rate = jnp.log(2) / rma_half_lives[0] # 100 hr serum half life
        target_freq = 1/15

        model = ForceRMA(max_rma_prod_rate, rma_rt_rate, deg_rate, target_freq)

        n_cycles = 50
        fs = 10
        t0 = 0; t1 = n_cycles // target_freq

        solution = model.simulate(
            t0=t0,
            t1=t1,
            dt0=0.01,
            y0=(max_rma_prod_rate/rma_rt_rate, max_rma_prod_rate/deg_rate),
            sampling_rate=fs,
            max_steps=10000
        )

        norm_plasma_rma = solution.plasma_rma / (max_rma_prod_rate/deg_rate)
        noisy_rma = model._apply_noise(norm_plasma_rma, std=0.2, prng_key=random.key(111))
        nperseg = len(noisy_rma) // 2
        f, psd = welch(noisy_rma - jnp.mean(noisy_rma), fs=fs, nperseg=nperseg)
        _, psd_clean = welch(norm_plasma_rma, fs=fs, nperseg=nperseg)
        fpeak = f[jnp.argmax(psd)]
        freq_match = jnp.isclose(fpeak, target_freq, rtol=0.05, atol=0)
        psd_noise = jnp.where(~jnp.isclose(f, target_freq, rtol=0.05), psd, jnp.nan)
        snr = fpeak / jnp.nanmean(psd_noise)
        print(fpeak)
        print(freq_match)
        print(snr)

        plt.plot(f, psd, 'lightgrey', label="Noise, $\sigma = 0.2$")
        plt.plot(f, psd_clean, 'k', label="Deterministic")

    freq_recovery_single_iter()
    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Power Spectral Density (PSD)")
    plt.tight_layout()
    sb.despine()
    plt.xlim(0, 0.1)
    # plot vertical line at oscillation frequency
    plt.savefig(os.path.join(data_dir, "example_noisy_psd_for_resolution.svg"))
    plt.gca()

    return


@app.cell
def _(max_rma_prod_rate, rma_half_lives, rma_rt_rate):
    # iterate over frequency range and fixed noise std
    def freq_recovery():
        #deg_rate = jnp.log(2) / rma_half_lives[0] # 100 hr serum half-life
        target_freqs = jnp.linspace(1/30, 1/3, 10)
        n_cycles = 50
        fs = 10
        simulation_config = {
            "t0": 0,
            "dt0": 0.01,
            "sampling_rate": fs,
            "max_steps": 10000
        }

        markers = ["s", "^", "d", "o"]
        for i, half_life in enumerate(rma_half_lives):
            percent_freq_recovery = []
            deg_rate = jnp.log(2) / half_life
            simulation_config["y0"] = (max_rma_prod_rate/rma_rt_rate, max_rma_prod_rate/deg_rate)
            for target_freq in target_freqs:
                model = ForceRMA(max_rma_prod_rate, rma_rt_rate, deg_rate, target_freq)
                simulation_config["t1"] = n_cycles // float(target_freq)
                recovery = model.freq_recovery(
                    simulation_config,
                    noise_std=0.05, # fixed noise
                    prng_key=random.key(randint(0, 2**32 - 1)),
                    target_freq=target_freq,
                    n_iter=1000,
                    fs=fs,
                    rtol=0.05,
                    min_snr=2
                )

                percent_freq_recovery.append(recovery)

            recovery_df = pl.DataFrame({
                "Frequency": list(target_freqs),
                "Percent Recovery": percent_freq_recovery
            })

            recovery_df.write_parquet(os.path.join(data_dir,f"202508_freq_recovery_deg_rate_{deg_rate}.parquet"))
            plt.plot(target_freqs, percent_freq_recovery, marker=markers[i])

        plt.xlabel("Frequency (1/hr)")
        plt.ylabel("Fraction of Frequency Recovery")
        #plt.legend(["100", "50", "25", "12.5"], title="RMA Half-Life (hr)", frameon=False)
        plt.tight_layout()
        sb.despine()
        plt.savefig(os.path.join(data_dir, "202508_freq_recovery_all.svg"))
        plt.gca()

    freq_recovery()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## 5. Compute power ratio of noisy RMA signal""")
    return


@app.cell
def _(max_rma_prod_rate, oscillation_freq, rma_half_lives, rma_rt_rate):
    from scipy.signal import ShortTimeFFT, spectrogram
    from scipy.signal.windows import hann, gaussian
    from jax.scipy.signal import stft
    def spec():
        model = ForceRMA(
            max_rma_prod_rate,
            rma_rt_rate,
            jnp.log(2)/rma_half_lives[0],
            1
        )

        # 3600 points
        # 720 hr long
        # sampling rate 5 / hr
        fs = 20
        solution = model.simulate(
            t0=0,
            t1=50,
            dt0=0.01,
            sampling_rate=fs,
            y0=(
                max_rma_prod_rate/rma_rt_rate,
                max_rma_prod_rate/(jnp.log(2)/rma_half_lives[0])
            ),
        )
        test = jnp.sin(2 * jnp.pi * 1/2* jnp.linspace(0, 50, 1000))
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(solution.ts, test)
        #window = gaussian(50, std=8, sym=False)
        #SFT = ShortTimeFFT(window, hop=10, fs=20, scale_to='psd', mfft=200)
        #Sx2 = SFT.spectrogram(solution.plasma_rma)
        #Sx2 = SFT.spectrogram(test)
        f, t, Sxx = stft(solution.plasma_rma, fs=fs, nperseg=250, nfft=250)
        ax[1].set(xlabel=("Time (1/hr)"), ylabel=("Frequency (1/hr)"), ylim=(f[0], 10))
        ax[1].axhline(y=1, color='r', linestyle='--', label=f'Signal Freq ({oscillation_freq:.4f} 1/hr)')
        ax[1].imshow(jnp.abs(Sxx)**2)

        #t = SFT.t(solution.plasma_rma.shape[-1])
        #f = SFT.f
        #f, t, Sxx = spectrogram(solution.plasma_rma, fs)

        #ax[1].imshow(Sx2, extent=SFT.extent(solution.plasma_rma.shape[-1]), aspect='auto', origin='lower')
        #ax[1].colorbar(label="PSD")


        #fig, ax = plt.subplots(figsize=(6, 4))
        #ax.pcolormesh(t, f, Sxx, shading='gouraud')
        #ax.axhline(y=oscillation_freq)
        #ax.set(ylim=(0, 0.02))


    return spec, stft


@app.cell
def _(spec):
    spec()
    plt.gca()
    return


@app.cell
def _(stft):
    def plot_spectrogram(signal: jnp.ndarray, fs: int) -> None:
        nperseg = len(signal) // 2
        noverlap = nperseg // 2
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
        # square Zxx
        Zxx2 = jnp.abs(Zxx) ** 2
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(signal)
        ax[1].set_ylim(0, 0.2)
        #fig, ax = plt.subplots(figsize=(10, 6))
        map = ax[1].pcolormesh(t, f, Zxx2, shading='gouraud')
        fig.colorbar(map, label="PSD", ax=ax[1])
    return (plot_spectrogram,)


@app.function
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
    return jnp.trapezoid(power[idx], dx=f[1] - f[0])


@app.cell
def _(max_rma_prod_rate, plot_spectrogram, rma_half_lives, rma_rt_rate):
    test = 7e-3 * (1 + jnp.sin(2 * jnp.pi * 1/10 * jnp.linspace(0, 50, 1500)))
    model = ForceRMA(
        max_rma_prod_rate,
        rma_rt_rate,
        jnp.log(2)/rma_half_lives[0],
        1/10
    )

    # 3600 points
    # 720 hr long
    # sampling rate 5 / hr
    fs = 30
    rma_deg_rate = jnp.log(2) / rma_half_lives[0]
    solution = model.simulate(
        t0=0,
        t1=50,
        dt0=0.1,
        sampling_rate=fs,
        y0=(
            max_rma_prod_rate/rma_rt_rate,
            max_rma_prod_rate/rma_deg_rate
        ),
    )
    norm_plasma_rma = solution.plasma_rma / (max_rma_prod_rate/rma_deg_rate)
    plot_spectrogram(solution.plasma_rma - jnp.mean(solution.plasma_rma), 20)
    plt.tight_layout()
    plt.show()
    return solution, test


@app.cell
def _(test):
    plt.plot(test/(2*7e-3))
    return


@app.cell
def _(solution):
    rma_f, rma_psd = welch(solution.plasma_rma - jnp.mean(solution.plasma_rma), fs=30, nperseg=len(solution.plasma_rma)//2)
    return rma_f, rma_psd


@app.cell
def _(rma_f, rma_psd):
    plt.semilogx(rma_f, rma_psd)
    plt.fill_between(rma_f, rma_psd, where=jnp.logical_and(rma_f >= 0.08, rma_f <= 0.12), color='skyblue')
    plt.show()
    return


@app.cell
def _(rma_f, rma_psd):
    avg_band_power = bandpower(rma_psd, rma_f, (0.05, 0.15))
    avg_band_power_not = bandpower(rma_psd, rma_f, (0.05, 0.15), logical_and=False)
    print(avg_band_power)
    print(avg_band_power_not)
    return avg_band_power, avg_band_power_not


@app.cell
def _(avg_band_power, avg_band_power_not):
    avg_band_power / avg_band_power_not
    return


@app.cell
def _(rma_f):
    rma_f
    return


@app.cell
def _(rma_psd):
    rma_psd[2]
    return


@app.cell
def _(max_rma_prod_rate, rma_half_lives, rma_rt_rate):

    def cutoff_freq_single_iter():
        ratios = []
        target_freq = jnp.linspace(1/30, 1/3, 10)
        for f in target_freq:
            test = 7e-3 * (1 + jnp.sin(2 * jnp.pi * 1/10 * jnp.linspace(0, 50, 1500)))

            model = ForceRMA(
                max_rma_prod_rate,
                rma_rt_rate,
                jnp.log(2)/rma_half_lives[0],
                f
            )

            # 3600 points
            # 720 hr long
            # sampling rate 5 / hr
            fs = 30
            rma_deg_rate = jnp.log(2) / rma_half_lives[0]
            solution = model.simulate(
                t0=0,
                t1=500,
                dt0=0.1,
                sampling_rate=fs,
                y0=(
                    max_rma_prod_rate/rma_rt_rate,
                    max_rma_prod_rate/rma_deg_rate
                ),
            )
            norm_plasma_rma = solution.plasma_rma / (max_rma_prod_rate/rma_deg_rate)
            # injection noise
            #plot_spectrogram(solution.plasma_rma - jnp.mean(solution.plasma_rma), 20)
            #plt.tight_layout()
            #plt.show()
            rma_f, rma_psd = welch(
                norm_plasma_rma - jnp.mean(norm_plasma_rma),
                fs=30,
                nperseg=len(solution.plasma_rma)//2
            )

            bands = (f - 0.25*f, f + 0.25*f)
            avg_band_power = bandpower(rma_psd, rma_f, bands)
            avg_band_power_not = bandpower(rma_psd, rma_f, bands, logical_and=False)
            ratio = avg_band_power / avg_band_power_not
            ratios.append(ratio)
        return ratios

    return (cutoff_freq_single_iter,)


@app.cell
def _(cutoff_freq_single_iter):
    cutoff_freq_single_iter()
    return


@app.cell
def _(max_rma_prod_rate, rma_half_lives, rma_rt_rate):
    def cutoff_freq():
        target_freq = jnp.linspace(1/30, 1/3, 10)
        noise_stds = jnp.linspace(0, 0.2, 20)
        deg_rate = jnp.log(2) / rma_half_lives[0] # 100 hr serum half-life
        n_cycles = 50
        fs = 10
        simulation_config = {
            "t0": 0,
            "dt0": 0.01,
            "y0": (max_rma_prod_rate/rma_rt_rate, max_rma_prod_rate/deg_rate),
            "sampling_rate": fs,
            "max_steps": 10000
        }

        avg_power = []
        freqs = []
        noise = []
        for (f, n) in product(target_freq, noise_stds):
            model = ForceRMA(max_rma_prod_rate, rma_rt_rate, deg_rate, f)
            simulation_config["t1"] = n_cycles // float(f)

            if n > 0:
                n_iter = 100
            else:
                n_iter = 1

            power = model.power_cutoff(
                simulation_config,
                noise_std=n, # fixed noise
                prng_key=random.key(randint(0, 2**32 - 1)),
                target_freq=f,
                n_iter=n_iter,
                fs=fs,
                rtol=0.05,
            )

            avg_power.append(power)
            freqs.append(f)
            noise.append(n)

        power_df = pl.DataFrame({
            "Frequency": freqs,
            "Noise": noise,
            "Average Power": avg_power
        })

        power_df.write_parquet(os.path.join(data_dir, f"202508_power_cutoff_deg_rate_{deg_rate}.parquet"))

    cutoff_freq()
    return


@app.cell
def _():
    power_df = pl.read_parquet(os.path.join(data_dir, "202508_power_cutoff_deg_rate_0.006931471805599453.parquet"))
    return (power_df,)


@app.cell
def _(power_df):
    power_df
    return


@app.cell
def _():
    # bandpower example schematic
    # exponential decay from 1 (max)
    example_noise = jnp.linspace(0,0.2, 20)
    example_relative_power = jnp.exp(-20*example_noise)

    plt.plot(example_noise, example_relative_power, 'k')
    plt.plot(example_noise, example_noise/2, '--', color='lightgrey')
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Relative Power")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "example_relative_power_schematic.svg"))
    plt.gca()
    return


@app.cell
def _(power_df):
    def _():
        # plot the power ratio vs noise for a given frequency
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        target_freqs = jnp.linspace(1/30, 1/3, 10)

        cmap = cm.get_cmap('flare_r')
        norm = Normalize(vmin=target_freqs.min(), vmax=target_freqs.max())

        for f in target_freqs:
            power_df_sub = power_df.filter(pl.col("Frequency") == f)
            color = cmap(norm(f))  # Get color for this frequency
            plt.plot(power_df_sub["Noise"], power_df_sub["Average Power"], color=color)

        plt.xlabel("Noise Standard Deviation")
        plt.ylabel("Relative Power")
        sb.despine()
        # add colorbar for target_freqs
        #plt.colorbar(label="Frequency (1/hr)")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array, just need the mappable object
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Frequency (1/hr)")
        cbar.set_label('Frequency (1/hr)', rotation=270, labelpad=25)
        plt.plot(power_df_sub["Noise"], power_df_sub["Noise"] / 2, '--', color='lightgrey')
        plt.tight_layout()

        plt.savefig(os.path.join(data_dir, "202508_power_ratio_100hr_half_life.svg"))
        return plt.gca()


    _()
    return


@app.cell
def _(max_rma_prod_rate, rma_half_lives, rma_rt_rate):
    def coherence_cutoff():
        target_freq = jnp.linspace(1/30, 1/3, 10)
        noise_stds = jnp.linspace(0, 0.2, 20)
        deg_rate = jnp.log(2) / rma_half_lives[0] # 100 hr serum half-life
        n_cycles = 50
        fs = 10
        simulation_config = {
            "t0": 0,
            "dt0": 0.01,
            "y0": (max_rma_prod_rate/rma_rt_rate, max_rma_prod_rate/deg_rate),
            "sampling_rate": fs,
            "max_steps": 10000
        }

        avg_coh = []
        freqs = []
        noise = []
        for (f, n) in product(target_freq, noise_stds):
            model = ForceRMA(max_rma_prod_rate, rma_rt_rate, deg_rate, f)
            simulation_config["t1"] = n_cycles // float(f)

            if n > 0:
                n_iter = 5000
            else:
                n_iter = 1

            _ts = jnp.linspace(simulation_config["t0"], simulation_config["t1"], int(fs*simulation_config["t1"]))
            input_signal = vmap(model._force_rma_prod_rate)(_ts)
            _, input_psd = welch(input_signal, fs=fs, nperseg=len(input_signal)//2)
            coh = model.coh_cutoff(
                simulation_config,
                input_signal,
                input_psd,
                noise_std=n, # fixed noise
                prng_key=random.key(randint(0, 2**32 - 1)),
                target_freq=f,
                n_iter=n_iter,
                fs=fs,
            )

            avg_coh.append(coh)
            freqs.append(f)
            noise.append(n)

        power_df = pl.DataFrame({
            "Frequency": freqs,
            "Noise": noise,
            "Coherence": avg_coh
        })

        power_df.write_parquet(os.path.join(data_dir, f"202508_coh_cutoff_deg_rate_{deg_rate}.parquet"))

    coherence_cutoff()
    return


@app.cell
def _():
    coh_df = pl.read_parquet(os.path.join(data_dir, "202508_coh_cutoff_deg_rate_0.006931471805599453.parquet"))
    return (coh_df,)


@app.cell
def _(coh_df):
    coh_df
    return


@app.cell
def _(coh_df):
    def _():
        # plot the power ratio vs noise for a given frequency
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        target_freqs = jnp.linspace(1/30, 1/3, 10)

        cmap = cm.get_cmap('flare_r')
        norm = Normalize(vmin=target_freqs.min(), vmax=target_freqs.max())

        for f in target_freqs:
            power_df_sub = coh_df.filter(pl.col("Frequency") == f)
            color = cmap(norm(f))  # Get color for this frequency
            plt.plot(power_df_sub["Noise"], power_df_sub["Coherence"], color=color)

        plt.xlabel("Noise Standard Deviation")
        plt.ylabel("Coherence (R)")
        sb.despine()
        # add colorbar for target_freqs
        #plt.colorbar(label="Frequency (1/hr)")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array, just need the mappable object
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Frequency (1/hr)")
        cbar.set_label('Frequency (1/hr)', rotation=270, labelpad=25)
        plt.tight_layout()

        plt.savefig(os.path.join(data_dir, "202508_coh_100hr_half_life.svg"))
        return plt.gca()


    _()
    return


@app.cell
def _():
    # calculate frequency where coherence drops to 0.5
    def coh_cutoff_freq():
        coh_df = pl.read_parquet(os.path.join(data_dir, "202508_coh_cutoff_deg_rate_0.006931471805599453.parquet"))
        coh_df = coh_df.filter(pl.col("Coherence") > 0.5)
        coh_df = coh_df.group_by("Noise").agg(pl.col("Frequency").max().alias("Frequency"))
        return coh_df
    cutoff_coh = coh_cutoff_freq()
    return (cutoff_coh,)


@app.cell
def _(cutoff_coh):
    cutoff_coh
    return


@app.cell
def _(cutoff_coh):
    plt.scatter(cutoff_coh["Noise"], cutoff_coh["Frequency"])
    return


@app.cell
def _():
    mo.md(r"""Sensitivy analysis of degradation rate, noise level, production rate, reverse transcytosis rate on temporal resolution and precision""")
    return


@app.cell
def _():
    from sensitivity import global_sensitivity

    def map_temporal_precision(params):
        prod_rate, rt_rate, deg_rate, noise_level = params

        target_freq = 1/10
        n_cycles = 50
        t1 = n_cycles // target_freq
        fs = 10
        simulation_config = {
            "t0": 0,
            "t1": t1, 
            "dt0": 0.01,
            "y0": (prod_rate/rt_rate, prod_rate/deg_rate),
            "sampling_rate": fs,
            "max_steps": 10000,
        }

        percent_freq_recovery = []
        model = ForceRMA(prod_rate, rt_rate, deg_rate, target_freq)
        recovery = model.freq_recovery(
            simulation_config,
            noise_std=noise_level, # fixed noise
            prng_key=random.key(randint(0, 2**32 - 1)),
            target_freq=target_freq,
            n_iter=100,
            fs=fs,
            rtol=0.05,
            min_snr=2
        )

        return recovery

    parameter_space = {
        "num_vars": 4,
        "names": ["$k_{RMA}$", "$k_{RT}$", "$\gamma_{RMA}$", "$\sigma$"],
        "bounds": [
            [3.5e-3, 1.05e-2],
            [0.3, 0.9],
            [7e-3, 5.5e-2],
            [0.01, 0.2]
        ]
    }

    #morris_tp_recovery, morris_tp_sens = global_sensitivity(map_temporal_precision, parameter_space, 10)
    from SALib.sample import morris as morris_sampler
    from SALib.analyze import morris as morris_analyzer

    tp_param_vectors = morris_sampler.sample(parameter_space, 250)
    tp_y = vmap(map_temporal_precision)(tp_param_vectors)
    tp_sens = morris_analyzer.analyze(parameter_space, tp_param_vectors, tp_y)
    return morris_analyzer, morris_sampler, parameter_space, tp_sens


@app.cell
def _(tp_sens):
    tp_sens
    return


@app.cell
def _(parameter_space, tp_sens):
    max_mu_star_tp = max(tp_sens["mu_star"])
    norm_mu_star_tp = tp_sens["mu_star"] / max_mu_star_tp 
    norm_mu_star_conf = tp_sens["mu_star_conf"] / max_mu_star_tp
    norm_sigma = tp_sens["sigma"] / jnp.max(tp_sens["sigma"])

    plt.bar(parameter_space["names"], norm_mu_star_tp, yerr=norm_mu_star_conf, color='lightgrey')
    plt.ylabel("Relative Ranking")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_tp.svg"))
    plt.gca()
    return (norm_sigma,)


@app.cell
def _(norm_sigma, parameter_space):
    plt.bar(parameter_space["names"], norm_sigma, color='lightgrey')
    plt.ylabel("Relative Nonlinearity or Interaction")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_tp.svg"))
    plt.gca()
    return


@app.cell
def _(morris_analyzer, morris_sampler, parameter_space):

    def map_temporal_resolution(params):
        prod_rate, rt_rate, deg_rate, noise_level = params

        target_freq = 1/10
        n_cycles = 50
        t1 = n_cycles // target_freq
        fs = 10
        simulation_config = {
            "t0": 0,
            "t1": t1, 
            "dt0": 0.01,
            "y0": (prod_rate/rt_rate, prod_rate/deg_rate),
            "sampling_rate": fs,
            "max_steps": 10000,
        }

        percent_freq_recovery = []
        model = ForceRMA(prod_rate, rt_rate, deg_rate, target_freq)
        power_ratio = model.power_cutoff(
            simulation_config,
            noise_std=noise_level, # fixed noise
            prng_key=random.key(randint(0, 2**32 - 1)),
            target_freq=target_freq,
            n_iter=100,
            fs=fs,
            rtol=0.05,
        )

        return power_ratio

    tr_param_vectors = morris_sampler.sample(parameter_space, 250)
    tr_y = vmap(map_temporal_resolution)(tr_param_vectors)
    tr_sens = morris_analyzer.analyze(parameter_space, tr_param_vectors, tr_y)
    return map_temporal_resolution, tr_sens


@app.cell
def _(parameter_space, tr_sens):
    max_mu_star_tr = max(tr_sens["mu_star"])
    norm_mu_star_tr = tr_sens["mu_star"] / max_mu_star_tr 
    norm_mu_star_conf_tr = tr_sens["mu_star_conf"] / max_mu_star_tr
    norm_sigma_tr = tr_sens["sigma"] / jnp.max(tr_sens["sigma"])

    plt.bar(parameter_space["names"], norm_mu_star_tr, yerr=norm_mu_star_conf_tr, color='lightgrey')
    plt.ylabel("Relative Ranking")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_tr.svg"))
    plt.gca()
    return (norm_sigma_tr,)


@app.cell
def _(norm_sigma_tr, parameter_space):
    plt.bar(parameter_space["names"], norm_sigma_tr, color='lightgrey')
    plt.ylabel("Relative Nonlinearity or Interaction")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_tr.svg"))
    plt.gca()
    return


@app.cell
def _(
    map_temporal_resolution,
    morris_analyzer,
    morris_sampler,
    parameter_space,
):

    def map_coherence(params):
        prod_rate, rt_rate, deg_rate, noise_level = params

        target_freq = 1/10
        n_cycles = 50
        t1 = n_cycles // target_freq
        fs = 10
        simulation_config = {
            "t0": 0,
            "t1": t1, 
            "dt0": 0.01,
            "y0": (prod_rate/rt_rate, prod_rate/deg_rate),
            "sampling_rate": fs,
            "max_steps": 10000,
        }

        percent_freq_recovery = []
        model = ForceRMA(prod_rate, rt_rate, deg_rate, target_freq)
        _ts = jnp.linspace(simulation_config["t0"], simulation_config["t1"], int(fs*simulation_config["t1"]))
        input_signal = vmap(model._force_rma_prod_rate)(_ts)
        _, input_psd = welch(input_signal, fs=fs, nperseg=len(input_signal)//2)

        coherence = model.coh_cutoff(
            simulation_config,
            input_signal,
            input_psd,
            noise_std=noise_level, # fixed noise
            prng_key=random.key(randint(0, 2**32 - 1)),
            target_freq=target_freq,
            n_iter=100,
            fs=fs,
        )

        return coherence


    coh_param_vectors = morris_sampler.sample(parameter_space, 250)
    coh_y = vmap(map_temporal_resolution)(coh_param_vectors)
    coh_sens = morris_analyzer.analyze(parameter_space, coh_param_vectors, coh_y)
    return (coh_sens,)


@app.cell
def _(coh_sens, parameter_space, tr_sens):

    max_mu_star_coh = max(tr_sens["mu_star"])
    norm_mu_star_coh = coh_sens["mu_star"] / max_mu_star_coh 
    norm_mu_star_conf_coh = coh_sens["mu_star_conf"] / max_mu_star_coh
    norm_sigma_coh = coh_sens["sigma"] / jnp.max(coh_sens["sigma"])

    plt.bar(parameter_space["names"], norm_mu_star_coh, yerr=norm_mu_star_conf_coh, color='lightgrey')
    plt.ylabel("Relative Ranking")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_mean_coh.svg"))
    plt.gca()
    return (norm_sigma_coh,)


@app.cell
def _(norm_sigma_coh, parameter_space):
    plt.bar(parameter_space["names"], norm_sigma_coh, color='lightgrey')
    plt.ylabel("Relative Nonlinearity or Interaction")
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "norm_morris_std_coh.svg"))
    plt.gca()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
