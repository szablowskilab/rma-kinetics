import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from rma_kinetics.models import ForceRMA
    from diffrax import PIDController, SaveAt

    from random import randint
    from jax import numpy as jnp, config as jax_config, random
    from jax.scipy.signal import welch
    from jax.lax import cond, fori_loop
    from itertools import product

    import polars as pl
    import seaborn as sb
    import matplotlib.pyplot as plt
    import os

    data_dir = os.path.join("notebooks", "data", "temporal_resolution")
    jax_config.update("jax_enable_x64", True)
    sb.set_theme("talk", "ticks", font="Arial")
    sb.set_palette("crest_r")
    return (
        ForceRMA,
        PIDController,
        SaveAt,
        data_dir,
        jnp,
        mo,
        os,
        pl,
        plt,
        product,
        randint,
        random,
        sb,
        welch,
    )


@app.cell
def _(ForceRMA, PIDController, SaveAt, jnp, plt, sb):
    # normalized plasma RMA signal over time w.r.t half-life
    rma_half_life = [100, 50, 25, 12.5]
    max_rma_prod_rate = 7e-3
    rt_rate = 1
    freq = 1/72
    noise_std = 0
    t0 = 0
    t1 = 504
    fs = 2
    def plot_deg_rate_sweep():
        for half_life in rma_half_life:
            deg_rate = jnp.log(2) / half_life
            model = ForceRMA(max_rma_prod_rate, rt_rate, deg_rate, freq)

            solution = model.simulate(
                t0=t0,
                t1=t1,
                dt0=0.1,
                y0=(max_rma_prod_rate/rt_rate, max_rma_prod_rate/deg_rate),
                saveat=SaveAt(ts=jnp.linspace(t0, t1, t1*fs)),
                stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
                max_steps=10000
            )

            plt.plot(solution.ts, solution.ys[1]/(max_rma_prod_rate/deg_rate), label=f"{half_life}")

    #fig, ax = plt.subplots(figsize=(10, 6))
    plot_deg_rate_sweep()
    plt.xlabel("Time (hr)")
    plt.ylabel("Normalized Plasma RMA (A.U.)")
    plt.legend(frameon=False, title="RMA Half-Life (hr)", bbox_to_anchor=(1,1))
    plt.tight_layout()
    sb.despine()

    plt.savefig("notebooks/data/temporal_resolution/normalized_rma_varying_deg_rate.svg")
    plt.gca()
    return fs, max_rma_prod_rate, rma_half_life, rt_rate, t0, t1


@app.cell
def _(
    ForceRMA,
    PIDController,
    SaveAt,
    fs,
    jnp,
    max_rma_prod_rate,
    plt,
    rma_half_life,
    rt_rate,
    sb,
    t0,
    t1,
):
    # dynamic range with varying oscillation frequencies and RMA half life
    freqs = [1/72, 1/48, 1/24, 1/12, 1/6]
    shapes = ["s", "^", "d", "o"]
    def plot_freq_sweep():
        for _i, half_life in enumerate(rma_half_life):
            ranges = []
            for freq in freqs:
                deg_rate = jnp.log(2) / half_life
                model = ForceRMA(max_rma_prod_rate, rt_rate, deg_rate, freq)

                solution = model.simulate(
                    t0=t0,
                    t1=t1,
                    dt0=0.1,
                    y0=(max_rma_prod_rate/rt_rate, max_rma_prod_rate/deg_rate),
                    saveat=SaveAt(ts=jnp.linspace(t0, t1, t1*fs)),
                    stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
                    max_steps=10000
                )

                norm_plasma_rma = solution.ys[1] / (max_rma_prod_rate/deg_rate)

                max_nm = jnp.max(norm_plasma_rma)
                min_nm = jnp.min(norm_plasma_rma)
                ranges.append(max_nm - min_nm)

            plt.plot(freqs, ranges, label=f"{half_life}", marker=shapes[_i])

    plot_freq_sweep()
    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Dynamic Range (A.U.)")
    plt.legend(frameon=False, title="RMA Half-Life (hr)")
    plt.tight_layout()
    sb.despine()
    plt.savefig("notebooks/data/temporal_resolution/dynamic_range_varying_deg_and_freq.svg")
    plt.gca()
    return (shapes,)


@app.cell
def _():
    # sanity check applying noise
    return


@app.cell
def _(ForceRMA, PIDController, SaveAt, data_dir, jnp, os, plt, random):
     # sanity check applying noise
    solutions = []
    test_noise_stds = [0, 0.2]

    def noise_sweep():
        key = random.key(4142)
        for sigma in test_noise_stds:
            key, subkey = random.split(key)
            model = ForceRMA(7e-3, 1, 7e-3, freq=1/100)
            solution = model.simulate(
                t0=0,
                t1=504,
                dt0=0.1,
                y0=(0, 0),
                saveat=SaveAt(ts=jnp.linspace(0, 504, 504*2)),
                stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
                max_steps=10000
            )

            if sigma > 0:
                plasma_rma = model._apply_noise(solution.ys[1], sigma, subkey)
            else:
                plasma_rma = solution.ys[1]

            solutions.append(plasma_rma)

        return solutions

    solutions = noise_sweep()
    plt.plot(jnp.linspace(0, 504, 504*2), solutions[0])

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "example_plasma_rma.svg"))
    plt.gca()
    return (solutions,)


@app.cell
def _(data_dir, jnp, os, plt, solutions):
    t = jnp.linspace(0, 504, 504*2)
    plt.plot(t, solutions[1], color='lightgrey', label="$\\sigma$ = 0.2")
    plt.plot(t, solutions[0], color='k', label="Deterministic")

    plt.xlabel("Time (hr)")
    plt.ylabel("Plasma RMA (nM)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "example_noisy_plasma_rma.svg"))
    plt.gca()
    return


@app.cell
def _(
    ForceRMA,
    PIDController,
    SaveAt,
    fs,
    jnp,
    max_rma_prod_rate,
    plt,
    rt_rate,
    t0,
    t1,
    welch,
):
    # determine temporal resolution
    deg_rate = 7e-3 # ~ 100 hr half life
    target_freq = 1/120

    model = ForceRMA(max_rma_prod_rate, rt_rate, deg_rate, target_freq)
    solution = model.simulate(
        t0=0,
        t1=t1,
        dt0=0.1,
        y0=(max_rma_prod_rate/rt_rate, max_rma_prod_rate/deg_rate),
        saveat=SaveAt(ts=jnp.linspace(t0, t1, t1*fs)),
        stepsize_controller=PIDController(atol=1e-5, rtol=1e-5),
        max_steps=10000
    )

    norm_plasma_rma = solution.ys[1] / (max_rma_prod_rate/deg_rate)

    fxx, psd = welch(norm_plasma_rma, fs=fs)
    fpeak = fxx[jnp.argmax(psd)]
    freq_match = jnp.isclose(fpeak, target_freq, rtol=0.05)
    psd_noise = jnp.where(~jnp.isclose(fxx, target_freq, 0.05), psd, jnp.nan)
    snr = fpeak / jnp.nanmean(psd_noise)

    plt.plot(fxx, psd)
    return fpeak, freq_match, snr, target_freq


@app.cell
def _(fpeak, freq_match, snr, target_freq):
    print(f"SNR: {snr}")
    print(f"Recovered Frequency: {fpeak}")
    print(f"Input frequency: {target_freq}")
    print(f"Recovered: {freq_match}")
    return


@app.cell
def _(mo):
    button = mo.ui.run_button()
    button
    return (button,)


@app.cell
def _(
    ForceRMA,
    PIDController,
    SaveAt,
    button,
    fs,
    jnp,
    mo,
    pl,
    product,
    randint,
    random,
    t0,
    t1,
):
    mo.stop(not button.value)
    #target_freqs = jnp.array([1/120, 1/96, 1/72, 1/48, 1/24, 1/12, 1/6, 1/3])
    #target_freqs = jnp.array([1/24, 1/12, 1/6, 1/3])
    target_freqs = jnp.linspace(1/30, 1/3, 24)

    #noise_levels = jnp.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    noise_levels = jnp.linspace(0.01, 0.1, 10)
    #noise_levels = jnp.array([0.0])

    rma_prod_rate = 5.6e-2
    rma_rt_rate = 1
    rma_deg_rate = 7e-3
    _t0 = 0
    _t1 = 504
    _fs = 2
    _n_iter = 2000
    _rtol = 0.05
    _min_snr = 2
    _n_cycles = 10

    sim_config = {
        "t0": 0,
        "t1": 504,
        "dt0": 0.1,
        "y0": (rma_prod_rate/rma_rt_rate, rma_prod_rate/rma_deg_rate),
        "saveat": SaveAt(ts=jnp.linspace(t0, t1, t1*fs)),
        "stepsize_controller": PIDController(atol=1e-5, rtol=1e-5)
    }


    _resolutions = []
    _noise = []
    _freqs = []

    for _freq, _noise_level in product(target_freqs, noise_levels):
        t1_i = _n_cycles / _freq
        sim_config["t1"] = t1_i 
        sim_config["saveat"] = SaveAt(ts=jnp.linspace(t0, t1_i, int(t1_i * _fs)))
        _model = ForceRMA(rma_prod_rate, rma_rt_rate, rma_deg_rate, _freq)
        tres = _model.max_temporal_resolution(
            sim_config,
            _noise_level,
            random.key(randint(0, 2**32 - 1)),
            _freq,
            _n_iter,
            _fs,
            _rtol,
            _min_snr
        )

        _resolutions.append(tres)
        _noise.append(_noise_level)
        _freqs.append(_freq)

    tres_df = pl.DataFrame({
        "Frequency": _freqs,
        "Noise": _noise,
        "Resolution": _resolutions
    })

    tres_df.write_parquet(f"notebooks/data/temporal_resolution/tres_prod_{rma_prod_rate}.deg_{rma_deg_rate}.parquet")

    return noise_levels, target_freqs, tres_df


@app.cell
def _(tres_df):
    tres_df
    return


@app.cell
def _(pl):
    def calculate_cutoff(tres_df, freq_values, noise_groups):
        cutoff_frequencies = []
        for group in noise_groups:
            group_res = tres_df.filter(pl.col("Noise") == group)
            cutoff_freq = group_res.filter(pl.col("Resolution") >= 0.95).sort("Frequency")["Frequency"].last()

            if cutoff_freq == None:
                cutoff_frequencies.append(0)
            else:
                cutoff_frequencies.append(cutoff_freq)

        return cutoff_frequencies

    return (calculate_cutoff,)


@app.cell
def _(noise_levels, pl, plt, sb, tres_df):
    #cutoff_freqs = calculate_cutoff(tres_df, target_freqs, noise_levels)
    for _n in noise_levels:
        test = tres_df.filter(pl.col("Noise") == _n)
        plt.plot(test["Frequency"], test["Resolution"], marker='o', markersize=6)

    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Resolution")
    plt.tight_layout()
    sb.despine()
    plt.gca()
    return


@app.cell
def _(calculate_cutoff, noise_levels, target_freqs, tres_df):
    cutoff = calculate_cutoff(tres_df, target_freqs, noise_levels)
    return (cutoff,)


@app.cell
def _(cutoff, noise_levels, plt):
    plt.plot([_n * 100 for _n in noise_levels], cutoff, 'o')
    plt.xlabel("% Noise")
    plt.ylabel("Frequency (1/hr)")
    return


@app.cell
def _(data_dir, os, pl, plt, sb, shapes):

    degs = ["0.007", "0.014", "0.028", "0.056"]
    for _i, _deg in enumerate(degs):
        tres_i = pl.read_parquet(os.path.join(data_dir, f"tres_prod_0.007.deg_{_deg}.parquet"))
        res = tres_i.filter(pl.col("Noise") == 0.05)
        plt.plot(res["Frequency"], res["Resolution"], marker=shapes[_i], markersize=6)

    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Temporal Resolution (%)")
    plt.legend(["100", "50", "25", "12.5"], title="RMA Half-Life (hr)", frameon=False)
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "tres_half_life_sweep.svg"))
    plt.gca()

    return (degs,)


@app.cell
def _(data_dir, degs, os, pl, plt, sb):
    for _val in degs:
        _tres_i = pl.read_parquet(os.path.join(data_dir, f"tres_prod_{_val}.deg_0.007.parquet"))
        _res = _tres_i.filter(pl.col("Noise") == 0.05)
        plt.plot(_res["Frequency"], _res["Resolution"], marker='o', markersize=6)

    plt.xlabel("Frequency (1/hr)")
    plt.ylabel("Temporal Resolution (%)")
    plt.legend(["7.0e-3", "1.4e-2", "2.8e-2", "5.6e-2"], title="RMA production (nM/hr)", frameon=False)
    sb.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "tres_prod_sweep.svg"))
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
