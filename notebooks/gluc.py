import polars as pl

def rlu_to_nm(rlu: float, vmax=168669125, km=227.2) -> float:
    """
    Convert RLU to Gluc concentration (nM) based on MM standard curve.

    Parameters
    ----------

    rlu : float
        RLU value

    vmax : float
        MM Vmax value (Optional, default = 168669125).

    km : float
        MM Km value (Optional, default = 227.2).

    Vmax and Km values come from Seo et al., 2024 supplemental
    figure 6.
    """
    return rlu * km / vmax

def get_gluc_conc(df: pl.DataFrame, vmax=168669125, km=227.2):
    """
    Calculate mean and standard deviation of Gluc concentrations
    from RLU values.

    Parameters
    ----------
    df : pl.DataFrame
        polars dataframe containing RLU values.
    vmax : float
        Vmax value for use in the rlu_to_nm function
    km : float
        Km value for use in the rlu_to_nm function
    """
    return df.with_columns(
        pl.col("rlu").map_elements(rlu_to_nm, return_dtype=pl.Float64).alias("gluc")
    )
