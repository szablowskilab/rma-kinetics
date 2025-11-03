import polars as pl

def rlu_to_nm_gluc(rlu: float, vmax=168669125, km=227.2) -> float:
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

def rlu_to_nm_rma(rlu) -> float:
    """
    Convert RLU to RMA concentration (nM) based on Lee et al., 2024

    Parameters
    ----------

    rlu : float
        RLU value
    """

    ng_ml = rlu * 0.0012 + 26.96
    return ng_ml / 44

def get_gluc_conc(df: pl.DataFrame, reporter: str):
    """
    Calculate mean and standard deviation of Gluc concentrations
    from RLU values.

    Parameters
    ----------
    df : pl.DataFrame
        polars dataframe containing RLU values.
    reporter : 'gluc' | 'rma'
        reporter type to use for conversion (either MM based gluc or linear RMA)
    """

    if reporter == 'gluc':
        return df.with_columns(
            pl.col("rlu").map_elements(rlu_to_nm_gluc, return_dtype=pl.Float64).alias("gluc")
        )
    elif reporter == 'rma':
        return df.with_columns(
            pl.col("rlu").map_elements(rlu_to_nm_rma, return_dtype=pl.Float64).alias("gluc")
        )
