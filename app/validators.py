def time(t0: float, t1: float):
    """
    Check simulation times
    """
    if t0 > t1:
        raise ValueError(f"Error: start time must be less than stop time. Got ({t0}, {t1})")
