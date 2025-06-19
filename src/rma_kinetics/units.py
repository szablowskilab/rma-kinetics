from equinox import Enumeration as EqxEnum


class Time(EqxEnum):
    """
    A time unit enumeration with the following variants:

    - hr
    - min
    - sec
    """
    hours = "hr"
    minutes = "min"
    seconds = "sec"


class Concentration(EqxEnum):
    """
    A concentration unit enumeration with the following variants:

    - nM
    - µM
    - mM
    """
    nanomolar = "nM"
    micromolar = "µM"
    millimolar = "mM"
