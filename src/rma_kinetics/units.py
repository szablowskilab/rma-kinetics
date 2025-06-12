from equinox import Enumeration as EqxEnum


class Time(EqxEnum):
    hours = "hr"
    minutes = "min"
    seconds = "sec"


class Concentration(EqxEnum):
    micromolar = "µM"
    nanomolar = "nM"
    millimolar = "mM"
