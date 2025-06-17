from equinox import Module as EqxModule
from jaxtyping import PyTree
from dataclasses import field


CNO_MW = 342.8 # g/mol

class CnoPKConfig(EqxModule):
    cno_dose: float
    cno_absorption_rate: float
    cno_elimination_rate: float
    cno_reverse_metabolism_rate: float
    clz_metabolism_rate: float
    cno_brain_transport_rate: float
    cno_plasma_transport_rate: float
    clz_brain_transport_rate: float
    clz_plasma_transport_rate: float
    clz_elimination_rate: float
    cno_plasma_vd: float
    cno_brain_vd: float
    clz_plasma_vd: float
    clz_brain_vd: float
    cno_nmol: float = field(init=False)

    def __post_init__(self):
        # convert cno_dose from mg to nmol
        self.cno_nmol = self.cno_dose / CNO_MW * 1e6


class CnoPK(EqxModule):
    config: CnoPKConfig

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)

        raise AttributeError(name)

    def _model(self, t: float, y: PyTree[float], args=None) -> PyTree[float]:
        peritoneal_cno, plasma_cno, brain_cno, plasma_clz, brain_clz = y

        peritoneal_cno_flux = self.cno_absorption_rate * peritoneal_cno
        brain_cno_influx = self.cno_brain_transport_rate * plasma_cno
        brain_cno_outflux = self.cno_plasma_transport_rate * brain_cno
        plasma_clz_influx = self.cno_reverse_metabolism_rate * plasma_cno
        plasma_clz_outflux = self.clz_metabolism_rate * plasma_clz
        brain_clz_influx = self.clz_brain_transport_rate * plasma_clz
        brain_clz_outflux = self.clz_plasma_transport_rate * brain_clz

        dperitoneal_cno = -peritoneal_cno_flux

        dplasma_cno = (peritoneal_cno_flux - (self.cno_elimination_rate * plasma_cno)
                       - brain_cno_influx + brain_cno_outflux
                       - plasma_clz_influx + plasma_clz_outflux)
        dbrain_cno = brain_cno_influx - brain_cno_outflux

        dplasma_clz = (plasma_clz_influx - plasma_clz_outflux - (self.clz_elimination_rate * plasma_clz)
                       - brain_clz_influx + brain_clz_outflux)
        dbrain_clz = brain_clz_influx - brain_clz_outflux

        return dperitoneal_cno, dplasma_cno, dbrain_cno, dplasma_clz, dbrain_clz
