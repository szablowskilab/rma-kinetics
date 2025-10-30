# PK Model

A two-compartment model for doxycycline administration within a given time interval.

$$\begin{align}
I(t) &= \begin{cases}
f_{intake} \frac{F_{Dox} C_{DH}}{V_D} && \tau_{0} < t< \tau_{1} \tag{1}\\
0
\end{cases} \\
\dot{[Dox_{P}]} &= k_{a_{Dox}}I(t) -k_{el_{Dox}}[Dox_{P}] - k_{PB_{Dox}}[Dox_{P}] + k_{BP_{Dox}}[Dox_{B}] \tag{2}\\
\dot{[Dox_{B}]} &= k_{PB_{Dox}}[Dox_{P}] - k_{BP_{Dox}}[Dox_{B}] \tag{3}
\end{align}
$$

See [DoxPKConfig](./config.md) for parameter details.

::: rma_kinetics.models.DoxPK
    options:
      members:
        - _intake
        - _model
