# PK Model

A two-compartment CNO/CLZ model.

$$\begin{align}
\dot{[CNO_{D}]} &= -k_{a_{CNO}}[CNO_{D}] \tag{1} \\
\dot{[CNO_{P}]} &= k_{a_{CNO}}[CNO_{D}] - k_{el_{CNO}}[CNO_{P}] - k_{RMet_{CNO}}[CNO_{P}]  \\
&+ k_{Met_{CLZ}}[CLZ_{P}] - k_{PB_{CNO}}[CNO_{P}] + k_{BP_{CNO}}[CNO_{B}] \tag{2} \\
\dot{[CNO_{B}]} &= k_{PB_{CNO}}[CNO_{P}] - k_{BP_{CNO}}[CNO_{B}] \tag{3} \\
\dot{[CLZ_{P}]} &= k_{RMet_{CNO}}[CNO_{P}] - k_{el_{CLZ}}[CLZ_{P}] \\
&- k_{Met_{CLZ}}[CLZ_{P}] - k_{PB_{CLZ}}[CLZ_{P}] + k_{BP_{CLZ}}[CLZ_{B}] \tag{4} \\
\dot{[CLZ_{B}]} &= k_{PB_{CLZ}}[CLZ_{P}] - k_{BP_{CLZ}}[CLZ_{B}] \tag{5}
\end{align}
$$

See [CnoPKConfig](./config.md) for parameter details.

:::rma_kinetics.models.CnoPK
