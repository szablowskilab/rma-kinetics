# Tet-Off RMA

Tetracycline transcriptional activator (tTA) induced RMA expression model.

::: rma_kinetics.models.TetRMA

## Example

## Model Equations

Note that this model assumes constitutive expression of tTA.

$$\begin{align}
\dot{[TA]} &= k_{TA} - \gamma_{TA}[TA] \tag{1} \\
[TA]_{SS} &= \frac{k_{TA}}{\gamma_{TA}} \tag{2}
\end{align}
$$

Doxycycline is the preferred inhibitor (although tetracycline or other
derivatives may be used by updating the [DoxPKConfig](./dox/config.md)).
The fraction of the transcriptional activator available for inducing RMA
expression is then modeled with a Hill function,

$$\begin{align}
\theta_{tTA} &= \frac{1}{1 + \frac{[Dox]}{K_{D_{Dox}}}} \tag{3} \\
\dot{[RMA_{B}]} &= \frac{\beta_{0_{RMA}} + \beta_{RMA}\left(\frac{\theta_{TA}[TA]}{K_{D_{TA}}}\right)^{n_{tTA}}}{1 + \left(\frac{\theta_{TA}[TA]}{K_{D_{TA}}}\right)^{n_{tTA}}} - k_{RT}[RMA_{B}] \tag{4} \\
\dot{[RMA_{P}]} &= k_{RT}[RMA_{B}] - \gamma_{RMA}[RMA_{P}] \tag{5}
\end{align}
$$
