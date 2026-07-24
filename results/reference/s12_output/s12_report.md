# S12 Report — R/F-Bifurkationsuniversality
Datum: 30.05.2026 | FUT/CoMath Simulation S12

## Kernfrage

S12 prüft, ob die S11-A-Formel

```text
r*(alpha) = 1 / (1 - alpha)
```

unter Netzwerk-Rekohärenz, unterschiedlichen J-Funktionen und aktiver Dämpfung erhalten bleibt.
Die getestete Erweiterung lautet:

```text
r*(alpha,beta) = (1 + beta) / (1 - alpha + beta)
```

## Setup

- Knoten: 12
- Seeds: 20
- Schritte: 500
- J-Funktionen: linf_exp, power, metric_exp
- alpha-Werte: 14
- beta-Werte: 7
- Memory-Cap: 50.0

## Schnellbefund

Für J=linf_exp liegen 98 von 98 Parameterpunkten im Bereich `analytic_tracking` oder `stable_horizon`.
Saturierte Memory-Regime: 0 von 98.

Bestes Spektrum bei beta=0.5:

```text
J=metric_exp, alpha=0.00000, beta=0.50, Ico=0.43294
```

## alpha_em-Schnitt

| J | beta | r_theory | r_sim | abs_error | class |
|---|------|----------|-------|-----------|-------|
| linf_exp | 0.00 | 1.00735 | 1.00735 | 0.00000 | stable_horizon |
| linf_exp | 0.05 | 1.00700 | 1.00700 | 0.00000 | stable_horizon |
| linf_exp | 0.10 | 1.00668 | 1.00668 | 0.00000 | stable_horizon |
| linf_exp | 0.25 | 1.00587 | 1.00587 | 0.00000 | stable_horizon |
| linf_exp | 0.50 | 1.00489 | 1.00489 | 0.00000 | stable_horizon |
| linf_exp | 1.00 | 1.00366 | 1.00366 | 0.00000 | stable_horizon |
| linf_exp | 2.00 | 1.00244 | 1.00244 | 0.00000 | stable_horizon |
| power | 0.00 | 1.00735 | 1.00735 | 0.00000 | stable_horizon |
| power | 0.05 | 1.00700 | 1.00700 | 0.00000 | stable_horizon |
| power | 0.10 | 1.00668 | 1.00668 | 0.00000 | stable_horizon |
| power | 0.25 | 1.00587 | 1.00587 | 0.00000 | stable_horizon |
| power | 0.50 | 1.00489 | 1.00489 | 0.00000 | stable_horizon |
| power | 1.00 | 1.00366 | 1.00366 | 0.00000 | stable_horizon |
| power | 2.00 | 1.00244 | 1.00244 | 0.00000 | stable_horizon |
| metric_exp | 0.00 | 1.00735 | 1.00735 | 0.00000 | stable_horizon |
| metric_exp | 0.05 | 1.00700 | 1.00700 | 0.00000 | stable_horizon |
| metric_exp | 0.10 | 1.00668 | 1.00668 | 0.00000 | stable_horizon |
| metric_exp | 0.25 | 1.00587 | 1.00587 | 0.00000 | stable_horizon |
| metric_exp | 0.50 | 1.00489 | 1.00489 | 0.00000 | stable_horizon |
| metric_exp | 1.00 | 1.00366 | 1.00366 | 0.00000 | stable_horizon |
| metric_exp | 2.00 | 1.00244 | 1.00244 | 0.00000 | stable_horizon |

## Interpretation

1. Wenn beta=0, testet S12 direkt die S11-A-Bifurkation.
2. Wenn beta>0, wird geprüft, ob R/F=1 als stabilisierbarer Horizont aus einer allgemeinen Gegenkopplung folgt.
3. Wenn alpha nahe alpha_em liegt, muss r* nahe 1 bleiben. Das ist die entscheidende Brücke zur FUT-Deutung des Elektron-Horizonts.
4. Wenn hohe alpha-Werte saturieren, ist das kein Fehler, sondern die numerische Signatur rekursiver Selbstverstärkung oberhalb des stabilen Fortsetzungsregimes.

## Dateien

- `s12_rf_bifurcation_universality.csv`
- `s12_a_bifurcation_curves.png`
- `s12_b_error_heatmap.png`
- `s12_c_ico_vs_alpha.png`

*S12 abgeschlossen.*
