# Ping Pong Rating Research

Comparing ELO variants on real table tennis match data to determine the best
predictive rating algorithm for a community-first ranked ping pong app.

---

## Motivation

This research is the foundation for a community table tennis app built for
local players in Tunisia. Before choosing a rating algorithm, we tested
multiple ELO variants on real match data to find which one best predicts
match outcomes — not just who wins, but how confidently.

---

## Research Question

Which ELO variant best predicts the outcome of a table tennis match —
Standard ELO, Margin of Victory ELO (set-based), Margin of Victory ELO
(points-based), or Glicko-2?

---

## Dataset

**One Month Table Tennis Dataset** by medaxone on Kaggle  
One month of Setka Cup matches including set scores and individual game points.

> Dataset is not included in this repo. Download it from Kaggle and place
> `Setka.csv` in the `/data` folder.

---

## Algorithms Tested

| Algorithm | Key Idea |
|---|---|
| Standard ELO | Baseline — win/loss only, K=32 |
| MOV ELO (Set-based) | Multiplier based on set differential |
| MOV ELO (Points-based) | Multiplier based on total point differential |
| Glicko-2 | Adds rating deviation and volatility |

---

## Evaluation Metrics

- **Accuracy** — did the higher-rated player win?
- **Brier Score** — how close were the predicted probabilities to reality?
- **Log-loss** — how well calibrated are the predictions overall?

---

## Project Structure
pingpong-rating-research/

├── notebooks/

│   ├── 01_exploration.ipynb

│   ├── 02_elo_standard.ipynb

│   ├── 03_elo_set_based.ipynb

│   ├── 04_elo_points_based.ipynb

│   ├── 05_glicko2.ipynb

│   └── 06_comparison.ipynb

├── src/

│   └── utils.py

├── data/          ← not tracked by git

├── results/       ← saved prediction CSVs per algorithm

└── README.md

---

## Status

🔵 In Progress

- [x] Project setup and data loading
- [x] Standard ELO baseline
- [ ] MOV ELO set-based
- [ ] MOV ELO points-based
- [ ] Glicko-2
- [ ] Comparison and findings

---

## Findings

*To be updated once all algorithms are evaluated.*

---

## Related

This research directly informs the algorithm choice for **[app name]**,
a community ping pong ranking app for local players.