# Can We Predict Who Wins a UFC Fight?

A binary classifier (Red corner vs. Blue corner) built on pre-fight stats only — career averages, physical attributes, streaks, rankings. No information from during or after the fight is used as a feature.

## The question

Two baselines set the bar before any model gets credit for "predicting" anything:

- **Always pick Red.** Red corner wins about 58% of the time historically — corner assignment isn't random.
- **Always pick the betting favorite.** Sportsbooks already price in a lot of what a model would have to learn from scratch.

If a model can't beat the second baseline, it isn't telling us anything Vegas doesn't already know.

## What I found

| Method | Accuracy | ROC-AUC |
|---|---|---|
| Always predict Red | 56.3% | — |
| Logistic Regression | 60.5% | 0.629 |
| Gradient Boosting | 62.1% | 0.655 |
| Random Forest | 63.9% | 0.675 |
| **Always predict the betting favorite** | **69.9%** | — |

Pre-fight stats beat doing nothing, but lose to the market. A second experiment that hands the model the betting odds directly pushes Random Forest to 70.2% — barely past the favorite baseline (69.6% on that subset), meaning the odds alone carry almost all the predictive signal; the fighter stats add very little on top.

**Takeaway:** career stats and physical differentials give a real edge over a coin flip, but Vegas still knows more than this model does. That's the honest result, not a setback — a model that beat the market using only public career stats would be the surprising (and suspicious) outcome.

## Data

[Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) (Kaggle) — fight-level data from 2010 to present, fighter physical attributes, career stats, and betting odds. Not included in this repo (~3 MB); see Setup below.

## Repo structure

```
ufc_winner_prediction.ipynb   # the analysis
requirements.txt              # pinned dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Download `ufc-master.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) and place it in the repo root, then run the notebook top to bottom.

## Method notes

- **Leakage check, done empirically, not assumed.** Columns counting career wins-by-method looked pre-fight but weren't — average `R_win_by_KO/TKO` is measurably higher for fights Red actually won by KO than otherwise, proving the counter includes the current fight's own result. Dropped, with the proof left in the notebook.
- **Chronological split**, not random — trained on earlier fights, tested on the most recent ~20%. A random split would let the model "see the future."
- **Betting odds held out of the primary model** and used only as a baseline/benchmark, plus one clearly separated experiment testing how much they help if included.

## Ideas for follow-up

- Predict method of victory (KO/TKO vs. submission vs. decision) instead of just the winner — a harder, more interesting target.
- Try XGBoost.
- Split by event date rather than row count, so no fight card is split across train/test.
- Check model calibration, not just accuracy.
