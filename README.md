# Can We Predict Who Wins a UFC Fight?

A binary classifier (Red corner vs. Blue corner) built on pre-fight stats only: career averages, physical attributes, streaks, rankings. Nothing from during or after the fight enters the feature set.

## The question

Two baselines set the bar before any model gets credit for predicting anything.

**Always pick Red.** Red corner wins about 58% of the time historically, so corner assignment isn't random.

**Always pick the betting favorite.** Sportsbooks already price in most of what a model would have to learn from scratch.

A model that can't beat the second baseline isn't telling us anything Vegas doesn't already know.

## What I found

| Method | Accuracy | ROC-AUC |
|---|---|---|
| Always predict Red | 56.3% | — |
| Logistic Regression | 60.5% | 0.629 |
| Gradient Boosting | 62.1% | 0.655 |
| Random Forest | 63.9% | 0.675 |
| **Always predict the betting favorite** | **69.9%** | — |

Pre-fight stats beat doing nothing. They lose to the market. A second experiment hands the model the betting odds directly and pushes Random Forest to 70.2%, barely past the favorite baseline's 69.6% on that same subset. The odds carry almost all the predictive signal; fighter stats add very little on top.

Career stats and physical differentials give a real edge over a coin flip. Vegas still knows more than this model does. A model that beat the market using only public career stats would be the surprising result here, not this one.

## Data

[Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) (Kaggle): fight-level data from 2010 to present, fighter physical attributes, career stats, and betting odds. Not included in this repo (~3 MB); see Setup below.

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

**Leakage check, done empirically, not assumed.** Columns counting career wins-by-method looked pre-fight but weren't. Average `R_win_by_KO/TKO` is measurably higher for fights Red actually won by KO than otherwise, which proves the counter includes the current fight's own result. I dropped these columns and left the proof in the notebook.

**Chronological split, not random.** The model trains on earlier fights and gets tested on the most recent 20%. A random split would let it see the future.

**Betting odds held out of the primary model.** I used them only as a baseline and benchmark, plus one clearly separated experiment testing how much they help when included.

## Ideas for follow-up

- Predict method of victory (KO/TKO vs. submission vs. decision) instead of just the winner. Harder target, more interesting result.
- Try XGBoost.
- Split by event date rather than row count, so no fight card gets split across train and test.
- Check model calibration, not just accuracy.
