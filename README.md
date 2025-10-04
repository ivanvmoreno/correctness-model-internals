# Predicting Correctness from Model Internals

Do Large Language Models (LLMs) Anticipate When They Will Answer Correctly?

To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model’s forthcoming answer will be correct.

Across three open‐source model families ranging from 7 to 70 billion parameters, projections on such "self‐correctness direction” learned on generic trivia questions predict success in-distribution and on diverse out‐of‐distribution knowledge datasets, outperforming black-box baselines and verbalised predicted confidence.

Predictive power saturates in intermediate layers, suggesting that self‐assessment emerges mid‐computation. Notably, generalisation falters on questions requiring mathematical reasoning.

Moreover, for models responding “I don’t know”, doing so strongly correlates with the probe score, indicating that the same direction also captures confidence.

By complementing findings on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes to elucidating LLM internals and is a proof-of-concept for a lightweight anticipatory tool that augments existing confidence estimators.

---

All datasets, experiment data and results are available for download at [Academic Torrents](https://academictorrents.com/details/011a9b941bd460d219e563eb5eccc470aadd8f20).
