# STAT 425 — Final Project Proposal

**Title:** Betting the Slate: Bayesian Portfolio Optimization for Studio Film Strategy

**Team:** Theo Auyeung · NetID: [INSERT]

---

## Question

What is the optimal annual film slate composition for a movie studio — across genre, budget tier, IP type, and release season — to maximize expected portfolio ROI while managing downside risk? And how should a studio update its greenlight decisions across years as film performance is observed?

This question matters because studios routinely treat film investment decisions in isolation, greenlighting projects one at a time rather than reasoning about their slate as a correlated portfolio. Bayesian inference lets us characterize the full posterior distribution over returns per film type — not just average historical ROI — and use that uncertainty directly in the optimization. The result is a framework analogous to Markowitz portfolio theory but grounded in the specific structure of the film industry.

---

## Dataset

**Primary source:** The Movies Dataset (Kaggle: `rounakbanik/the-movies-dataset`) — ~45,000 films with production budget, revenue, genres, release date, production companies, runtime, keywords, cast/crew, and 26M user ratings from MovieLens. Coverage: 1900s–2017.

**Files used:**
- `movies_metadata.csv` — budget, revenue, genres, release date, production companies, runtime, vote metrics
- `keywords.csv` — keyword tags per film (joined on movie ID), used for IP type classification
- `credits.csv` — cast and crew (joined on movie ID), used for director-level features if needed

**Note on coverage:** Budget/revenue fields have partial coverage (`budget = 0` indicates missing). After filtering to films with valid budget and revenue, the working dataset is estimated at 8,000–12,000 films — well-suited for hierarchical inference with meaningful data per genre × season cell. Data runs through 2017, so post-streaming-war era (Disney+, HBO Max) is a noted limitation.

**Key variables:**
- Response: ROI = (revenue − budget) / budget — derived directly from dataset fields
- Genre — parsed from JSON `genres` field; primary genre used for grouping
- Budget tier (micro <$10M, indie $10–40M, mid $40–100M, tentpole $100–200M, mega >$200M) — derived from `budget`
- IP type (original, sequel, franchise entry, adaptation/remake) — derived from `keywords.csv` tags and title pattern matching
- Release season (Winter, Spring, Summer, Fall, Awards window) — derived from `release_date`
- Studio — parsed from JSON `production_companies` field; primary company used
- Quality control: `vote_average` and `vote_count` from metadata

---

## Bayesian Approach

**Model:** Bayesian hierarchical regression on ROI with partial pooling across genre, budget tier, IP type, release season, and studio:

```
ROI_i ~ Normal(μ_i, σ²)
μ_i = β_global + α_genre[g] + α_budget[b] + α_IP[k]
    + α_season_genre[s, g] + α_studio[u]

α_genre[g]        ~ Normal(μ_genre, σ_genre²)
α_studio[u]       ~ Normal(μ_studio, σ_studio²)
α_season_genre[s,g] ~ Normal(μ_sg, σ_sg²)
```

Partial pooling is the core contribution: rare genre × season combinations borrow strength from adjacent cells rather than overfitting on thin data. Inference via MCMC (NUTS) in PyMC.

**Portfolio layer:** Monte Carlo draws from the joint posterior simulate portfolio-level expected return and variance for any given slate composition, producing an efficient frontier with uncertainty bands.

**Sequential decision:** A simplified two-stage update — after observing Year 1 film performance, the posterior on sequel/franchise ROI is revised via Bayesian updating to re-optimize the Year 2 slate. This mirrors how studios actually make greenlight decisions dynamically.

**Model comparison:** WAIC across nested models to test whether IP type, release timing, and studio identity each credibly improve fit beyond genre + budget alone.

**Prior assumptions:** Weakly informative Normal priors on fixed effects (centered at 0, SD = 5–10 ROI units); HalfNormal on variance components. Prior sensitivity analysis on the genre-level hyperprior SD.

---

## Plan

| Week | Work |
|---|---|
| Nov 3–9 | Pull and clean The Numbers + Box Office Mojo data; define ROI; classify budget tier, IP type, release season |
| Nov 10–16 | Fit M1 (baseline) and M2 (full hierarchical) in PyMC; initial posterior summaries; submit proposal |
| Nov 17–23 | Portfolio simulation layer; efficient frontier; WAIC model comparison |
| Nov 24–30 | Sequential Year 1 → Year 2 update; R visualizations; presentation prep |
| Dec 1–4 | Final report write-up; code cleanup; AI-use record |
