# STAT 425 — Bayesian Inference Final Project Ideas
**Rice University | Fall 2026 | Due: Proposal Nov 16 · Report Dec 4**

---

## Preference Order: Idea 2 > Idea 1

---

## Idea 2 — COVID as a Natural Experiment for Release Strategy ⭐ (Top Pick)

### The Question
COVID forced studios to release the same types of films via three strategies they'd never have chosen voluntarily: delayed theatrical, straight-to-streaming, and hybrid PVOD. This is a natural experiment. What does the posterior on each strategy's effect tell us about optimal release decisions in a post-COVID world where streaming is now a permanent option?

### Data Concern (Addressed)
The dataset is tighter than it feels but workable. 2020–2022 gives you several hundred treatment films across PVOD (Trolls World Tour, Mulan, Raya, Black Widow, Jungle Cruise), straight-to-streaming (Netflix originals, HBO Max exclusives like Zack Snyder's Justice League), and delayed theatrical (No Time to Die, Top Gun: Maverick). Scope to wide-release films with budgets >$20M to keep the comparison apples-to-apples. ~300–500 films in the window is enough for Bayesian hierarchical inference, especially with partial pooling across genres.

### Process

**Step 1 — Build the Dataset**
- Films from 2018–2023 (pre-COVID baseline + COVID window + recovery)
- Sources: The Numbers (budget, gross, release date, distributor), Box Office Mojo (release type), IMDb (user rating + rating count), Rotten Tomatoes / Metacritic (quality scores)
- Classify each film into: `normal_theatrical`, `delayed_theatrical`, `PVOD`, `straight_streaming`
- This classification is mostly available from Wikipedia release history and Box Office Mojo records
- Outcomes of interest: IMDb user rating (persistent quality signal), IMDb rating count (audience reach proxy), RT audience score

**Step 2 — Handle Selection Bias with Bayesian Propensity Scoring**
Studios didn't randomly assign films to release strategy — tentpoles delayed, smaller films went to streaming. You need to control for this non-random assignment. Build a Bayesian propensity score model:
```
P(treatment = streaming | genre, budget, studio, Metacritic_score) ~ Bayesian logistic regression
```
Use posterior propensity weights in the outcome model to adjust for selection. This is the methodologically serious piece that makes the causal inference defensible.

**Step 3 — Bayesian Causal Regression**
```
outcome_i ~ Normal(μ_i, σ²)
μ_i = α + β_treatment · treatment_i
    + β_quality · Metacritic_i
    + β_budget · log(budget_i)
    + α_genre[genre_i]           ← hierarchical, partial pooling
    + α_studio_tier[studio_i]    ← hierarchical, partial pooling

α_genre[g] ~ Normal(μ_genre, σ_genre²)   ← genres share a global mean
```
The posterior on `β_treatment` is the credible effect of going to streaming vs. theatrical, controlling for quality and genre.

**Step 4 — Heterogeneous Treatment Effects**
The average effect hides the interesting story. A $200M tentpole hurt badly by streaming; a $15M indie drama may have reached more audience. Model genre × treatment interactions:
```
β_treatment[genre] ~ Normal(μ_treatment, σ_treatment²)
```
Posterior per genre cell tells the real policy story.

**Step 5 — Posterior Predictive Policy Output**
Given a new film's genre, budget, and quality score, output the posterior distribution on each outcome under theatrical vs. streaming. This is a concrete actionable recommendation for studios.

### Bayesian Methods Covered
- Hierarchical priors with partial pooling (genre, studio tier)
- Bayesian propensity score for causal adjustment
- Posterior credible intervals on treatment effects
- Heterogeneous treatment effects via hierarchical interactions
- WAIC model comparison (with vs. without propensity weighting, with vs. without genre interactions)
- Posterior predictive checks

### Policy Takeaways
- Which genres lose most from streaming release (tentpoles, franchise entries)?
- Which genres are neutral or positive (mid-budget dramas, documentaries)?
- For studios: given a film in development at $X budget in genre G, what's the posterior on the revenue/audience gap between theatrical and streaming?

### Datasets
| Source | Contents | Access |
|---|---|---|
| The Numbers | Budget, gross, release date, distributor | free, thenumbers.com |
| Box Office Mojo | Release type, opening weekend | free |
| IMDb | User ratings + rating counts | Kaggle IMDb datasets |
| Rotten Tomatoes | Critic + audience scores | Kaggle RT dataset |
| Metacritic | Metascore | Kaggle or scrape |

---

## Idea 1 — Bayesian Studio Portfolio Optimization ⭐ (Strong Second)

### The Question
Studios don't make one film at a time — they run a multi-year slate where decisions compound. A studio commits capital across many films simultaneously, some 2–3 years from release. The question: given posterior uncertainty over ROI distributions per film type, what is the optimal annual slate composition to maximize cumulative portfolio performance? And within that, do release timing decisions (season × genre) credibly shift expected returns?

### Why This Is More Than a Regression
Treating studios like investors with a film portfolio changes the question from "what predicts a film's ROI" to "what is the optimal allocation strategy across film types given uncertainty in each type's return distribution." The Bayesian posterior is the direct input to the optimizer — the efficient frontier you output has uncertainty bands, not a single optimal answer. Studios can see how sensitive the optimal mix is to different prior beliefs about genre performance.

### The Sequential / Policy Angle
Studios face compounding decisions across years:
- A successful original in Year 1 creates a sequel option in Year 2
- Over-investing in franchises in Year 1 locks you into sequel obligations in Years 2–3
- This is structurally similar to a POMDP: current allocation decisions affect future state (your franchise commitments, market positioning, talent relationships)

For the scope of this project, model a simplified 2-year horizon with one sequential decision node. The full POMDP extension is a stretch goal but worth mentioning as future work.

### Process

**Step 1 — Build the Dataset**
- The Numbers: all wide-release films 1995–2023 with production budget, worldwide gross, studio, genre, release date
- Define ROI = (worldwide gross − budget) / budget
- Classify each film along four dimensions:
  - **Budget tier:** micro (<$10M), indie ($10–40M), mid ($40–100M), tentpole ($100–200M), mega (>$200M)
  - **IP type:** original, sequel, franchise entry, adaptation/remake
  - **Genre:** action, drama, comedy, horror, animation, sci-fi, thriller, etc.
  - **Release season:** Winter (Jan–Feb), Spring (Mar–Apr), Summer (May–Aug), Fall (Sep–Nov), Awards (Oct–Dec)
- ~8,000–12,000 films over this window, plenty of data per cell with partial pooling

**Step 2 — Bayesian Hierarchical Model on ROI**
```
ROI_i ~ Normal(μ_i, σ²)
μ_i = β_global
    + α_genre[genre_i]
    + α_budget_tier[budget_tier_i]
    + α_IP[IP_type_i]
    + α_season_genre[season_i, genre_i]      ← release timing interaction
    + α_studio[studio_i]

# Hierarchical priors (partial pooling)
α_genre[g]        ~ Normal(μ_genre, σ_genre²)
α_studio[s]       ~ Normal(μ_studio, σ_studio²)
α_season_genre[s,g] ~ Normal(μ_sg, σ_sg²)

# Hyperpriors
μ_genre   ~ Normal(0, 5)
σ_genre   ~ HalfNormal(2)
σ        ~ HalfNormal(5)
```
Partial pooling is critical: a studio with 3 horror films shrinks toward the genre-wide mean; a studio with 40 horror films trusts its own data. Rare genre × season cells borrow strength from adjacent cells.

**Step 3 — Release Timing Findings**
The `α_season_genre` term answers: for a $60M thriller, does a September release credibly outperform a January release, and by how much? Posterior credible intervals on each season × genre combination give you a direct release calendar recommendation. Expected findings:
- Summer (May–Jul) credibly boosts action/tentpole ROI
- Awards window (Oct–Nov) credibly boosts drama ROI  
- January/February is credibly negative across almost all genre × budget cells

**Step 4 — Posterior over Portfolio ROI**
For a given slate composition (e.g., 3 tentpole action films + 2 mid-budget dramas + 1 indie original), draw S samples from the joint posterior over each film's ROI distribution and compute portfolio-level expected return and variance per draw. This gives a full posterior distribution over portfolio outcomes — not a point estimate.

Account for correlation: films in the same genre releasing the same season are correlated (they compete). Model this with a covariance structure on the genre × season residuals.

**Step 5 — Efficient Frontier**
Sweep across slate compositions and plot expected portfolio ROI vs. variance. The efficient frontier tells you: for a studio with $400M annual budget and moderate risk tolerance, here's the posterior-optimal allocation. Crucially, the frontier itself has uncertainty bands — the Bayesian approach shows you not just where the frontier is but how confident you should be in it.

**Step 6 — Sequential Decision: Year 2 Greenlight**
Simple extension: given Year 1 film performance (now observed), use Bayesian updating to revise posteriors on sequel/franchise ROI and re-optimize the Year 2 slate. This is the sequential policy layer — not a full POMDP but a two-stage Bayesian decision problem that shows how the optimal slate changes as information arrives.

**Step 7 — Compare Actual vs. Optimal**
For each major studio (Disney, Universal, A24, Lionsgate, Netflix), compare their actual historical slate composition to the posterior-optimal efficient frontier. Who is most efficiently allocated? Who is taking excess risk without commensurate return?

### Bayesian Methods Covered
- Bayesian hierarchical regression with multi-level partial pooling
- Interaction effects with hierarchical priors (season × genre)
- Posterior Monte Carlo simulation for portfolio-level statistics
- Correlated residuals / multivariate normal extension
- WAIC model comparison (does IP type matter? does release timing matter beyond genre?)
- Bayesian updating for the sequential Year 1 → Year 2 decision
- Posterior predictive for specific film + slate recommendations

### Policy Takeaways
- **For studios:** What slate composition maximizes expected ROI for a given budget and risk tolerance? Where does your current strategy sit on the efficient frontier?
- **For release strategy:** Which genre × season combinations are credibly high-value vs. over-crowded?
- **For franchise decisions:** Given Year 1 performance, what does the posterior say about whether to greenlight the sequel?
- **For investors / streamers:** Which studio's portfolio strategy has the best posterior risk-return profile?

### Datasets
| Source | Contents | Access |
|---|---|---|
| The Movies Dataset | ~45k films; budget, revenue, genres, release date, production companies, keywords, credits, 26M user ratings | Kaggle: `rounakbanik/the-movies-dataset` |
| IMDb (supplementary) | Ratings for quality control | Kaggle |

**Files:** `movies_metadata.csv`, `keywords.csv` (joined on movie ID), `credits.csv` (joined on movie ID)

**Note:** Budget/revenue fields have partial coverage — filter to films with non-zero budget AND revenue, leaving ~8,000–12,000 usable films. Coverage runs through 2017; post-streaming-war era is a noted limitation.

---

## Summary Comparison

| | Idea 2 (COVID Natural Experiment) | Idea 1 (Studio Portfolio Optimization) |
|---|---|---|
| **Data scope** | Tight but sufficient (~400 films) | Large (30 years, ~10k films) |
| **Signal risk** | Low — COVID effect is real and documented | Low |
| **Bayesian depth** | High (causal inference + hierarchical) | High (hierarchical + portfolio + sequential) |
| **Policy story** | Strong and timely | Very strong — multi-layer decisions |
| **Novelty** | High (natural experiment framing) | High (portfolio + sequential policy framing) |
| **Scope for 4 weeks** | Manageable | Ambitious but scoped |
| **Connects to your capstone** | Partial | Strong (sequential decision-making) |

---

## Key Dates
- **Nov 6:** Team roster due
- **Nov 16:** One-page proposal due (question, dataset, initial Bayesian approach)
- **Nov 30 / Dec 2:** Presentations (5–8 min)
- **Dec 4:** Final report + code due (max 10 pages, R or Python)
