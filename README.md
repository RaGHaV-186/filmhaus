# 🎬 Filmhaus Recommender

A four-model movie recommendation system built on MovieLens-1M, with an emphasis on **honest evaluation** over headline numbers. Four recommendation strategies are implemented from the ground up and compared against two business-motivated KPIs — recommendation quality and catalogue coverage — revealing a real accuracy–coverage trade-off. The models are served through a local Streamlit app.

---

## The scenario (a framing device)

To ground the metrics in something concrete, the project imagines a fictional mid-size streaming service, **"Filmhaus."** This scenario is a *framing device*, not a real business — its purpose is to motivate *why* certain metrics matter, so evaluation choices are driven by product thinking rather than by whatever is easiest to compute.

Filmhaus cares about two things a recommender can influence:

1. **Are the recommendations good?** — do users actually like what we surface? → measured by **Precision@10** (relevance = rating ≥ 4)
2. **Are we using our catalogue?** — a service that licenses thousands of titles but only ever recommends a handful is wasting its content spend → measured by **Coverage** (fraction of the catalogue recommended across all users)

These two pull in opposite directions, and quantifying that tension is the core contribution of the project.

---

## The data

- **MovieLens-1M**: 1,000,209 ratings from 6,040 users across 3,706 rated films.
- **A coverage ceiling in the data itself**: the full catalogue lists 3,883 films, so **177 were never rated by anyone** — they can never be recommended, capping coverage before any model is built.
- **Severe temporal front-loading**: **90.5%** of all ratings fall in the year 2000, with a single month (November 2000) holding ~29% of the entire dataset.

### Honest train/test split

The data is split **by time, not randomly** — the model trains on older ratings and is tested on newer ones, so it never "sees the future." A random split would leak future ratings into training and inflate every result. Because of the front-loading, a calendar-based cutoff would leave a starved test set, so the split is placed at the **80th percentile of ratings by time** (cutoff: **2 December 2000**), giving an 800,164 / 200,045 train/test split.

Evaluation is restricted to a **fair, comparable population** of **1,122 users** who both liked ≥1 film in the test period *and* have training history to personalize from. Every model is scored on this identical set, under an identical rule (recommend unseen films only), so the comparison is fair.

---

## The four models

| # | Model | Approach | Type |
|---|-------|----------|------|
| 1 | **Popular** | Most-rated unseen films, same for everyone | Non-personalized baseline |
| 2 | **Based on your history** | Films similar to those the user rated highly | Item-based collaborative filtering |
| 3 | **People like you** | Films liked by the user's nearest taste-neighbours | User-based collaborative filtering (KNN, cosine, k=20) |
| 4 | **Your taste tribe** | Films favoured by the user's cluster | K-Means personas on SVD-reduced users |

The collaborative-filtering logic (similarity, neighbour selection, pooling) is implemented from scratch on top of standard numerical primitives (scikit-learn's `cosine_similarity`, NumPy), rather than via a black-box recommender library, to keep every step inspectable.

---

## Results

All figures on the same 1,122-user fair evaluation (Coverage % is out of the 3,662 films present in the training matrix).

| Model | Precision@10 | Coverage (distinct films) | Coverage % |
|-------|:------------:|:-------------------------:|:----------:|
| Popular (baseline) | 0.200 | 136 | 3.7% |
| Based on your history (item CF) | **0.219** | 313 | 8.5% |
| People like you (user CF) | 0.202 | **678** | **18.5%** |
| Your taste tribe (K-Means) | 0.201 | 231 | 6.3% |

### What the numbers say

**On a *fair* baseline, personalization's precision edge is modest.** Item-based CF beats popularity by **9.6%** on Precision@10 (0.219 vs 0.200) — not the dramatic win a broken baseline would suggest. This is a known and important result: **popularity is a deceptively strong baseline in offline evaluation**, because popular films are popular precisely among the users being scored. An honest, modest margin is the point.

**On coverage, personalization wins decisively.** The popularity baseline exposes only 3.7% of the catalogue; user-based CF reaches 18.5% — **5× the variety**. For a service paying to license a large catalogue, this is where personalization earns its keep.

**The accuracy–coverage trade-off is real and measurable.** Item-based CF is more *accurate* but *narrower* (0.219 precision, 8.5% coverage); user-based CF is slightly less accurate but far *broader* (0.202 precision, 18.5% coverage). Which to prefer is a **business decision**, not a purely technical one — precision serves the next click, coverage serves catalogue ROI and discovery.

**K-Means personas underperformed — an honest negative result.** Clustering tied the baseline on precision (0.201) and trailed CF on coverage. The cause is measurable: on this sparse data, **47.9% of users collapsed into a single cluster** lacking enough signal to form a distinct segment — even after SVD reduction to 50 latent factors (which retained 40.6% of variance). Personas recommend a *tribe's* average taste rather than an *individual's*, so per-user CF wins. Clustering suits denser data or a fast coarse-lookup layer, not a primary recommender here.

---

## Limitations

- **Unrated = 0**: unrated films are treated as a 0 rating, so the model cannot distinguish "disliked" from "unseen" — a standard but real simplification.
- **Sparse clustering**: K-Means produces one dominant low-activity cluster (47.9% of users) regardless of SVD preprocessing.
- **Cold start**: users with no training history fall back to the popularity baseline.
- **Offline proxies**: Precision@10 and coverage are offline stand-ins for real business KPIs; directions are more trustworthy than absolute magnitudes.

---

## Tech stack

- **Python**, **pandas**, **NumPy**, **scikit-learn** (cosine similarity, TruncatedSVD, K-Means)
- **Streamlit** — interactive local app (type a user ID, see all four models' recommendations side by side)

---

## Running locally

```bash
git clone https://github.com/RaGHaV-186/filmhaus.git
cd filmhaus
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Place the MovieLens-1M .dat files in data/
# Run the notebook (filmhaus.ipynb) top to bottom to generate models.pkl
streamlit run app.py
```

The app loads the saved models and serves recommendations from all four strategies for any user ID.

---

## What I'd do next

- **Matrix factorization / two-tower neural model** — learn latent user/item factors with gradient descent rather than SVD; the natural next step from the collaborative-filtering foundation here.
- **Similarity-weighted neighbour pooling** — weight neighbours by similarity instead of a flat sum.
- **Time-decayed popularity** — a stronger, more production-realistic baseline.
- **Deployment** — the app runs locally; a hosted version would need a platform with a free CPU tier (the free ML-hosting landscape shifted during this project).

---

## What this project demonstrates

- End-to-end recommender pipeline: raw data → time-based split → four models → fair evaluation → interactive app
- Collaborative filtering (item- and user-based / KNN) implemented and understood from first principles
- SVD dimensionality reduction and K-Means clustering, with an honest measured negative result
- Evaluation rigour: a fair baseline, a leak-free split, two KPIs, and a demonstrated accuracy–coverage trade-off