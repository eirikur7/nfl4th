# Course context for the 4th down project

**Course:** T-811 Applied Probability (Hagnýt líkindafræði), Reykjavik University, Fall 2026. Teacher: Gunnar Gunnarsson.
**Student:** MSc engineering student. Python is the main language. Prefers intuition first, then math, then a concrete example. Icelandic native speaker who learned the field in English.

## How the course is graded
- Student-led homework presentations: 10%
- **Two group projects: 30%.** Up to 4 people per group. The stated focus is **"solving a challenging problem using simulations"**, so Monte Carlo simulation should be at the core of the project, not an add-on.
- Three in-class exams: 60% (best two count).
- The syllabus does not specify project deliverables (report, code, presentation) or language. Ask the student before assuming.

## Lecture topics (L1–L16; L9 had no lecture notes, exams were around L8 and L16)

| Lecture | Topic | Key tools |
|---|---|---|
| L1 | Sample spaces, axioms, counting | Inclusion–exclusion, uniform probability, **Monte Carlo** (procedure, convergence), permutations/combinations |
| L2 | Conditional probability, Bayes, independence | Probability trees, **law of total probability**, Bayes, base rates, binomial/geometric/hypergeometric |
| L3 | Discrete random variables | pmf/CDF, Bernoulli, binomial, Poisson, E[X], Var(X), linearity, **flaw of averages** E[g(X)] ≠ g(E[X]), entropy |
| L4 | Joint distributions | Joint/marginal pmf, conditional pmf, **conditional expectation and total expectation**, covariance, correlation |
| L5 | Continuous distributions | pdf/CDF, uniform, exponential, **normal** (z-scores, Φ), t-distribution, joint pdf |
| L6 | Derived variables and limit theorems | CDF method, Monte Carlo for derived variables, MGF, moments, **LLN, CLT**, normal approx. to binomial, **confidence intervals** |
| L7 | Stochastic processes | Bernoulli process, geometric waiting time, memorylessness, **random walks**, diffusion, Gaussian random walk |
| L8 | Poisson processes | Exponential gaps, exponential races, Gamma, splitting/merging, nonhomogeneous and compound Poisson |
| L10 | **Markov chains** | Markov property, transition matrix P, Chapman–Kolmogorov (P^m), transient/recurrent/**absorbing** states, irreducibility, stationary distribution πP = π, gambler's ruin |
| L11 | Queueing theory | M/M/1, traffic intensity ρ, Little's law, M/G/1 (simulate it), **Kelly criterion** (simulating betting strategies) |
| L12 | Reliability theory | Structure functions (series/parallel/k-out-of-n), r(p), survival function, failure rate, Weibull, Monte Carlo for lifetimes |
| L13 | Martingales and Brownian motion | Filtrations, conditional expectation, fair games, **optional stopping**, gambler's ruin, Brownian motion, Brownian bridge |
| L14 | Itô calculus | Itô integral, quadratic variation, Itô's formula, geometric Brownian motion |
| L15 | Stock markets and no arbitrage | Order books, discounting, bonds/yields, forwards, options, put–call parity |
| L16 | Arbitrage and options | Long/short, forward pricing, FX forwards, option payoffs and bounds, binomial trees |

**Most relevant to the 4th down project:** Monte Carlo (L1, L6), conditional probability and total probability/expectation (L2, L4), Bernoulli/binomial (L2–L3), CLT and confidence intervals (L6), and especially **Markov chains with absorbing states** (L10), since a drive is a Markov chain. L11's Kelly section (comparing strategies by simulation) and L13's "you can't beat a fair game" are useful framing too.

Course style: lecture notebooks are Python (numpy, scipy, matplotlib) with simulation alongside the math. The student's own notes are in `../Notes/L*.md` if more detail is needed on any topic.

## The project description (`../Verkefni/Verkefnislysing_4th_down.md`, in Icelandic)
Question: when should an NFL team go for it on 4th down instead of punting or kicking a field goal, depending on **field position** and **yards to go**?

Proposed method in the description:
1. Estimate from data: p(d) = P(1st down | d yards to go), p_FG(x) = P(FG good | distance x), and the punt-length distribution.
2. Value field position with **expected points (EP)**.
3. Compute the expected value of each option, e.g. E[go] = p(d)·EP(1st down) − (1−p(d))·EP_opponent(same spot).
4. **Monte Carlo** over the uncertainty in the estimated probabilities, to get P(going for it is better) instead of a yes/no.
5. Result: a heatmap (field position × yards to go) of the recommended option, compared with what coaches actually do.
- Scope in the description: **expected points, not win probability**; quarters 1–3 and close games only.
- Data: nflverse play-by-play (nflreadpy). Comparisons: Romer (2006), Ben Baldwin's nfl4th model.

## What is already built (`nfl4th/` folder)
The project is already far along, and **the implementation went further than the description**: see `README.md` and `results/RESULTS.md` for the full picture.
- A **Markov-chain drive simulator**. State = (down, yards to go, pos). Absorbing states: TD, FG, missed FG, punt, turnover, turnover on downs, safety. Games are 12 drives per team.
- Estimated from 2018–2025 nflverse data (~372k plays): empirical yard distributions per down/distance/zone, logistic p_FG(x), empirical punts, and the coaches' actual go/FG/punt behaviour.
- **11 strategies** compared by **win probability** over 200,000 simulated games each, against a team that plays like real coaches. Includes two-proportion z-tests, 95% CIs, parameter sweeps, and validation (coaches vs coaches ≈ 50%; simulated p(d) and drive outcomes match the data within 1–2 pp).
- Headline result: "go for it with ≤ 7 yards to go, anywhere" wins 55.0% (+4.8 pp vs coaches). Field position adds nothing significant beyond yards to go. Never going for it is worst (45.8%).
- Run with: `& C:\Users\ebben\miniforge3\envs\nfl4th\python.exe run_all.py` (conda env `nfl4th`, Miniforge).

**Note the gap with the description:** the description asks for an **expected-points** analysis with **Monte Carlo over parameter uncertainty** and an **EP-based heatmap**; the build uses **win probability** from simulated games, with a heatmap over strategy thresholds. Check with the student which framing the final deliverable should use, or whether the EP analysis should be added alongside.
