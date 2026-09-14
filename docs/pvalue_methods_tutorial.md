# How the p-values in this project are calculated

**Audience:** a university junior who has completed introductory probability, statistics, and linear algebra.

This document explains the actual methods implemented in this repository. It is not a generic list of statistical tests. The three inference families are:

1. **OLS regression with Newey–West (HAC) standard errors**
2. **Matched-control permutation tests**
3. **Unknown-date structural-break search with a moving-block bootstrap**

They answer different questions and do **not** all use a Student-$t$ distribution.

---

## 0. Start with the quantity being tested

### 0.1 Five-minute log returns

For five-minute bar $i$, let $C_i$ be its closing BTC price. The log return is

$$
r_i = \log(C_i)-\log(C_{i-1})
    = \log\left(\frac{C_i}{C_{i-1}}\right).
$$

Why log returns?

- Returns over consecutive intervals add: $\log(C_2/C_0)=\log(C_1/C_0)+\log(C_2/C_1)$.
- They treat equal percentage rises and falls more symmetrically than dollar changes.
- At five-minute frequency, the log return and ordinary percentage return are nearly identical.

### 0.2 Realised variance

For Eastern-calendar day $d$, daily realised variance is

$$
RV_d = \sum_{i\in d} r_i^2.
$$

The squared return removes direction. A $+2\%$ move and a $-2\%$ move contribute the same amount of variation.

Let $US(d)$ contain bars opening from 09:30 up to, but not including, 16:00 New York time. Then

$$
RV^{US}_d = \sum_{i\in US(d)} r_i^2.
$$

The outcome used by most tests is

$$
y_d = \frac{RV^{US}_d}{RV_d}.
$$

Interpretation: if $y_d=0.40$, then 40% of that day's measured BTC variance occurred during US equity hours.

Why use a **share**?

Suppose total BTC volatility doubles during a bull market but its intraday allocation does not change. Both $RV^{US}_d$ and $RV_d$ roughly double, while their ratio remains stable. The share therefore focuses on **when** volatility occurs rather than simply **how much** volatility exists.

### 0.3 The null hypothesis

Every p-value starts with a null hypothesis $H_0$, usually "there is no effect."

A p-value means:

> If the null hypothesis and the test's assumptions were true, what fraction of repeated datasets would produce a test statistic at least as extreme as ours?

It does **not** mean:

- the probability that $H_0$ is true;
- the probability that the result happened "by chance";
- the size or practical importance of the effect;
- the probability that ETFs caused the effect.

A p-value can measure incompatibility with a statistical null. Causality comes from research design and assumptions, not from a small number.

---

# Part 1 — OLS regression with Newey–West standard errors

## 1.1 What is OLS?

**OLS** means **ordinary least squares**. It estimates coefficients by choosing the fitted line that minimises the sum of squared residuals.

In matrix notation,

$$
\mathbf y=\mathbf X\boldsymbol\beta+\boldsymbol\varepsilon,
$$

where:

- $\mathbf y$: observed outcome values;
- $\mathbf X$: columns of explanatory variables;
- $\boldsymbol\beta$: unknown coefficients;
- $\boldsymbol\varepsilon$: unexplained errors.

OLS chooses

$$
\hat{\boldsymbol\beta}
=\arg\min_{\boldsymbol\beta}
(\mathbf y-\mathbf X\boldsymbol\beta)'
(\mathbf y-\mathbf X\boldsymbol\beta).
$$

Solving the first-order condition gives

$$
\hat{\boldsymbol\beta}
=(\mathbf X'\mathbf X)^{-1}\mathbf X'\mathbf y.
$$

The residual for observation $d$ is

$$
\hat\varepsilon_d=y_d-\hat y_d.
$$

**Coefficient:** estimated effect size.  
**Standard error:** estimated uncertainty of the coefficient.  
**Test statistic:** coefficient divided by its standard error.

---

## 1.2 Before-versus-after model

Define the event dummy

$$
Post_d =
\begin{cases}
0,&d<\text{11 January 2024},\\
1,&d\geq\text{11 January 2024}.
\end{cases}
$$

The model is

$$
y_d=\alpha+\beta Post_d+\varepsilon_d.
$$

Interpretation:

- $\alpha$ is the pre-launch mean.
- $\alpha+\beta$ is the post-launch mean.
- $\beta$ is the difference between the two means.

Because the only regressor is a zero/one dummy, OLS gives exactly

$$
\hat\beta=\bar y_{\text{post}}-\bar y_{\text{pre}}.
$$

Project result:

$$
\bar y_{\text{pre}}=0.37905,\qquad
\bar y_{\text{post}}=0.39784,
$$

so

$$
\hat\beta=0.01880.
$$

That is an increase of **1.88 percentage points**, not 1.88%.

This coefficient is easy. The difficult part is estimating its uncertainty correctly.

---

## 1.3 Why the ordinary textbook standard error is unsafe

The simplest OLS standard error assumes:

### Homoskedasticity

$$
Var(\varepsilon_d\mid X)=\sigma^2
$$

for every day. In plain language: errors have the same variance throughout the sample.

Financial data is usually **heteroskedastic**:

- calm days have small errors;
- crisis days have large errors;
- uncertainty changes through time.

### No autocorrelation

$$
Cov(\varepsilon_d,\varepsilon_{d-\ell}\mid X)=0
\quad\text{for }\ell\neq0.
$$

In plain language: today's error tells us nothing about tomorrow's error.

Financial volatility tends to cluster. A high-volatility day is often followed by another high-volatility day. Therefore daily errors may be **serially correlated** or **autocorrelated**.

If correlated observations are treated as independent, the effective amount of information is exaggerated. Standard errors can become too small and p-values too optimistic.

---

## 1.4 What Newey–West does

**Newey–West** is a covariance estimator. It does not change the OLS coefficient $\hat\beta$. It changes the estimated variance of that coefficient.

It is also called a **HAC** estimator:

- **H**eteroskedasticity
- **A**nd
- **A**utocorrelation
- **C**onsistent

The usual OLS covariance estimator has the form

$$
\widehat{Var}_{OLS}(\hat{\boldsymbol\beta})
=\hat\sigma^2(\mathbf X'\mathbf X)^{-1}.
$$

The HAC "sandwich" estimator is

$$
\widehat{Var}_{HAC}(\hat{\boldsymbol\beta})
=(\mathbf X'\mathbf X)^{-1}
\mathbf S
(\mathbf X'\mathbf X)^{-1},
$$

where $\mathbf S$ contains:

1. squared residual contributions, which handle heteroskedasticity;
2. cross-products between residuals separated by several lags, which handle autocorrelation.

A simplified expression is

$$
\mathbf S
=\mathbf\Gamma_0+
\sum_{\ell=1}^{L}w_\ell
\left(\mathbf\Gamma_\ell+\mathbf\Gamma_\ell'\right),
$$

with

$$
\mathbf\Gamma_\ell
=\sum_{d=\ell+1}^{T}
\mathbf x_d\hat\varepsilon_d
\hat\varepsilon_{d-\ell}\mathbf x_{d-\ell}'.
$$

The implementation uses Bartlett weights:

$$
w_\ell=1-\frac{\ell}{L+1}.
$$

Nearby residual relationships receive more weight; relationships near the maximum lag receive less.

### Buzzword: lag

A lag is a time separation:

- lag 1: yesterday;
- lag 2: two days ago;
- lag 7: one week ago.

The project fixes the maximum lag using

$$
L=\left\lfloor
4\left(\frac{T}{100}\right)^{2/9}
\right\rfloor.
$$

This avoids choosing a lag after seeing which value gives the desired p-value.

Actual values:

| Model | Observations | HAC lags |
| --- | ---: | ---: |
| Before/after | 1,449 days | 7 |
| ETF-flow regression | 486 flow days | 5 |
| Equity-hours versus CME-evening | 1,000 ETF-open days | 6 |

---

## 1.5 From Newey–West standard error to p-value

The test statistic is

$$
z=\frac{\hat\beta-\beta_0}{SE_{HAC}(\hat\beta)}.
$$

Usually $\beta_0=0$, so

$$
z=\frac{\hat\beta}{SE_{HAC}(\hat\beta)}.
$$

For the before/after comparison:

$$
z=2.0851.
$$

The code uses Statsmodels robust covariance with `use_t=False`. Therefore its p-value uses the **standard normal distribution**, not Student's $t$-distribution:

$$
p_{two-sided}
=P(|Z|\geq |2.0851|)
=2[1-\Phi(2.0851)]
\approx0.0371.
$$

Here $\Phi$ is the standard normal cumulative distribution function.

### Important naming correction

Some variables in the code and output are named `t`, `t_hac`, or `t_stat_hac`. Numerically they are "estimate divided by robust standard error," but the reference distribution is standard normal because `use_t=False`. Calling them **HAC z-statistics** is more precise.

### Confidence interval

A 95% asymptotic confidence interval is approximately

$$
\hat\beta\pm1.96\,SE_{HAC}(\hat\beta).
$$

Interpretation: under repeated sampling and valid assumptions, 95% of intervals constructed by this method would contain the true coefficient. It does not mean there is a 95% probability that this already-computed interval contains the coefficient.

---

## 1.6 One-sided versus two-sided tests

A two-sided alternative is

$$
H_1:\beta\neq0.
$$

It counts extreme positive and negative results:

$$
p_{two}=2[1-\Phi(|z|)].
$$

A positive one-sided alternative is

$$
H_1:\beta>0.
$$

For positive $z$,

$$
p_{one}=1-\Phi(z)=\frac{p_{two}}2.
$$

A one-sided test is legitimate only when the direction was fixed **before** seeing the data. It cannot be selected because the estimated sign happened to be convenient.

---

## 1.7 ETF-flow regression

The pre-registered primary model is

$$
y_d
=\alpha
+\beta FlowZ_d
+\gamma\log(RV_d)
+\sum_m\delta_m Month_{m,d}
+\varepsilon_d.
$$

### What each term means

#### $y_d$: US-session variance share

The fraction of daily BTC variance occurring from 09:30–16:00 ET.

#### Absolute ETF flow

The research mechanism concerns trading generated by both:

- creations/inflows;
- redemptions/outflows.

Therefore the dose is the magnitude:

$$
|Flow_d|.
$$

A $-\$500m$ redemption and $+\$500m$ creation both have magnitude $\$500m$.

#### Winsorisation

The largest 1% of flow magnitudes are capped at the 99th-percentile value:

$$
Flow^{win}_d=\min(|Flow_d|,q_{0.99}).
$$

This prevents a few enormous days from determining the fitted slope.

**Winsorisation is not deletion.** The observations remain; only extreme values are capped.

#### Standardisation / z-score

$$
FlowZ_d
=\frac{Flow^{win}_d-\overline{Flow^{win}}}
{SD(Flow^{win})}.
$$

Then $\beta$ means the change in variance share associated with a **one-standard-deviation** increase in flow magnitude.

#### $\log(RV_d)$

Total daily variance is right-skewed. Taking its logarithm compresses huge values:

$$
\log(RV_d).
$$

It controls for whether the day was generally volatile.

#### Month fixed effects

For every calendar month except one reference month, the model includes a zero/one indicator.

Example:

$$
Month_{\text{Feb 2024},d}
=1
$$

only for dates in February 2024.

These **fixed effects** absorb the average level of each month. Consequently, $\beta$ is identified mainly by comparing high-flow and low-flow days **within the same month**, rather than comparing an entire calm year with a volatile year.

### Actual result

$$
\hat\beta=0.003123,\qquad
SE_{HAC}=0.006475.
$$

Therefore

$$
z=\frac{0.003123}{0.006475}=0.4823.
$$

The two-sided normal p-value is

$$
p_{two}=2[1-\Phi(0.4823)]
\approx0.6296.
$$

Because the pre-registered prediction was positive,

$$
p_{one}=1-\Phi(0.4823)
\approx0.3148.
$$

Interpretation:

- estimated slope is slightly positive;
- uncertainty is about twice as large as the estimate;
- the result is compatible with zero;
- there is no convincing evidence that larger same-day flow magnitude is associated with a larger US-hours variance share.

This does **not** prove the coefficient is exactly zero.

---

## 1.8 Equity-hours versus CME-evening regression

The windows have unequal lengths:

- US equity window: 390 minutes;
- CME evening comparison: 360 minutes.

Comparing raw shares directly creates a mechanical window-length problem. The model instead constructs

$$
q_d
=\log\left(\frac{RV^{equity}_d}{390}\right)
-\log\left(\frac{RV^{evening}_d}{360}\right).
$$

Equivalently,

$$
q_d
=\log\left(
\frac{RV^{equity}_d/390}
{RV^{evening}_d/360}
\right).
$$

This compares variance **per minute** in the two windows on the same day.

Then estimate

$$
q_d=\alpha+\beta Post_d+\varepsilon_d
$$

using Newey–West standard errors.

Actual result:

$$
\hat\beta=0.132845,\qquad
SE_{HAC}=0.063635.
$$

Thus

$$
z=\frac{0.132845}{0.063635}=2.0876
$$

and

$$
p=2[1-\Phi(2.0876)]
\approx0.0368.
$$

Because the outcome is a log ratio, convert the coefficient to a proportional change:

$$
\exp(0.132845)-1\approx0.142.
$$

Interpretation: the equity-window/evening variance-intensity ratio was estimated to rise by approximately **14.2%** after launch.

This rejects equal scaling of those two windows. It does not prove ETFs were the cause, because US news and macroeconomic announcements also concentrate in equity hours.

---

## 1.9 Placebo dates

A **placebo test** repeats the same before/after model at dates where no ETF launch occurred:

$$
y_d=\alpha+\beta PlaceboPost_d+\varepsilon_d.
$$

If many arbitrary dates produce "significant breaks," the series probably has a broad trend or multiple regimes. A small p-value at the real date is then less special.

In this project, most placebo dates were significant. That weakens a unique January-2024 causal interpretation.

This is not a multiple-testing-adjusted family in the compact report. It is primarily a diagnostic: "Does the real date stand out?"

---

# Part 2 — Matched-control permutation tests

## 2.1 What question do they answer?

Bitcoin trades every day, but ETF creation/redemption is unavailable:

- on weekends;
- on NYSE holidays.

If the ETF channel concentrates variance into US equity hours, closed days should show less concentration than nearby normal weekdays after launch.

A direct comparison of all holidays with all weekdays is poor because:

- holidays occur in particular months;
- volatility regimes change over time;
- Christmas 2022 is not naturally comparable with an arbitrary weekday in 2025.

The implementation therefore matches each closed day to nearby ordinary trading days.

---

## 2.2 Step 1: construct a local difference

For closed day $h$, let $N(h)$ be nearby regular weekdays:

- holiday test: within $\pm5$ calendar days;
- weekend test: within $\pm3$ calendar days.

Define

$$
\Delta_h
=y_h-\frac{1}{|N(h)|}\sum_{d\in N(h)}y_d.
$$

Interpretation:

- $\Delta_h=0$: closed day resembles nearby open weekdays;
- $\Delta_h<0$: closed day has a smaller US-hours share;
- $\Delta_h>0$: closed day has a larger US-hours share.

Matching removes much of the local volatility regime because treated and control dates are close together.

### Buzzword: counterfactual

The **counterfactual** is what would have happened under a condition we did not observe.

We cannot observe the same date both as a holiday and as a normal trading day. The average of nearby regular days is used as an estimated counterfactual.

---

## 2.3 Step 2: difference in differences

Before ETF launch, there was no ETF primary market to be closed. After launch, closed days lack that channel.

The statistic is

$$
DID
=\overline{\Delta}_{post}
-\overline{\Delta}_{pre}.
$$

This is a **difference in differences**:

1. within each period, difference between closed and nearby open days;
2. difference between that gap after and before launch.

The directional prediction is

$$
H_1:DID<0.
$$

Why negative? After launch, ordinary weekdays can contain ETF activity while closed days cannot, so the closed-day minus open-day gap should become more negative.

---

## 2.4 Why not use a t-test?

There are only 20 post-launch holidays. A t-test would rely heavily on assumptions about:

- approximate normality;
- stable variance;
- independent observations.

A permutation test instead creates its own reference distribution from the observed $\Delta_h$ values.

---

## 2.5 How the permutation test works

Suppose there are $n$ matched closed-day differences:

$$
\Delta_1,\ldots,\Delta_n.
$$

Each has a label:

$$
G_h=
\begin{cases}
0,&\text{pre-launch},\\
1,&\text{post-launch}.
\end{cases}
$$

The observed statistic is

$$
T_{obs}
=mean(\Delta_h\mid G_h=1)
-mean(\Delta_h\mid G_h=0).
$$

Under the null hypothesis that the pre/post label has no relationship with the deltas, shuffle the labels while keeping:

- every $\Delta_h$ fixed;
- the number of pre and post labels fixed.

For permutation $b$:

$$
T_b
=mean(\Delta_h\mid G_h^{(b)}=1)
-mean(\Delta_h\mid G_h^{(b)}=0).
$$

Repeat this 5,000 times to obtain a simulated null distribution.

Because the prediction is negative, count permutations satisfying

$$
T_b\leq T_{obs}.
$$

The implementation uses an add-one correction:

$$
p
=\frac{1+\#\{T_b\leq T_{obs}\}}
{1+B},
\qquad B=5000.
$$

The correction prevents a reported p-value of exactly zero.

---

## 2.6 Weekend result

Observed result:

$$
T_{obs}=-0.06664.
$$

That is a **6.66 percentage-point** post-launch decline in the weekend-versus-nearby-weekday gap.

No shuffled statistic was as negative:

$$
\#\{T_b\leq T_{obs}\}=0.
$$

Therefore

$$
p=\frac{1+0}{5001}
=0.00019996
\approx0.0002.
$$

This is the smallest p-value possible with 5,000 permutations.

It is strong statistical evidence that the weekend/open-weekday relationship changed. However, weekends differ from weekdays in liquidity and participant composition, not only ETF availability. The causal interpretation is therefore weaker than the small p-value may suggest.

---

## 2.7 Holiday result

Observed result:

$$
T_{obs}=-0.04486.
$$

That is a **4.49 percentage-point** estimated decline in the holiday-versus-nearby-weekday gap.

There were 544 shuffled statistics at least as negative, so

$$
p=\frac{1+544}{5001}
=0.10898.
$$

Interpretation:

- estimated direction agrees with the ETF-channel prediction;
- the p-value does not cross 0.05;
- with only 20 post-launch holidays, uncertainty is large;
- report "directionally consistent but imprecise," not "no effect."

---

## 2.8 Assumption: exchangeability

Permutation inference relies on **exchangeability** under the null.

Informally, if the null were true, relabelling observations as pre or post should not change their joint distribution.

This assumption is strong for time series. Pre/post labels are tied to time, and the deltas may still contain:

- trends;
- serial correlation;
- regime changes.

Therefore calling this permutation p-value literally "exact" would be too strong unless exchangeability is credible. Local matching helps remove changing regimes, but it does not guarantee exchangeability.

This is an important limitation to understand and defend:

> The permutation distribution is a transparent finite-sample reference that avoids a small-sample normal approximation, but its validity still requires matched deltas to be exchangeable under the null.

Possible stronger extensions:

- permute labels only within calendar strata;
- use random contiguous blocks rather than individual labels;
- run a randomisation-inference design based on placebo event dates;
- report effect sizes and intervals alongside p-values.

---

# Part 3 — Unknown-date break search with moving-block bootstrap

## 3.1 Why search for a break date?

The before/after model forces the break to occur on 11 January 2024 because that is the date we selected.

This cannot answer:

> If the data were allowed to choose, would it independently choose January 2024?

The unknown-date test scans many possible dates and selects the strongest mean shift.

It is similar to a **Quandt–Andrews supremum-Wald test**.

### Buzzword: structural break

A structural break is a change in a model parameter. Here the parameter is the mean US-session variance share.

Before candidate date $k$:

$$
y_d=\mu_1+\varepsilon_d.
$$

After candidate date $k$:

$$
y_d=\mu_2+\varepsilon_d.
$$

The null is

$$
H_0:\mu_1=\mu_2
$$

for every admissible break date.

---

## 3.2 Test statistic at one candidate date

For split position $k$, calculate

$$
\bar y_1(k)=\frac1{k}\sum_{d=1}^{k}y_d
$$

and

$$
\bar y_2(k)=\frac1{T-k}\sum_{d=k+1}^{T}y_d.
$$

Calculate within-segment sums of squares:

$$
SS_1(k)=\sum_{d=1}^{k}(y_d-\bar y_1)^2,
$$

$$
SS_2(k)=\sum_{d=k+1}^{T}(y_d-\bar y_2)^2.
$$

The pooled variance estimate is

$$
s^2(k)=\frac{SS_1(k)+SS_2(k)}{T-2}.
$$

The standard error of the mean difference is

$$
SE(k)
=\sqrt{s^2(k)
\left(\frac1k+\frac1{T-k}\right)}.
$$

Then

$$
t(k)=\frac{\bar y_2(k)-\bar y_1(k)}{SE(k)}
$$

and the Wald statistic is

$$
W(k)=t(k)^2.
$$

Squaring makes large positive and negative breaks both count as evidence against no break.

---

## 3.3 Why trim the sample?

A split near the first or last observation leaves one side with very few data points, making its estimated mean and variance unstable.

The implementation removes the first and last 15% of dates as candidates:

$$
k\in[0.15T,\;0.85T].
$$

This is called **15% trimming**.

---

## 3.4 Supremum-Wald statistic

Calculate $W(k)$ at every allowed date and take the maximum:

$$
SupWald=\max_k W(k).
$$

The selected break date is

$$
\hat k=\arg\max_k W(k).
$$

### Why an ordinary p-value is invalid

If we tested one date, a conventional reference distribution might be usable under strong assumptions.

Here we search hundreds of dates and keep only the largest statistic. Even under no true break, the maximum of hundreds of noisy statistics will often be large.

This is a **data-snooping** or **multiple-search** problem. Comparing the maximum with an ordinary one-date $t$, $F$, or $\chi^2$ distribution would severely understate the p-value.

We instead simulate the distribution of the **maximum search statistic** under no break.

---

## 3.5 Bootstrap: the basic idea

A **bootstrap** approximates repeated sampling by generating many artificial datasets from the observed data.

For this test, construct null residuals:

$$
\hat\varepsilon_d=y_d-\bar y.
$$

These residuals preserve the observed deviations but remove any explicit fitted mean break.

For each bootstrap replication:

1. resample residuals;
2. add them to the overall mean;
3. scan all candidate dates;
4. save that artificial series' maximum Wald statistic.

After $B=400$ replications, the bootstrap values

$$
SupWald_1^*,\ldots,SupWald_{400}^*
$$

form the null distribution.

---

## 3.6 Why moving blocks?

An ordinary iid bootstrap samples individual residuals independently.

That destroys time-series dependence. If several adjacent daily shares tend to move together, sampling them independently makes the simulated series unrealistically noisy and memoryless.

A **moving-block bootstrap** samples contiguous blocks:

$$
(\hat\varepsilon_s,\hat\varepsilon_{s+1},\ldots,
\hat\varepsilon_{s+L-1}).
$$

Blocks are sampled with replacement and concatenated until a length-$T$ artificial series is produced.

Within each block, observed short-run dependence is preserved.

### Choosing block length

The code first estimates first-order autocorrelation:

$$
\hat\rho_1
=\frac{\sum_{d=2}^{T}(y_d-\bar y)(y_{d-1}-\bar y)}
{\sum_{d=1}^{T}(y_d-\bar y)^2}.
$$

The conventional baseline is

$$
L_0\approx T^{1/3}.
$$

It also computes the AR(1) integral-timescale approximation

$$
\tau=\frac{1+\hat\rho_1}{1-\hat\rho_1}
$$

and requires approximately

$$
L\geq2\tau.
$$

The final block length is the larger of the conventional and persistence-aware choices, capped so it cannot consume most of the sample.

Why? Blocks that are too short destroy dependence and can manufacture apparently significant breaks.

---

## 3.7 Bootstrap p-value

Let $S_{obs}$ be the observed maximum statistic. Count

$$
c=\#\{S_b^*\geq S_{obs}\}.
$$

The p-value is

$$
p=\frac{1+c}{1+B}.
$$

The project used $B=400$ and obtained

$$
p=0.0124688=\frac5{401}.
$$

Therefore four bootstrap maxima were at least as large as the observed maximum:

$$
c=4.
$$

Interpretation:

> Under the fitted no-break bootstrap process, a maximum break statistic this large appeared in about 1.25% of replications.

This supports the existence of some mean instability in the series.

---

## 3.8 What the selected date means

The selected date was 18 January 2023, 358 days before ETF launch.

This means:

- among the scanned candidate splits, that date produced the largest standardised mean difference;
- January 2024 was not the strongest single split;
- the series likely contains a wider trend or earlier regime change.

It does **not** mean:

- the true economic change definitely occurred on exactly 18 January 2023;
- the bootstrap p-value is the probability that this date is correct;
- the model has produced a confidence interval for the break date;
- the 2024 ETF launch had no incremental effect;
- the selected break was caused by a known event in January 2023.

The bootstrap tests "is the maximum unusually large under the simulated no-break process?" It does not identify causality.

---

# Part 4 — Which distribution did each result use?

| Reported result | Model/statistic | Reference distribution | Why |
| --- | --- | --- | --- |
| Before/after launch | Post dummy in OLS | Asymptotic standard normal using Newey–West SE | Large daily time series with heteroskedasticity and autocorrelation |
| ETF-flow relationship | Multiple OLS regression | Asymptotic standard normal using Newey–West SE | Persistent financial variables and controls |
| Equity hours vs CME evening | OLS on log variance ratio | Asymptotic standard normal using Newey–West SE | Daily within-day contrast remains a time series |
| Placebo dates | Repeated post-dummy OLS | Asymptotic standard normal using Newey–West SE | Same known-date model at fake dates |
| Weekend control | Matched difference-in-differences | Empirical permutation distribution | Avoid small-sample parametric approximation |
| Holiday control | Matched difference-in-differences | Empirical permutation distribution | Very small holiday sample |
| Unknown-date break | Maximum squared mean-difference statistic | Moving-block bootstrap distribution | Corrects both date search and time dependence |

No single test is universally best. The method must match:

- the question;
- sample size;
- dependence structure;
- whether a date was fixed or searched;
- whether the alternative was directional.

---

# Part 5 — Buzzword glossary

## Alternative hypothesis

The effect predicted if the null is false, such as $\beta>0$ or $DID<0$.

## Asymptotic

An approximation justified as sample size becomes large. Newey–West z p-values are asymptotic.

## Autocorrelation / serial correlation

Correlation between a variable or error and its own past values.

## Bootstrap

Constructing artificial samples from observed data to approximate a statistic's sampling distribution.

## Causal identification

The assumptions and comparisons that allow an association to be interpreted as causal. A small p-value alone is not identification.

## Coefficient

Estimated change in an outcome associated with a one-unit change in a regressor, holding other included regressors fixed.

## Confidence interval

A range produced by a repeated-sampling procedure designed to contain the true parameter at a stated long-run frequency.

## Counterfactual

The outcome that would have occurred under a condition that was not observed.

## Covariance matrix

A matrix containing estimated variances of coefficients on its diagonal and covariances between coefficients off-diagonal.

## Difference in differences

A difference between groups, followed by a difference between periods.

## Dummy / indicator variable

A variable equal to zero or one, such as `post launch`.

## Endogenous break date

A break date chosen by the data rather than specified before the analysis. Here "endogenous" means data-selected; it does not mean the same thing as regressor endogeneity.

## Exchangeability

Under the null, observations or labels can be permuted without changing their joint distribution.

## Fixed effect

A separate intercept adjustment for each group or period, such as each calendar month.

## HAC

Heteroskedasticity-and-autocorrelation-consistent covariance estimation.

## Heteroskedasticity

Errors have unequal variances across observations.

## Lag

A past time separation, such as one day earlier.

## Moving-block bootstrap

A bootstrap that resamples consecutive observations in blocks to preserve short-run time dependence.

## Null distribution

The distribution of a test statistic assuming the null hypothesis is true.

## Null hypothesis

The baseline proposition being tested, usually "no effect" or "no break."

## OLS

Ordinary least squares: coefficients selected to minimise summed squared residuals.

## One-sided test

A test with a pre-specified directional alternative, such as $\beta>0$.

## p-value

Under the null and model assumptions, the probability of a statistic at least as extreme as the observed statistic.

## Permutation test

A test that constructs a null distribution by shuffling labels and recomputing the statistic.

## Placebo test

A test deliberately applied where the hypothesised mechanism should be absent.

## Regressor

An explanatory variable in a regression.

## Residual

Observed outcome minus fitted outcome.

## Standard error

Estimated standard deviation of an estimator across hypothetical repeated samples.

## Standardisation / z-score

Subtract a variable's mean and divide by its standard deviation.

## Structural break

A change in a model parameter at some point in time.

## Supremum-Wald

The largest Wald statistic obtained while scanning candidate break dates.

## Test statistic

A number summarising how far the observed result lies from the null relative to estimated noise.

## Two-sided test

A test that counts sufficiently large positive or negative departures from zero.

## Wald statistic

Generally, a squared standardised distance between an estimate and its null value.

## Winsorisation

Capping extreme values at chosen percentiles without deleting observations.

---

# Part 6 — How to explain the methods orally

## Twenty-second version

> I used three kinds of inference. Large daily regressions use Newey–West standard errors because volatility data is heteroskedastic and serially correlated. Holiday and weekend controls use label permutations because the holiday sample is small. The unknown break-date test uses a moving-block bootstrap because searching many dates inflates the maximum statistic and ordinary resampling would destroy time dependence.

## If asked why not use a t-test everywhere

> Each question has a different sampling problem. The daily regressions have enough observations for HAC asymptotics. Holidays have too few observations for a comfortable parametric approximation, so I form an empirical permutation distribution. The break test searches hundreds of dates, so its null must reproduce the entire search; a one-date t distribution would be invalid.

## If asked whether p=0.037 proves ETFs caused the shift

> No. It says the observed before/after coefficient is unusual under a no-mean-shift regression with the HAC approximation. Causality requires the controls. The null flow result, significant placebo dates, and earlier data-selected break prevent a clean causal claim.

## If asked why p=0.109 is not “no holiday effect”

> The estimated holiday effect is economically large and has the predicted sign, but only 20 post-launch holidays create wide uncertainty. Failure to reject is not proof of zero; I describe it as directionally consistent but imprecise.

## If asked what Newey–West changed

> It did not change the estimated mean difference or slope. It changed the covariance matrix and therefore the standard error, z-statistic, confidence interval, and p-value.

---

# Part 7 — Final cautions

1. **Statistical significance is not economic significance.** Always report effect size.
2. **Failure to reject is not proof of no effect.** Consider power and uncertainty.
3. **A small p-value is not causality.** Research design matters.
4. **One-sided tests must be declared before seeing the sign.**
5. **Searching dates requires correcting for the search.**
6. **Permutation tests still need exchangeability.**
7. **Bootstrap design matters.** Independent resampling can be wrong for time series.
8. **The code labels HAC statistics as `t`, but the implementation uses an asymptotic normal reference distribution.**

The most defensible overall interpretation is:

> The data contains evidence of greater BTC variance concentration during US equity hours, but the combined tests do not establish ETF creation/redemption as the sole cause of the broader structural shift.
