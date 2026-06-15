// =============================================================================
// lmc_popularity.stan
// Bayesian Hierarchical Beta Regression: LMC → Streaming Popularity
//
// Model structure
// ---------------
//   Likelihood : Beta(mu_i, phi)  [popularity scaled to (0,1)]
//   Link       : logit(mu_i)
//
//   Fixed effects (song level):
//     - LMC mean (z-scored)
//     - Functional LMC trajectory via B-spline basis (K basis functions)
//     - LMC consistency (SD of smooth, z-scored)
//     - Audio controls: energy, valence, danceability (z-scored)
//     - Song age (z-scored)
//     - Orientation dummy (1 = narrative)
//     - LMC_mean × orientation interaction
//
//   Random effects:
//     - Artist intercepts + artist-varying LMC slope
//       (song nested in artist; correlated via Cholesky LKJ)
//     - Genre intercepts
//       (artist nested in genre)
//
//   Level-2 (artist) model:
//     - Artist intercepts predicted by orientation
//
//   Priors (weakly informative, theoretically grounded):
//     - Global intercept       : Normal(0, 1.5)
//     - Fixed slopes           : Normal(0, 0.5)
//     - Functional coefficients: RW1 smoothness prior
//     - sigma_artist           : HalfNormal(0, 0.5)
//     - sigma_genre            : HalfNormal(0, 0.5)
//     - LKJ correlation        : LKJ(2)         [mild regularisation]
//     - phi (Beta precision)   : Gamma(2, 0.1)  [allows broad dispersion]
// =============================================================================

data {
  // ── Dimensions ─────────────────────────────────────────────────────────────
  int<lower=1> N;          // number of songs
  int<lower=1> J;          // number of artists
  int<lower=1> G;          // number of genres
  int<lower=1> K;          // number of B-spline basis functions (trajectory)

  // ── Outcome ────────────────────────────────────────────────────────────────
  vector<lower=0,upper=1>[N] y;   // popularity scaled to (0,1)

  // ── Song-level predictors (all continuous already z-scored in R) ──────────
  vector[N] lmc_mean_z;          // mean LMC across lines
  vector[N] lmc_consistency_z;   // SD of smooth LMC (low = consistent)
  vector[N] energy_z;
  vector[N] valence_z;
  vector[N] danceability_z;
  vector[N] song_age_z;
  vector[N] orientation;         // 1 = narrative, 0 = production

  // ── Functional trajectory: B-spline projection ───────────────────────────
  // B[i, k] = k-th basis function evaluated at song i's LMC trajectory
  // (precomputed in R as the integral of basis × LMC curve)
  matrix[N, K] B;                // N × K basis projection matrix

  // ── Group indices ──────────────────────────────────────────────────────────
  array[N] int<lower=1,upper=J> artist_id;
  array[N] int<lower=1,upper=G> genre_id;
  array[J] int<lower=1,upper=G> artist_genre_id;  // genre of each artist

  // ── Artist-level predictor ─────────────────────────────────────────────────
  vector[J] artist_orientation;   // orientation of each artist (0/1)
}

transformed data {
  // Centre orientation for numerical stability
  real mean_orientation = mean(orientation);
  vector[N] orientation_c = orientation - mean_orientation;
  vector[J] artist_orient_c = artist_orientation - mean_orientation;
}

parameters {
  // ── Global intercept ───────────────────────────────────────────────────────
  real alpha_global;

  // ── Fixed effect slopes ────────────────────────────────────────────────────
  real beta_lmc;              // LMC mean effect
  real beta_lmc_orient;       // LMC × orientation interaction
  real beta_consistency;      // LMC consistency (low SD = more consistent)
  real beta_energy;
  real beta_valence;
  real beta_danceability;
  real beta_age;
  real beta_orientation;      // main effect of orientation

  // ── Functional trajectory coefficients (K-vector, smoothness-penalised) ───
  vector[K] theta;            // B-spline weights for LMC trajectory

  // ── Artist random effects: intercept + LMC slope (correlated) ────────────
  matrix[J, 2] z_artist;     // non-centred parameterisation
  vector<lower=0>[2] sigma_artist;
  cholesky_factor_corr[2] L_artist;

  // ── Genre random intercepts ───────────────────────────────────────────────
  vector[G] z_genre;
  real<lower=0> sigma_genre;

  // ── Artist intercept level-2 predictor ────────────────────────────────────
  real gamma_orient;          // orientation effect on artist intercept

  // ── Beta precision ────────────────────────────────────────────────────────
  real<lower=0> phi;

  // ── Smoothness penalty parameter (RW1 prior on theta) ────────────────────
  real<lower=0> tau;          // controls wiggliness of trajectory effect
}

transformed parameters {
  // ── Non-centred artist random effects ─────────────────────────────────────
  // Column 1 = artist intercept offset, column 2 = artist LMC slope offset
  matrix[J, 2] u_artist = (diag_pre_multiply(sigma_artist, L_artist) * z_artist')';

  // ── Genre intercepts ───────────────────────────────────────────────────────
  vector[G] u_genre = sigma_genre * z_genre;

  // ── Linear predictor ──────────────────────────────────────────────────────
  vector[N] eta;
  for (i in 1:N) {
    int j = artist_id[i];
    int g = genre_id[i];

    // Artist intercept: global + level-2 orientation + random deviate
    real artist_intercept = alpha_global
                          + gamma_orient * artist_orient_c[j]
                          + u_artist[j, 1]
                          + u_genre[g];

    // LMC slope: fixed + artist-varying deviate
    real lmc_slope_j = beta_lmc + u_artist[j, 2];

    eta[i] = artist_intercept
           + lmc_slope_j       * lmc_mean_z[i]
           + beta_lmc_orient   * lmc_mean_z[i] * orientation_c[i]
           + beta_consistency  * lmc_consistency_z[i]
           + beta_energy       * energy_z[i]
           + beta_valence      * valence_z[i]
           + beta_danceability * danceability_z[i]
           + beta_age          * song_age_z[i]
           + beta_orientation  * orientation_c[i]
           + dot_product(B[i], theta);   // functional trajectory term
  }
}

model {
  // ── Likelihood ─────────────────────────────────────────────────────────────
  vector[N] mu = inv_logit(eta);
  // Beta parameterisation: alpha = mu*phi, beta = (1-mu)*phi
  y ~ beta(mu * phi, (1 - mu) * phi);

  // ── Priors: fixed effects ──────────────────────────────────────────────────
  alpha_global    ~ normal(0, 1.5);
  beta_lmc        ~ normal(0, 0.5);    // expect positive, modest effect
  beta_lmc_orient ~ normal(0, 0.5);
  beta_consistency ~ normal(0, 0.5);
  beta_energy     ~ normal(0, 0.5);
  beta_valence    ~ normal(0, 0.5);
  beta_danceability ~ normal(0, 0.5);
  beta_age        ~ normal(0, 0.5);
  beta_orientation ~ normal(0, 0.5);

  // ── Prior: functional trajectory (RW1 smoothness) ─────────────────────────
  // Penalises large differences between adjacent spline coefficients.
  // tau controls the effective degree of smoothness — high tau = smoother.
  tau    ~ normal(0, 1);   // half-normal because tau > 0
  theta[1] ~ normal(0, 1); // anchor first coefficient
  for (k in 2:K)
    theta[k] ~ normal(theta[k-1], tau);   // RW1 prior

  // ── Priors: random effects ────────────────────────────────────────────────
  sigma_artist ~ normal(0, 0.5);    // HalfNormal (implied by prior + constraint)
  sigma_genre  ~ normal(0, 0.5);
  L_artist     ~ lkj_corr_cholesky(2);    // mild regularisation on correlation
  gamma_orient ~ normal(0, 0.5);

  // Non-centred parameterisation for artist random effects
  to_vector(z_artist) ~ std_normal();
  z_genre ~ std_normal();

  // ── Prior: Beta precision ─────────────────────────────────────────────────
  phi ~ gamma(2, 0.1);    // mean=20, SD~14; allows moderate to high precision
}

generated quantities {
  // ── Log-likelihood for LOO-CV ─────────────────────────────────────────────
  vector[N] log_lik;
  vector[N] y_rep;         // posterior predictive replications
  vector[N] mu_hat;        // posterior mean predictions

  for (i in 1:N) {
    real mu_i  = inv_logit(eta[i]);
    real a     = mu_i * phi;
    real b     = (1 - mu_i) * phi;
    log_lik[i] = beta_lpdf(y[i] | a, b);
    y_rep[i]   = beta_rng(a, b);
    mu_hat[i]  = mu_i;
  }

  // ── Derived quantities of interest ────────────────────────────────────────
  // Artist-level correlation between intercept and LMC slope
  real rho_artist = L_artist[2, 1];   // off-diagonal of Cholesky factor

  // Marginal SD of popularity predictions (on probability scale)
  // Useful for interpreting effect sizes
  real sd_y_rep = sd(y_rep);

  // Ratio of artist to genre variance (ICC decomposition)
  real var_artist = square(sigma_artist[1]);
  real var_genre  = square(sigma_genre);
  real icc_artist = var_artist / (var_artist + var_genre + square(pi()) / 3.0);
}
