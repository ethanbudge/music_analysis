// =============================================================================
// model_timeline_v2.stan — Lyric Timeline LMC → Popularity (MuLan)
//
// Uses WhisperX-derived trajectory shape features:
//   mean_lmc, slope, curve, early→late change, consistency (SD)
// Includes mood tag controls. No quadratic term.
//
// This is the richest model — it adds bivariate artist random effects
// (intercept + LMC slope) to capture artist-level heterogeneity in
// how much congruence matters. The two random effects are allowed to
// correlate, estimated via a Cholesky-factored correlation matrix.
//
// Partial pooling
// ---------------
//   Artist: bivariate NCP random effects
//     [α_artist[j], β_lmc_artist[j]] = means + diag(σ) · L · z[j]
//     where L is the lower Cholesky factor of the 2×2 correlation matrix.
//     This pools BOTH the baseline AND the LMC slope toward their
//     orientation-conditioned means. The correlation Rho_artist[1,2]
//     captures whether high-baseline artists also show stronger LMC effects.
//
//   Genre: random intercepts (same as other models)
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  // Trajectory features (z-scored)
  vector[N] mean_lmc_z;
  vector[N] lmc_slope_z;
  vector[N] lmc_curve_z;
  vector[N] lmc_change_z;
  vector[N] lmc_sd_z;

  // Controls
  vector[N] song_age_z;
  vector[N] mood_happy_z;
  vector[N] mood_sad_z;
  vector[N] mood_relaxed_z;
  vector[N] mood_aggressive_z;
  vector[N] mood_party_z;
  vector[N] danceability_z;
  vector[N] voice_instr_z;

  // Group indices
  array[N] int<lower=1, upper=N_artist> artist_id;
  array[N] int<lower=1, upper=N_genre>  genre_id;

  vector<lower=0, upper=1>[N_artist] orientation;
}

parameters {
  real mu_global;

  // Population-level trajectory effects
  real beta_lmc;
  real beta_slope;
  real beta_curve;
  real beta_change;
  real beta_sd;

  // Controls
  real beta_age;
  real beta_happy;
  real beta_sad;
  real beta_relaxed;
  real beta_aggressive;
  real beta_party;
  real beta_dance;
  real beta_voice;

  // Orientation moderation
  real gamma_intercept;
  real gamma_lmc;

  // Bivariate artist RE (NCP)
  matrix[2, N_artist] z_artist;
  vector<lower=0>[2] sigma_artist;
  cholesky_factor_corr[2] L_artist;

  // Genre random intercepts
  vector[N_genre] alpha_genre;
  real<lower=0> sigma_genre;

  real<lower=0> phi;
}

transformed parameters {
  matrix[2, N_artist] artist_re;
  artist_re = diag_pre_multiply(sigma_artist, L_artist) * z_artist;

  vector[N_artist] alpha_artist;
  vector[N_artist] beta_lmc_artist;

  for (j in 1:N_artist) {
    alpha_artist[j]    = mu_global
                         + gamma_intercept * orientation[j]
                         + artist_re[1, j];
    beta_lmc_artist[j] = beta_lmc
                         + gamma_lmc * orientation[j]
                         + artist_re[2, j];
  }

  vector[N] eta;
  for (i in 1:N) {
    int j = artist_id[i];
    eta[i] = alpha_artist[j]
             + alpha_genre[genre_id[i]]
             + beta_lmc_artist[j] * mean_lmc_z[i]
             + beta_slope  * lmc_slope_z[i]
             + beta_curve  * lmc_curve_z[i]
             + beta_change * lmc_change_z[i]
             + beta_sd     * lmc_sd_z[i]
             + beta_age    * song_age_z[i]
             + beta_happy      * mood_happy_z[i]
             + beta_sad        * mood_sad_z[i]
             + beta_relaxed    * mood_relaxed_z[i]
             + beta_aggressive * mood_aggressive_z[i]
             + beta_party      * mood_party_z[i]
             + beta_dance      * danceability_z[i]
             + beta_voice      * voice_instr_z[i];
  }

  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_lmc        ~ normal(0, 0.5);
  beta_slope      ~ normal(0, 0.5);
  beta_curve      ~ normal(0, 0.5);
  beta_change     ~ normal(0, 0.5);
  beta_sd         ~ normal(0, 0.5);
  beta_age        ~ normal(0, 0.5);
  beta_happy      ~ normal(0, 0.5);
  beta_sad        ~ normal(0, 0.5);
  beta_relaxed    ~ normal(0, 0.5);
  beta_aggressive ~ normal(0, 0.5);
  beta_party      ~ normal(0, 0.5);
  beta_dance      ~ normal(0, 0.5);
  beta_voice      ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);
  gamma_lmc       ~ normal(0, 0.5);

  to_vector(z_artist) ~ std_normal();
  sigma_artist        ~ normal(0, 0.5);
  L_artist            ~ lkj_corr_cholesky(2);

  alpha_genre ~ normal(0, sigma_genre);
  sigma_genre ~ normal(0, 0.5);

  phi ~ gamma(4, 0.1);

  y ~ beta(mu * phi, (1.0 - mu) * phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  matrix[2, 2] Rho_artist;
  vector[N] marginal_lmc;

  Rho_artist = multiply_lower_tri_self_transpose(L_artist);

  for (i in 1:N) {
    real a = fmax(mu[i] * phi, 1e-6);
    real b = fmax((1.0 - mu[i]) * phi, 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
    marginal_lmc[i] = beta_lmc_artist[artist_id[i]]
                      * mu[i] * (1.0 - mu[i]) * 100.0;
  }
}
