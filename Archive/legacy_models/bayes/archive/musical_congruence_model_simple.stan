// =============================================================================
// musical_congruence_model_simple.stan
//
// Bayesian Hierarchical Beta Regression — no audio feature controls.
// Use this when Spotify audio features (energy, valence, danceability)
// are unavailable.
//
// Structure
// ---------
//   Likelihood  : Beta regression (logit link)
//   Level 1     : Song — LMC level + shape predictors only
//   Level 2     : Artist — bivariate normal random effects (intercept +
//                 LMC slope), mean moderated by orientation
//   Level 3     : Genre — normal random intercepts
//   Nonlinearity: Quadratic term on mean LMC (tests inverted-U)
// =============================================================================

data {

  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;   // boundary-adjusted popularity

  // LMC predictors (all z-scored)
  vector[N] mean_lmc_z;
  vector[N] mean_lmc_z2;
  vector[N] lmc_slope_z;
  vector[N] lmc_curve_z;
  vector[N] lmc_change_z;
  vector[N] lmc_sd_z;

  // Group indices
  array[N] int<lower=1, upper=N_artist> artist_id;
  array[N] int<lower=1, upper=N_genre>  genre_id;

  // Artist-level covariate: 0 = production, 1 = narrative
  vector<lower=0, upper=1>[N_artist] orientation;
}

parameters {

  real mu_global;

  real beta_lmc;
  real beta_lmc2;
  real beta_slope;
  real beta_curve;
  real beta_change;
  real beta_sd;

  real gamma_intercept;
  real gamma_lmc;

  matrix[2, N_artist] z_artist;
  vector<lower=0>[2]  sigma_artist;
  cholesky_factor_corr[2] L_artist;

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
    eta[i] = alpha_artist[artist_id[i]]
             + alpha_genre[genre_id[i]]
             + beta_lmc_artist[artist_id[i]] * mean_lmc_z[i]
             + beta_lmc2   * mean_lmc_z2[i]
             + beta_slope  * lmc_slope_z[i]
             + beta_curve  * lmc_curve_z[i]
             + beta_change * lmc_change_z[i]
             + beta_sd     * lmc_sd_z[i];
  }

  vector<lower=0, upper=1>[N] mu_pred;
  mu_pred = inv_logit(eta);
}

model {

  mu_global        ~ normal(0, 1.5);
  beta_lmc         ~ normal(0, 0.5);
  beta_lmc2        ~ normal(0, 0.25);
  beta_slope       ~ normal(0, 0.5);
  beta_curve       ~ normal(0, 0.5);
  beta_change      ~ normal(0, 0.5);
  beta_sd          ~ normal(0, 0.5);
  gamma_intercept  ~ normal(0, 0.5);
  gamma_lmc        ~ normal(0, 0.5);

  to_vector(z_artist) ~ std_normal();
  sigma_artist        ~ normal(0, 0.5);
  L_artist            ~ lkj_corr_cholesky(2);

  alpha_genre ~ normal(0, sigma_genre);
  sigma_genre ~ normal(0, 0.5);

  phi ~ gamma(4, 0.1);

  y ~ beta(mu_pred * phi, (1.0 - mu_pred) * phi);
}

generated quantities {

  vector[N] y_rep;
  vector[N] log_lik;
  matrix[2, 2] Rho_artist;
  vector[N] marginal_lmc;

  Rho_artist = multiply_lower_tri_self_transpose(L_artist);

  for (i in 1:N) {
    real a = fmax(mu_pred[i] * phi, 1e-6);
    real b = fmax((1.0 - mu_pred[i]) * phi, 1e-6);

    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);

    marginal_lmc[i] = beta_lmc_artist[artist_id[i]]
                      * mu_pred[i] * (1.0 - mu_pred[i])
                      * 100.0;
  }
}
