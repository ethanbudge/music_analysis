// =============================================================================
// model_track_v4.stan — Single-measure LMC → popularity, GENERIC control matrix.
//
// Same reparameterized structure as v3 (no artist effect, non-centred genre,
// genre-varying precision submodel, recalibrated priors) but the hardcoded mood
// + song-age controls are replaced by a generic design matrix X[N, K] with a
// single coefficient vector beta_ctrl[K]. This is what lets run_models.R toggle
// the control set (mood / MERT PCs / both / none) WITHOUT touching the model.
//
// `lmc_z` is one standardised LMC measure (song-wide or a line-window mean).
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  vector[N] lmc_z;

  int<lower=0> K;                 // number of control covariates
  matrix[N, K] X;                 // control design matrix (mood and/or MERT PCs + age)

  array[N] int<lower=1, upper=N_genre> genre_id;
  vector<lower=0, upper=1>[N] orientation;
  vector<lower=0, upper=1>[N] orientation_known;
}

parameters {
  real mu_global;
  real beta_lmc;
  vector[K] beta_ctrl;

  real gamma_intercept;
  real gamma_lmc;

  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;

  vector[N_genre] z_lmc_genre;
  real<lower=0> sigma_lmc_genre;

  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  vector[N_genre] alpha_genre    = sigma_genre * z_genre;
  vector[N_genre] beta_lmc_genre = beta_lmc + sigma_lmc_genre * z_lmc_genre;
  vector[N_genre] log_phi_genre  = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] orient_k = orientation .* orientation_known;
  vector[N] ctrl = K > 0 ? X * beta_ctrl : rep_vector(0.0, N);

  vector[N] eta;
  vector<lower=0>[N] phi;
  for (i in 1:N) {
    real lmc_slope = beta_lmc_genre[genre_id[i]] + gamma_lmc * orient_k[i];
    eta[i] = mu_global
             + gamma_intercept * orient_k[i]
             + alpha_genre[genre_id[i]]
             + lmc_slope * lmc_z[i]
             + ctrl[i];
    phi[i] = exp(log_phi_genre[genre_id[i]]);
  }
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_lmc        ~ normal(0, 0.5);
  beta_ctrl       ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);
  gamma_lmc       ~ normal(0, 0.5);

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);

  z_lmc_genre     ~ std_normal();
  sigma_lmc_genre ~ normal(0, 0.3);

  phi_intercept   ~ normal(1.8, 0.5);
  z_phi_genre     ~ std_normal();
  sigma_phi_genre ~ normal(0, 0.5);

  y ~ beta(mu .* phi, (1.0 - mu) .* phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] marginal_lmc;
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real b = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
    real lmc_slope = beta_lmc_genre[genre_id[i]] + gamma_lmc * orient_k[i];
    marginal_lmc[i] = lmc_slope * mu[i] * (1.0 - mu[i]) * 100.0;
  }
}
