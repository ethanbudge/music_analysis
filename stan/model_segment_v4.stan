// =============================================================================
// model_segment_v4.stan — Chorus vs. non-chorus LMC → popularity (generic X).
//
// v3 segment structure with the hardcoded controls replaced by the generic
// control matrix X[N, K] / beta_ctrl[K] (see model_track_v4.stan). The chorus
// slope varies by genre and is moderated by orientation; the non-chorus slope
// is a single coefficient. This is the "segment only" model in the
// segment / curvature / segment+curvature comparison.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  vector[N] chorus_lmc_z;
  vector[N] nonchorus_lmc_z;

  int<lower=0> K;
  matrix[N, K] X;

  array[N] int<lower=1, upper=N_genre> genre_id;
  vector<lower=0, upper=1>[N] orientation;
  vector<lower=0, upper=1>[N] orientation_known;
}

parameters {
  real mu_global;
  real beta_chorus;
  real beta_nonchorus;
  vector[K] beta_ctrl;

  real gamma_intercept;
  real gamma_chorus;

  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;

  vector[N_genre] z_chorus_genre;
  real<lower=0> sigma_chorus_genre;

  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  vector[N_genre] alpha_genre       = sigma_genre * z_genre;
  vector[N_genre] beta_chorus_genre = beta_chorus + sigma_chorus_genre * z_chorus_genre;
  vector[N_genre] log_phi_genre     = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] orient_k = orientation .* orientation_known;
  vector[N] ctrl = K > 0 ? X * beta_ctrl : rep_vector(0.0, N);

  vector[N] eta;
  vector<lower=0>[N] phi;
  for (i in 1:N) {
    real chorus_slope = beta_chorus_genre[genre_id[i]] + gamma_chorus * orient_k[i];
    eta[i] = mu_global
             + gamma_intercept * orient_k[i]
             + alpha_genre[genre_id[i]]
             + chorus_slope    * chorus_lmc_z[i]
             + beta_nonchorus  * nonchorus_lmc_z[i]
             + ctrl[i];
    phi[i] = exp(log_phi_genre[genre_id[i]]);
  }
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_chorus     ~ normal(0, 0.5);
  beta_nonchorus  ~ normal(0, 0.5);
  beta_ctrl       ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);
  gamma_chorus    ~ normal(0, 0.5);

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);

  z_chorus_genre     ~ std_normal();
  sigma_chorus_genre ~ normal(0, 0.3);

  phi_intercept   ~ normal(1.8, 0.5);
  z_phi_genre     ~ std_normal();
  sigma_phi_genre ~ normal(0, 0.5);

  y ~ beta(mu .* phi, (1.0 - mu) .* phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  real chorus_vs_nonchorus = beta_chorus - beta_nonchorus;
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real b = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
  }
}
