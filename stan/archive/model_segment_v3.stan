// =============================================================================
// model_segment_v3.stan — Reparameterized chorus vs. non-chorus LMC → popularity.
//
// Same reparameterization as model_track_v3.stan (see stan/MODEL_NOTES.md):
//   • artist random effect REMOVED (94% singletons drove the bad diagnostics);
//   • genre intercept NON-CENTRED;
//   • precision submodel: phi varies by genre on the log scale;
//   • RE-CALIBRATED phi prior (centred near the data-implied ~4–8, not ~40);
//   • orientation_known indicator so 38% "unknown" rows get no fake 0.5 covariate.
//
// The chorus slope still varies by genre and is moderated by orientation; the
// non-chorus slope is a single fixed coefficient. Data contract matches
// model_segment.stan PLUS `orientation_known`.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  vector[N] chorus_lmc_z;
  vector[N] nonchorus_lmc_z;
  vector[N] song_age_z;
  vector[N] mood_happy_z;
  vector[N] mood_sad_z;
  vector[N] mood_relaxed_z;
  vector[N] mood_aggressive_z;
  vector[N] mood_party_z;
  vector[N] danceability_z;
  vector[N] voice_instr_z;

  array[N] int<lower=1, upper=N_genre> genre_id;

  vector<lower=0, upper=1>[N] orientation;
  vector<lower=0, upper=1>[N] orientation_known;
}

parameters {
  real mu_global;
  real beta_chorus;
  real beta_nonchorus;
  real beta_age;
  real beta_happy;
  real beta_sad;
  real beta_relaxed;
  real beta_aggressive;
  real beta_party;
  real beta_dance;
  real beta_voice;

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

  vector[N] eta;
  vector<lower=0>[N] phi;
  for (i in 1:N) {
    real orient_k     = orientation[i] * orientation_known[i];
    real chorus_slope = beta_chorus_genre[genre_id[i]] + gamma_chorus * orient_k;
    eta[i] = mu_global
             + gamma_intercept * orient_k
             + alpha_genre[genre_id[i]]
             + chorus_slope    * chorus_lmc_z[i]
             + beta_nonchorus  * nonchorus_lmc_z[i]
             + beta_age        * song_age_z[i]
             + beta_happy      * mood_happy_z[i]
             + beta_sad        * mood_sad_z[i]
             + beta_relaxed    * mood_relaxed_z[i]
             + beta_aggressive * mood_aggressive_z[i]
             + beta_party      * mood_party_z[i]
             + beta_dance      * danceability_z[i]
             + beta_voice      * voice_instr_z[i];
    phi[i] = exp(log_phi_genre[genre_id[i]]);
  }
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_chorus     ~ normal(0, 0.5);
  beta_nonchorus  ~ normal(0, 0.5);
  beta_age        ~ normal(0, 0.5);
  beta_happy      ~ normal(0, 0.5);
  beta_sad        ~ normal(0, 0.5);
  beta_relaxed    ~ normal(0, 0.5);
  beta_aggressive ~ normal(0, 0.5);
  beta_party      ~ normal(0, 0.5);
  beta_dance      ~ normal(0, 0.5);
  beta_voice      ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);
  gamma_chorus    ~ normal(0, 0.5);

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);

  z_chorus_genre     ~ std_normal();
  sigma_chorus_genre ~ normal(0, 0.3);

  phi_intercept   ~ normal(1.8, 0.5);   // exp(1.8) ≈ 6
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
