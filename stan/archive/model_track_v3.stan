// =============================================================================
// model_track_v3.stan — Reparameterized single-measure LMC → popularity.
//
// Reference implementation of the recommendations in stan/MODEL_NOTES.md. It is a
// drop-in alternative to model_track.stan that targets the two reported problems
// (high Pareto-k everywhere, E-BFMI < 0.3 on all chains):
//
//   1. NO artist random effect. With ~94% singleton artists it was a near
//      per-observation parameter — the main driver of both bad diagnostics. Its
//      role is taken over by a precision submodel (below).
//   2. Genre intercept is now NON-CENTRED (was the one centred term left).
//   3. PRECISION SUBMODEL: phi varies by genre on the log scale instead of a
//      single global, mis-priored phi. Heteroscedastic popularity stops making
//      ordinary songs look like outliers ⇒ fewer influential points.
//   4. RE-CALIBRATED priors: phi is centred near the data-implied ~4–8, not ~40.
//   5. orientation_known indicator so the 38% "unknown" rows don't get a fake
//      0.5 covariate and don't pull the moderation toward zero.
//
// Data contract is the same as model_track.stan PLUS one new vector,
// `orientation_known` (1 if genre/orientation were recovered, else 0). Build it
// in R as as.numeric(df$orientation %in% c("narrative","production")).
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  vector[N] lmc_z;
  vector[N] song_age_z;
  vector[N] mood_happy_z;
  vector[N] mood_sad_z;
  vector[N] mood_relaxed_z;
  vector[N] mood_aggressive_z;
  vector[N] mood_party_z;
  vector[N] danceability_z;
  vector[N] voice_instr_z;

  array[N] int<lower=1, upper=N_genre> genre_id;

  vector<lower=0, upper=1>[N] orientation;        // narrative(1)/production(0); 0 where unknown
  vector<lower=0, upper=1>[N] orientation_known;   // 1 if recovered, else 0
}

parameters {
  real mu_global;
  real beta_lmc;
  real beta_age;
  real beta_happy;
  real beta_sad;
  real beta_relaxed;
  real beta_aggressive;
  real beta_party;
  real beta_dance;
  real beta_voice;

  real gamma_intercept;    // orientation → intercept (applied only where known)
  real gamma_lmc;          // orientation → LMC slope

  // Genre intercept (non-centred).
  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;

  // Genre-varying LMC slope (non-centred).
  vector[N_genre] z_lmc_genre;
  real<lower=0> sigma_lmc_genre;

  // Precision submodel: log(phi) = phi_intercept + genre deviation (non-centred).
  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  vector[N_genre] alpha_genre    = sigma_genre     * z_genre;
  vector[N_genre] beta_lmc_genre = beta_lmc + sigma_lmc_genre * z_lmc_genre;
  vector[N_genre] log_phi_genre  = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] eta;
  vector<lower=0>[N] phi;
  for (i in 1:N) {
    real orient_k   = orientation[i] * orientation_known[i];
    real lmc_slope  = beta_lmc_genre[genre_id[i]] + gamma_lmc * orient_k;
    eta[i] = mu_global
             + gamma_intercept * orient_k
             + alpha_genre[genre_id[i]]
             + lmc_slope * lmc_z[i]
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
  beta_lmc        ~ normal(0, 0.5);
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

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);

  z_lmc_genre     ~ std_normal();
  sigma_lmc_genre ~ normal(0, 0.3);

  // Precision: centred near the data-implied phi (~4–8), not the old ~40.
  phi_intercept   ~ normal(1.8, 0.5);   // exp(1.8) ≈ 6
  z_phi_genre     ~ std_normal();
  sigma_phi_genre ~ normal(0, 0.5);

  y ~ beta(mu .* phi, (1.0 - mu) .* phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] marginal_lmc;                  // approx. percentage-point effect of LMC
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real b = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
    real orient_k  = orientation[i] * orientation_known[i];
    real lmc_slope = beta_lmc_genre[genre_id[i]] + gamma_lmc * orient_k;
    marginal_lmc[i] = lmc_slope * mu[i] * (1.0 - mu[i]) * 100.0;
  }
}
