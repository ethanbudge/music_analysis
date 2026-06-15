// =============================================================================
// model_segment_v2.stan — Segment-level LMC → Popularity (MuLan)
//
// Decomposes track-level LMC into chorus, verse, and consistency.
// Includes mood tag controls. No quadratic term.
//
// This is the model used for simulation-based parameter recovery
// (see simulation section of report.qmd).
//
// Partial pooling (identical structure to model_track_v2)
// -------------------------------------------------------
//   Artist: NCP random intercepts
//     α_artist[j] = μ_global + γ_int · orient[j] + σ_artist · z[j]
//
//   Genre: hierarchical random intercepts
//     α_genre[g] ~ N(0, σ_genre)
//
// Key generated quantity: chorus_vs_verse
//   = β_chorus − β_verse
//   Posterior probability that this > 0 answers: "does chorus
//   congruence matter more than verse congruence?"
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  // Segment-level LMC predictors (z-scored)
  vector[N] chorus_lmc_z;
  vector[N] verse_lmc_z;
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

  // Segment LMC effects
  real beta_chorus;
  real beta_verse;
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
  real gamma_chorus;     // orientation × chorus LMC

  // Artist random intercepts (NCP)
  vector[N_artist] z_artist;
  real<lower=0> sigma_artist;

  // Genre random intercepts
  vector[N_genre] alpha_genre;
  real<lower=0> sigma_genre;

  real<lower=0> phi;
}

transformed parameters {
  vector[N_artist] alpha_artist;
  for (j in 1:N_artist)
    alpha_artist[j] = mu_global
                      + gamma_intercept * orientation[j]
                      + sigma_artist * z_artist[j];

  vector[N] eta;
  for (i in 1:N) {
    int j = artist_id[i];
    eta[i] = alpha_artist[j]
             + alpha_genre[genre_id[i]]
             + (beta_chorus + gamma_chorus * orientation[j]) * chorus_lmc_z[i]
             + beta_verse      * verse_lmc_z[i]
             + beta_sd         * lmc_sd_z[i]
             + beta_age        * song_age_z[i]
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
  beta_chorus     ~ normal(0, 0.5);
  beta_verse      ~ normal(0, 0.5);
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
  gamma_chorus    ~ normal(0, 0.5);

  z_artist     ~ std_normal();
  sigma_artist ~ normal(0, 0.5);
  alpha_genre  ~ normal(0, sigma_genre);
  sigma_genre  ~ normal(0, 0.5);
  phi          ~ gamma(4, 0.1);

  y ~ beta(mu * phi, (1.0 - mu) * phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] marginal_chorus;

  // Direct contrast: does chorus LMC matter more than verse LMC?
  real chorus_vs_verse = beta_chorus - beta_verse;

  for (i in 1:N) {
    real a = fmax(mu[i] * phi, 1e-6);
    real b = fmax((1.0 - mu[i]) * phi, 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
    marginal_chorus[i] = (beta_chorus + gamma_chorus * orientation[artist_id[i]])
                         * mu[i] * (1.0 - mu[i]) * 100.0;
  }
}
