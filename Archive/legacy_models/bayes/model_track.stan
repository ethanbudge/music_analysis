// =============================================================================
// model_track_v2.stan — Track-level LMC → Popularity
//
// Used for both MuLan (M1) and CLAP (M2) — identical structure, different data.
//
// Changes from v1
// ----------------
//   • Removed β_lmc2 (quadratic / inverted-U term)
//   • Added 7 Essentia mood tag controls
//   • Verified partial pooling structure
//
// Partial pooling
// ---------------
//   Artist: random intercepts via NCP (non-centred parameterisation)
//     α_artist[j] = μ_global + γ_int · orient[j] + σ_artist · z_artist[j]
//     where z_artist[j] ~ N(0,1)
//     This pools artist baselines toward the grand mean, with shrinkage
//     controlled by σ_artist. Artists with fewer songs are shrunk more.
//
//   Genre: random intercepts
//     α_genre[g] ~ N(0, σ_genre)
//     Pools genre baselines toward zero (absorbed into μ_global).
//     With only 4 genres, strong pooling is appropriate — the data
//     are too sparse to estimate 4 genre effects without regularisation.
//
//   Both levels are estimated simultaneously; the shrinkage is automatic
//   and data-adaptive. This is the Bayesian equivalent of crossed random
//   effects in lme4: (1 | artist) + (1 | genre).
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  // LMC predictor (z-scored)
  vector[N] lmc_z;

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

  // Artist-level: orientation (0 = production, 1 = narrative)
  vector<lower=0, upper=1>[N_artist] orientation;
}

parameters {
  real mu_global;

  // LMC effect
  real beta_lmc;

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
  real gamma_intercept;   // orientation → baseline popularity
  real gamma_lmc;         // orientation × LMC interaction

  // Artist random intercepts (NCP)
  vector[N_artist] z_artist;
  real<lower=0> sigma_artist;

  // Genre random intercepts
  vector[N_genre] alpha_genre;
  real<lower=0> sigma_genre;

  // Beta precision
  real<lower=0> phi;
}

transformed parameters {
  // ── Reconstruct artist intercepts ──────────────────────────────────────
  vector[N_artist] alpha_artist;
  for (j in 1:N_artist)
    alpha_artist[j] = mu_global
                      + gamma_intercept * orientation[j]
                      + sigma_artist * z_artist[j];

  // ── Linear predictor ───────────────────────────────────────────────────
  vector[N] eta;
  for (i in 1:N) {
    int j = artist_id[i];
    eta[i] = alpha_artist[j]
             + alpha_genre[genre_id[i]]
             + (beta_lmc + gamma_lmc * orientation[j]) * lmc_z[i]
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
  // ── Priors ─────────────────────────────────────────────────────────────
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

  // Artist NCP
  z_artist     ~ std_normal();
  sigma_artist ~ normal(0, 0.5);

  // Genre pooling
  alpha_genre ~ normal(0, sigma_genre);
  sigma_genre ~ normal(0, 0.5);

  // Precision
  phi ~ gamma(4, 0.1);

  // ── Likelihood ─────────────────────────────────────────────────────────
  y ~ beta(mu * phi, (1.0 - mu) * phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] marginal_lmc;

  for (i in 1:N) {
    real a = fmax(mu[i] * phi, 1e-6);
    real b = fmax((1.0 - mu[i]) * phi, 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
    marginal_lmc[i] = (beta_lmc + gamma_lmc * orientation[artist_id[i]])
                      * mu[i] * (1.0 - mu[i]) * 100.0;
  }
}
