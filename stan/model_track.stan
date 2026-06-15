// =============================================================================
// model_track.stan — Single-measure LMC → Spotify popularity (Beta regression)
//
// Generic track-level model: `lmc_z` is ONE standardised LMC measure chosen by
// the analyst when preparing the data — song-wide, any line-level window mean
// (exact / buf1 / buf5 / buf10), or a segment measure. Fitting it once per
// measure and comparing by LOO answers "which congruence method best predicts
// popularity?". Used for both MuLan and CLAP.
//
// Structure (adapted from the v2 family for the broad LRCLIB sample)
// ------------------------------------------------------------------
//   • Genre is the primary grouping factor (random intercept + genre-varying
//     LMC slope, non-centred).
//   • Artist random intercepts are retained but shrink toward zero where an
//     artist contributes a single song (common in a broad sample).
//   • Orientation (narrative=1 / production=0) is now a *song-level* recovered
//     covariate, moderating both the intercept and the LMC slope.
//   • librosa mood proxies + song age are controls.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;            // popularity / 100 (boundary-adjusted)

  vector[N] lmc_z;                          // the chosen standardised LMC measure
  vector[N] song_age_z;
  vector[N] mood_happy_z;
  vector[N] mood_sad_z;
  vector[N] mood_relaxed_z;
  vector[N] mood_aggressive_z;
  vector[N] mood_party_z;
  vector[N] danceability_z;
  vector[N] voice_instr_z;

  array[N] int<lower=1, upper=N_artist> artist_id;
  array[N] int<lower=1, upper=N_genre>  genre_id;

  vector<lower=0, upper=1>[N] orientation;  // song-level narrative(1)/production(0)
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

  real gamma_intercept;     // orientation → intercept
  real gamma_lmc;           // orientation → LMC slope

  vector[N_artist] z_artist;
  real<lower=0> sigma_artist;

  vector[N_genre] alpha_genre;
  real<lower=0> sigma_genre;

  vector[N_genre] z_lmc_genre;          // genre-varying LMC slope (NCP)
  real<lower=0> sigma_lmc_genre;

  real<lower=0> phi;
}

transformed parameters {
  vector[N_artist] alpha_artist = sigma_artist * z_artist;

  vector[N_genre] beta_lmc_genre;
  for (g in 1:N_genre)
    beta_lmc_genre[g] = beta_lmc + sigma_lmc_genre * z_lmc_genre[g];

  vector[N] eta;
  for (i in 1:N) {
    real lmc_slope_i = beta_lmc_genre[genre_id[i]] + gamma_lmc * orientation[i];
    eta[i] = mu_global
             + gamma_intercept * orientation[i]
             + alpha_artist[artist_id[i]]
             + alpha_genre[genre_id[i]]
             + lmc_slope_i * lmc_z[i]
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

  z_artist     ~ std_normal();
  sigma_artist ~ normal(0, 0.5);
  alpha_genre  ~ normal(0, sigma_genre);
  sigma_genre  ~ normal(0, 0.5);

  z_lmc_genre     ~ std_normal();
  sigma_lmc_genre ~ normal(0, 0.3);

  phi ~ gamma(4, 0.1);

  y ~ beta(mu * phi, (1.0 - mu) * phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] marginal_lmc;                  // approx. percentage-point effect of LMC
  for (i in 1:N) {
    real a = fmax(mu[i] * phi, 1e-6);
    real b = fmax((1.0 - mu[i]) * phi, 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
    real lmc_slope_i = beta_lmc_genre[genre_id[i]] + gamma_lmc * orientation[i];
    marginal_lmc[i] = lmc_slope_i * mu[i] * (1.0 - mu[i]) * 100.0;
  }
}
