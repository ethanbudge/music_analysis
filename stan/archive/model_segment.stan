// =============================================================================
// model_segment.stan — Chorus vs. non-chorus LMC → popularity (Beta regression)
//
// Tests whether congruence *in the chorus* matters more than congruence in the
// rest of the song. Both segment LMC measures (seg_chorus, seg_nonchorus) enter
// simultaneously; the chorus slope varies by genre and is moderated by
// orientation. Same genre/artist hierarchy and controls as model_track.stan.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
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

  array[N] int<lower=1, upper=N_artist> artist_id;
  array[N] int<lower=1, upper=N_genre>  genre_id;

  vector<lower=0, upper=1>[N] orientation;
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

  vector[N_artist] z_artist;
  real<lower=0> sigma_artist;

  vector[N_genre] alpha_genre;
  real<lower=0> sigma_genre;

  vector[N_genre] z_chorus_genre;
  real<lower=0> sigma_chorus_genre;

  real<lower=0> phi;
}

transformed parameters {
  vector[N_artist] alpha_artist = sigma_artist * z_artist;

  vector[N_genre] beta_chorus_genre;
  for (g in 1:N_genre)
    beta_chorus_genre[g] = beta_chorus + sigma_chorus_genre * z_chorus_genre[g];

  vector[N] eta;
  for (i in 1:N) {
    real chorus_slope_i = beta_chorus_genre[genre_id[i]] + gamma_chorus * orientation[i];
    eta[i] = mu_global
             + gamma_intercept * orientation[i]
             + alpha_artist[artist_id[i]]
             + alpha_genre[genre_id[i]]
             + chorus_slope_i  * chorus_lmc_z[i]
             + beta_nonchorus  * nonchorus_lmc_z[i]
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

  z_artist     ~ std_normal();
  sigma_artist ~ normal(0, 0.5);
  alpha_genre  ~ normal(0, sigma_genre);
  sigma_genre  ~ normal(0, 0.5);

  z_chorus_genre     ~ std_normal();
  sigma_chorus_genre ~ normal(0, 0.3);

  phi ~ gamma(4, 0.1);

  y ~ beta(mu * phi, (1.0 - mu) * phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  real chorus_vs_nonchorus = beta_chorus - beta_nonchorus;
  for (i in 1:N) {
    real a = fmax(mu[i] * phi, 1e-6);
    real b = fmax((1.0 - mu[i]) * phi, 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
  }
}
