// =============================================================================
// model_track.stan — Track-level LMC → Popularity
//
// Used for both Model 1 (MuLan) and Model 2 (CLAP). The two models
// share identical structure — only the data differ.
//
// Rationale
// ---------
// This is the simplest and most conservative test of the LMC hypothesis.
// Each song gets a single scalar LMC score: the cosine similarity between
// its full audio embedding and its full lyric embedding. No temporal or
// structural information is used.
//
// This model asks: does the overall alignment between what a song sounds
// like and what its lyrics say predict how many people listen to it?
//
// By running the same model on MuLan and CLAP scores, we test whether
// the choice of embedding model matters for the LMC construct.
//
// Structure
// ---------
//   Y_i ~ Beta(μ_i, φ)
//   logit(μ_i) = α[artist] + α[genre]
//                + β_lmc · LMC_z
//                + β_lmc2 · LMC_z²         [inverted-U test]
//                + β_age · song_age_z       [release recency control]
//                + γ_lmc · orientation       [moderation]
//
//   α[artist] ~ N(μ_global + γ_int · orientation, σ_artist)
//   α[genre]  ~ N(0, σ_genre)
//
// The quadratic β_lmc2 tests the Askin & Mauskapf (2017) optimal
// distinctiveness prediction: too much congruence is as bad as too little.
// If β_lmc2 < 0, the relationship is an inverted-U.
//
// song_age controls for the mechanical relationship between release
// recency and Spotify's popularity score (which is recency-weighted).
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  vector[N] lmc_z;
  vector[N] lmc_z2;
  vector[N] song_age_z;

  array[N] int<lower=1, upper=N_artist> artist_id;
  array[N] int<lower=1, upper=N_genre>  genre_id;

  vector<lower=0, upper=1>[N_artist] orientation;
}

parameters {
  real mu_global;
  real beta_lmc;
  real beta_lmc2;
  real beta_age;
  real gamma_intercept;
  real gamma_lmc;

  vector[N_artist] z_artist;
  real<lower=0> sigma_artist;

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

  // Artist-varying LMC slope: population slope + orientation shift
  // No random slope here (track-level model is intentionally lean)
  vector[N] eta;
  for (i in 1:N)
    eta[i] = alpha_artist[artist_id[i]]
             + alpha_genre[genre_id[i]]
             + (beta_lmc + gamma_lmc * orientation[artist_id[i]]) * lmc_z[i]
             + beta_lmc2 * lmc_z2[i]
             + beta_age  * song_age_z[i];

  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_lmc        ~ normal(0, 0.5);
  beta_lmc2       ~ normal(0, 0.25);
  beta_age        ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);
  gamma_lmc       ~ normal(0, 0.5);

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
