// =============================================================================
// model_segment.stan — Segment-level LMC → Popularity (MuLan)
//
// Rationale
// ---------
// This model decomposes the single track-level LMC score into
// structurally meaningful components: chorus LMC, verse LMC, and the
// consistency of congruence across sections. This tests a richer
// hypothesis than the track model:
//
//   H1: Not all parts of the song contribute equally to listener
//       engagement. Chorus congruence should matter more than verse
//       congruence because choruses are:
//       (a) the most repeated section — higher exposure = stronger
//           fluency effects (Zajonc mere exposure)
//       (b) the section most likely to generate sing-along engagement
//       (c) the section Spotify's algorithm detects for "song anchor"
//
//   H2: Consistency matters. A song with uniformly moderate congruence
//       may outperform one with wild swings between high and low LMC,
//       because inconsistency disrupts processing fluency.
//
// When chorus or verse LMC is unavailable for a song (because Genius
// didn't label sections), the R code fills those values with mean LMC
// and flags them. The Stan model handles this gracefully because the
// values are still on the same scale — it just means those songs contribute
// less distinguishing information for the chorus vs. verse comparison.
//
// Structure
// ---------
//   logit(μ_i) = α[artist] + α[genre]
//                + β_chorus · chorus_lmc_z
//                + β_verse  · verse_lmc_z
//                + β_sd     · lmc_sd_z
//                + β_age    · song_age_z
//                + γ_lmc    · orientation (moderates chorus effect)
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  vector[N] chorus_lmc_z;    // mean LMC for chorus sections
  vector[N] verse_lmc_z;     // mean LMC for verse sections
  vector[N] lmc_sd_z;        // SD of LMC across all sections (consistency)
  vector[N] song_age_z;

  array[N] int<lower=1, upper=N_artist> artist_id;
  array[N] int<lower=1, upper=N_genre>  genre_id;

  vector<lower=0, upper=1>[N_artist] orientation;
}

parameters {
  real mu_global;

  real beta_chorus;      // chorus congruence effect
  real beta_verse;       // verse congruence effect
  real beta_sd;          // congruence consistency (negative = lower SD helps)
  real beta_age;

  real gamma_intercept;
  real gamma_chorus;     // orientation × chorus interaction

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

  vector[N] eta;
  for (i in 1:N) {
    int j = artist_id[i];
    eta[i] = alpha_artist[j]
             + alpha_genre[genre_id[i]]
             + (beta_chorus + gamma_chorus * orientation[j]) * chorus_lmc_z[i]
             + beta_verse * verse_lmc_z[i]
             + beta_sd    * lmc_sd_z[i]
             + beta_age   * song_age_z[i];
  }

  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_chorus     ~ normal(0, 0.5);
  beta_verse      ~ normal(0, 0.5);
  beta_sd         ~ normal(0, 0.5);
  beta_age        ~ normal(0, 0.5);
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

  // Marginal effect of chorus LMC (the primary coefficient of interest)
  vector[N] marginal_chorus;

  // Contrast: how much more does chorus LMC matter than verse LMC?
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
