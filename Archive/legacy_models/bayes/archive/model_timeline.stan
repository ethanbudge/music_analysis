// =============================================================================
// model_timeline.stan — Lyric Timeline LMC → Popularity (MuLan)
//
// Rationale
// ---------
// This model uses the richest representation of LMC: the trajectory of
// congruence over the course of the song, measured line by line via
// WhisperX forced alignment. From each song's timeline, five features
// are extracted:
//
//   mean_lmc       The overall level of congruence (replicates Models 1/2)
//   lmc_slope      Does LMC trend upward or downward across the song?
//                  A positive slope means the song "builds" toward
//                  greater alignment. Theory: rising LMC mirrors
//                  narrative arc and may produce a satisfying resolution.
//   lmc_curve      Quadratic curvature of the LMC trajectory.
//                  Negative = inverted-U arc (peaks mid-song).
//                  This tests whether mid-song congruence peaks
//                  (typically at the chorus) drive engagement.
//   lmc_change     Late-half minus early-half mean LMC.
//                  Correlated with slope but distinct: captures whether
//                  the payoff is in the second half regardless of the
//                  trajectory shape. Relates to the "peak-end rule"
//                  (Kahneman) — listeners may weight later moments more.
//   lmc_sd         Standard deviation of smoothed LMC trajectory.
//                  Low SD = congruence is consistent throughout.
//                  High SD = congruence is volatile.
//                  Processing fluency theory predicts consistent LMC
//                  is beneficial; narrative transportation theory predicts
//                  some variation (reflecting dramatic arc) may help.
//
// The key advancement over the track and segment models is that these
// features characterise the *dynamics* of congruence, not just its level.
// If this model's LOO-CV outperforms Models 1–3, it demonstrates that
// how congruence unfolds over a song matters above and beyond how much
// congruence there is on average.
//
// This model also includes artist random slopes on mean LMC (via NCP),
// allowing the level effect to vary by artist — some artists may benefit
// more from congruence than others.
//
// Structure
// ---------
//   logit(μ_i) = α[artist] + α[genre]
//                + β_lmc[artist] · mean_lmc_z
//                + β_lmc2 · mean_lmc_z²
//                + β_slope · lmc_slope_z
//                + β_curve · lmc_curve_z
//                + β_change · lmc_change_z
//                + β_sd · lmc_sd_z
//                + β_age · song_age_z
//
//   [α[artist], β_lmc[artist]] ~ MVN(orientation-moderated means, Σ)
//   α[genre] ~ N(0, σ_genre)
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_artist;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  vector[N] mean_lmc_z;
  vector[N] mean_lmc_z2;
  vector[N] lmc_slope_z;
  vector[N] lmc_curve_z;
  vector[N] lmc_change_z;
  vector[N] lmc_sd_z;
  vector[N] song_age_z;

  array[N] int<lower=1, upper=N_artist> artist_id;
  array[N] int<lower=1, upper=N_genre>  genre_id;

  vector<lower=0, upper=1>[N_artist] orientation;
}

parameters {
  real mu_global;

  // Population-level effects
  real beta_lmc;
  real beta_lmc2;
  real beta_slope;
  real beta_curve;
  real beta_change;
  real beta_sd;
  real beta_age;

  // Orientation moderators
  real gamma_intercept;
  real gamma_lmc;

  // Artist random effects: bivariate (intercept + LMC slope), NCP
  matrix[2, N_artist] z_artist;
  vector<lower=0>[2] sigma_artist;
  cholesky_factor_corr[2] L_artist;

  // Genre random intercepts
  vector[N_genre] alpha_genre;
  real<lower=0> sigma_genre;

  real<lower=0> phi;
}

transformed parameters {
  matrix[2, N_artist] artist_re;
  artist_re = diag_pre_multiply(sigma_artist, L_artist) * z_artist;

  vector[N_artist] alpha_artist;
  vector[N_artist] beta_lmc_artist;

  for (j in 1:N_artist) {
    alpha_artist[j]    = mu_global
                         + gamma_intercept * orientation[j]
                         + artist_re[1, j];
    beta_lmc_artist[j] = beta_lmc
                         + gamma_lmc * orientation[j]
                         + artist_re[2, j];
  }

  vector[N] eta;
  for (i in 1:N) {
    eta[i] = alpha_artist[artist_id[i]]
             + alpha_genre[genre_id[i]]
             + beta_lmc_artist[artist_id[i]] * mean_lmc_z[i]
             + beta_lmc2   * mean_lmc_z2[i]
             + beta_slope  * lmc_slope_z[i]
             + beta_curve  * lmc_curve_z[i]
             + beta_change * lmc_change_z[i]
             + beta_sd     * lmc_sd_z[i]
             + beta_age    * song_age_z[i];
  }

  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_lmc        ~ normal(0, 0.5);
  beta_lmc2       ~ normal(0, 0.25);
  beta_slope      ~ normal(0, 0.5);
  beta_curve      ~ normal(0, 0.5);
  beta_change     ~ normal(0, 0.5);
  beta_sd         ~ normal(0, 0.5);
  beta_age        ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);
  gamma_lmc       ~ normal(0, 0.5);

  to_vector(z_artist) ~ std_normal();
  sigma_artist        ~ normal(0, 0.5);
  L_artist            ~ lkj_corr_cholesky(2);

  alpha_genre ~ normal(0, sigma_genre);
  sigma_genre ~ normal(0, 0.5);

  phi ~ gamma(4, 0.1);

  y ~ beta(mu * phi, (1.0 - mu) * phi);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  matrix[2, 2] Rho_artist;
  vector[N] marginal_lmc;

  Rho_artist = multiply_lower_tri_self_transpose(L_artist);

  for (i in 1:N) {
    real a = fmax(mu[i] * phi, 1e-6);
    real b = fmax((1.0 - mu[i]) * phi, 1e-6);
    y_rep[i]   = beta_rng(a, b);
    log_lik[i] = beta_lpdf(y[i] | a, b);
    marginal_lmc[i] = beta_lmc_artist[artist_id[i]]
                      * mu[i] * (1.0 - mu[i]) * 100.0;
  }
}
