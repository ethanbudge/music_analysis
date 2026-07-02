// =============================================================================
// model_segment_curve_v4.stan — Section-aware functional regression: the
// "segment + curvature" model that UNIFIES the timeline and segment analyses.
//
// Instead of one coefficient function β(t), the congruence trajectory is split
// into its chorus and non-chorus parts and each gets its OWN coefficient
// function:
//
//     effect_s = ∫ β_c(t)  x^chorus_s(t)    dt   +   ∫ β_nc(t) x^nonchorus_s(t) dt
//              ≈ Σ_k b_c[k] · Zc[s,k]            +   Σ_k b_nc[k] · Znc[s,k]
//
// where Zc / Znc are the functional predictors computed over chorus-only and
// non-chorus-only lines respectively (precomputed in R from lmc_lines.csv +
// is_chorus). Comparing β_c(t) vs β_nc(t) answers BOTH "where in the song does
// congruence matter" and "does it matter more in the chorus", over position.
//
// This is the third model in the segment / curvature / segment+curvature LOO
// comparison. Both coefficient functions share the RW2 P-spline smoothing idea
// (each with its own smoothing scale). Same hierarchy / precision / controls.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  int<lower=1> Kb;
  matrix[N, Kb] Zc;              // functional predictors over CHORUS lines
  matrix[N, Kb] Znc;            // functional predictors over NON-CHORUS lines

  int<lower=1> G;
  matrix[G, Kb] Bgrid;

  int<lower=0> K;
  matrix[N, K] X;

  array[N] int<lower=1, upper=N_genre> genre_id;
  vector<lower=0, upper=1>[N] orientation;
  vector<lower=0, upper=1>[N] orientation_known;
}

parameters {
  real mu_global;
  // Non-centred RW2 P-splines for β_chorus(t) and β_nonchorus(t).
  real bc_level;  real bc_slope;  vector[Kb - 2] z_bc;
  real bnc_level; real bnc_slope; vector[Kb - 2] z_bnc;
  real<lower=0> sigma_bc;
  real<lower=0> sigma_bnc;
  vector[K] beta_ctrl;
  real gamma_intercept;

  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;

  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  vector[Kb] b_c;
  vector[Kb] b_nc;
  b_c[1]  = bc_level;   b_c[2]  = bc_level  + bc_slope;
  b_nc[1] = bnc_level;  b_nc[2] = bnc_level + bnc_slope;
  for (k in 3:Kb) {
    b_c[k]  = 2 * b_c[k - 1]  - b_c[k - 2]  + sigma_bc  * z_bc[k - 2];
    b_nc[k] = 2 * b_nc[k - 1] - b_nc[k - 2] + sigma_bnc * z_bnc[k - 2];
  }

  vector[N_genre] alpha_genre   = sigma_genre * z_genre;
  vector[N_genre] log_phi_genre = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] orient_k = orientation .* orientation_known;
  vector[N] ctrl = K > 0 ? X * beta_ctrl : rep_vector(0.0, N);

  vector[N] eta = mu_global
                  + gamma_intercept * orient_k
                  + alpha_genre[genre_id]
                  + Zc * b_c
                  + Znc * b_nc
                  + ctrl;
  vector<lower=0>[N] phi = exp(log_phi_genre[genre_id]);
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_ctrl       ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);

  // Non-centred RW2 P-spline priors on each coefficient function.
  bc_level ~ normal(0, 1);  bc_slope ~ normal(0, 1);  z_bc  ~ std_normal();
  bnc_level ~ normal(0, 1); bnc_slope ~ normal(0, 1); z_bnc ~ std_normal();
  sigma_bc  ~ normal(0, 1);
  sigma_bnc ~ normal(0, 1);

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);

  phi_intercept   ~ normal(1.8, 0.5);
  z_phi_genre     ~ std_normal();
  sigma_phi_genre ~ normal(0, 0.5);

  y ~ beta(mu .* phi, (1.0 - mu) .* phi);
}

generated quantities {
  vector[G] beta_chorus_t    = Bgrid * b_c;
  vector[G] beta_nonchorus_t = Bgrid * b_nc;
  vector[N] y_rep;
  vector[N] log_lik;
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real bb = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, bb);
    log_lik[i] = beta_lpdf(y[i] | a, bb);
  }
}
