// =============================================================================
// model_curve_v4.stan — Scalar-on-function regression of popularity on the
// line-by-line LMC *trajectory* (the "curvature" model).
//
// Replaces the old two-stage summary (mean/slope/curve/sd of LMC) with a proper
// functional regression. Each song s contributes a congruence function x_s(t)
// over normalised song position t ∈ [0,1] (its per-line LMC series). The effect
// on popularity is the functional inner product
//
//     congruence_effect_s = ∫ β(t) x_s(t) dt  ≈  Σ_k b_k · Z[s,k]
//
// where β(t) = Σ_k b_k φ_k(t) is a B-spline expansion and Z[s,k] = ∫ x_s(t) φ_k(t) dt
// is precomputed in R. β(t) is smoothed by a 2nd-order random-walk (P-spline)
// prior, so the data choose how wiggly the "where in the song congruence matters"
// curve is. Same genre hierarchy / precision submodel / generic controls as v4.
//
// Bgrid[G,Kb] (basis on a plotting grid) is passed in so beta_t = Bgrid·b is a
// generated quantity — the report plots β(t) with a credible band directly.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;

  vector<lower=0, upper=1>[N] y;

  int<lower=1> Kb;                // functional basis dimension
  matrix[N, Kb] Z;               // functional predictors  ∫ x_s(t) φ_k(t) dt

  int<lower=1> G;                 // plotting-grid resolution
  matrix[G, Kb] Bgrid;           // basis evaluated on the grid (for β(t) output)

  int<lower=0> K;
  matrix[N, K] X;

  array[N] int<lower=1, upper=N_genre> genre_id;
  vector<lower=0, upper=1>[N] orientation;
  vector<lower=0, upper=1>[N] orientation_known;
}

parameters {
  real mu_global;
  // Non-centred RW2 P-spline: level + slope + standardised 2nd-difference
  // innovations. This breaks the sigma_b–vs–b funnel that caused divergences /
  // low E-BFMI in the centred version.
  real b_level;                  // β(t) overall level  (b[1])
  real b_slope;                  // β(t) initial slope  (b[2]-b[1])
  vector[Kb - 2] z_b;            // standardised RW2 innovations
  real<lower=0> sigma_b;         // P-spline (RW2) smoothing scale
  vector[K] beta_ctrl;
  real gamma_intercept;

  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;

  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  // Reconstruct the spline weights from the non-centred RW2 parameterization.
  vector[Kb] b;
  b[1] = b_level;
  b[2] = b_level + b_slope;
  for (k in 3:Kb) b[k] = 2 * b[k - 1] - b[k - 2] + sigma_b * z_b[k - 2];

  vector[N_genre] alpha_genre   = sigma_genre * z_genre;
  vector[N_genre] log_phi_genre = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] orient_k = orientation .* orientation_known;
  vector[N] ctrl = K > 0 ? X * beta_ctrl : rep_vector(0.0, N);

  vector[N] eta = mu_global
                  + gamma_intercept * orient_k
                  + alpha_genre[genre_id]
                  + Z * b
                  + ctrl;
  vector<lower=0>[N] phi = exp(log_phi_genre[genre_id]);
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_ctrl       ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);

  // Non-centred RW2 P-spline priors.
  b_level ~ normal(0, 1);
  b_slope ~ normal(0, 1);
  z_b     ~ std_normal();
  sigma_b ~ normal(0, 1);

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);

  phi_intercept   ~ normal(1.8, 0.5);
  z_phi_genre     ~ std_normal();
  sigma_phi_genre ~ normal(0, 0.5);

  y ~ beta(mu .* phi, (1.0 - mu) .* phi);
}

generated quantities {
  vector[G] beta_t = Bgrid * b;   // the estimated coefficient function β(t)
  vector[N] y_rep;
  vector[N] log_lik;
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real bb = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, bb);
    log_lik[i] = beta_lpdf(y[i] | a, bb);
  }
}
