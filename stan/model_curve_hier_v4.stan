// =============================================================================
// model_curve_hier_v4.stan — GENRE-VARYING penalized-spline curvature model.
//
// The flexible counterpart to model_curve_poly_v4: the congruence coefficient
// function β_g(t) is a B-spline that varies by genre. A smooth NON-CENTRED RW2
// population curve β(t) = Σ b_k φ_k(t) plus per-genre coefficient deviations that
// are shrunk toward zero (hierarchical partial pooling), so genres with few songs
// stay near the population shape while data-rich genres can depart from it.
//
// Same data contract as model_curve_v4 (Z, Bgrid, controls, genre) — no new R
// inputs. generated quantities returns β(t) and β_g(t) per genre for plotting.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;
  vector<lower=0, upper=1>[N] y;

  int<lower=3> Kb;
  matrix[N, Kb] Z;
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
  // Non-centred RW2 population spline.
  real b_level;
  real b_slope;
  vector[Kb - 2] z_b;
  real<lower=0> sigma_b;
  // Non-centred genre deviations of the spline weights (shrunk toward 0).
  matrix[Kb, N_genre] z_delta;
  real<lower=0> tau_delta;

  vector[K] beta_ctrl;
  real gamma_intercept;

  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;

  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  vector[Kb] b;
  b[1] = b_level;
  b[2] = b_level + b_slope;
  for (k in 3:Kb) b[k] = 2 * b[k - 1] - b[k - 2] + sigma_b * z_b[k - 2];

  // Per-genre spline weights = population + shrunk deviation.
  matrix[Kb, N_genre] b_genre;
  for (g in 1:N_genre) b_genre[, g] = b + tau_delta * z_delta[, g];

  vector[N_genre] alpha_genre   = sigma_genre * z_genre;
  vector[N_genre] log_phi_genre = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] orient_k = orientation .* orientation_known;
  vector[N] ctrl = K > 0 ? X * beta_ctrl : rep_vector(0.0, N);

  vector[N] eta;
  for (i in 1:N)
    eta[i] = mu_global
             + gamma_intercept * orient_k[i]
             + alpha_genre[genre_id[i]]
             + Z[i] * b_genre[, genre_id[i]]
             + ctrl[i];
  vector<lower=0>[N] phi = exp(log_phi_genre[genre_id]);
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  b_level ~ normal(0, 1);
  b_slope ~ normal(0, 1);
  z_b     ~ std_normal();
  sigma_b ~ normal(0, 1);
  to_vector(z_delta) ~ std_normal();
  tau_delta ~ normal(0, 0.3);           // genre-deviation shrinkage
  beta_ctrl       ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);

  phi_intercept   ~ normal(1.8, 0.5);
  z_phi_genre     ~ std_normal();
  sigma_phi_genre ~ normal(0, 0.5);

  y ~ beta(mu .* phi, (1.0 - mu) .* phi);
}

generated quantities {
  vector[G] beta_t = Bgrid * b;
  matrix[G, N_genre] beta_t_genre;
  for (g in 1:N_genre) beta_t_genre[, g] = Bgrid * b_genre[, g];

  vector[N] y_rep;
  vector[N] log_lik;
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real bb = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, bb);
    log_lik[i] = beta_lpdf(y[i] | a, bb);
  }
}
