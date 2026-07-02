// =============================================================================
// model_curve_poly_v4.stan — GENRE-VARYING orthogonal-polynomial curvature model.
//
// The parsimonious, well-conditioned alternative to the penalized spline: the
// congruence coefficient function is a low-order polynomial in song position,
//
//     β_g(t) = θ_{g,0} + θ_{g,1}·t + θ_{g,2}·t² (+ …)   (degree D-1)
//
// and the polynomial coefficients VARY BY GENRE, partially pooled toward a
// population polynomial (hierarchical / "the quadratic effect differs by genre").
// The functional term for song s is the inner product of its congruence curve
// with β_{genre(s)}(t), precomputed in R as P[s,·] (orthogonal-polynomial moments
// of the curve). Orthogonal columns + only D·N_genre coefficients ⇒ clean geometry
// (no smoothing-scale funnel), and β_g(t) is directly interpretable per genre.
//
// generated quantities returns β(t) (population) and β_g(t) for every genre, on a
// plotting grid, so the report can draw one trajectory per genre.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;
  vector<lower=0, upper=1>[N] y;

  int<lower=2> D;                // number of polynomial terms (degree + 1)
  matrix[N, D] P;               // orthogonal-polynomial moments of each song's curve
  int<lower=1> G;
  matrix[G, D] Bgrid;           // polynomial basis on the plotting grid

  int<lower=0> K;
  matrix[N, K] X;
  array[N] int<lower=1, upper=N_genre> genre_id;
  vector<lower=0, upper=1>[N] orientation;
  vector<lower=0, upper=1>[N] orientation_known;
}

parameters {
  real mu_global;
  vector[D] theta;                 // population polynomial coefficients
  matrix[D, N_genre] z_theta;      // non-centred genre deviations
  vector<lower=0>[D] tau_theta;    // per-order genre-variation scale

  vector[K] beta_ctrl;
  real gamma_intercept;

  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;

  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  // Genre-specific polynomial coefficients (columns) = population + scaled dev.
  matrix[D, N_genre] theta_genre;
  for (g in 1:N_genre)
    theta_genre[, g] = theta + tau_theta .* z_theta[, g];

  vector[N_genre] alpha_genre   = sigma_genre * z_genre;
  vector[N_genre] log_phi_genre = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] orient_k = orientation .* orientation_known;
  vector[N] ctrl = K > 0 ? X * beta_ctrl : rep_vector(0.0, N);

  vector[N] eta;
  for (i in 1:N)
    eta[i] = mu_global
             + gamma_intercept * orient_k[i]
             + alpha_genre[genre_id[i]]
             + P[i] * theta_genre[, genre_id[i]]
             + ctrl[i];
  vector<lower=0>[N] phi = exp(log_phi_genre[genre_id]);
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  theta           ~ normal(0, 0.5);
  to_vector(z_theta) ~ std_normal();
  tau_theta       ~ normal(0, 0.5);     // shrinks genre curves toward the population
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
  vector[G] beta_t = Bgrid * theta;                 // population coefficient function
  matrix[G, N_genre] beta_t_genre;                  // one β_g(t) per genre
  for (g in 1:N_genre) beta_t_genre[, g] = Bgrid * theta_genre[, g];

  vector[N] y_rep;
  vector[N] log_lik;
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real bb = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, bb);
    log_lik[i] = beta_lpdf(y[i] | a, bb);
  }
}
