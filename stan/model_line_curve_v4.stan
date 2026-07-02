// =============================================================================
// model_line_curve_v4.stan — EXPERIMENTAL one-stage line-level + curvature model.
//
// Scaffold for combining the line-by-line series with the trajectory in a single
// joint model (no pre-summarising, uncertainty propagated). Two linked likelihoods
// share a latent per-song congruence offset u[s]:
//
//   Line measurement model (M line observations):
//       lmc[m] ~ Normal( f(position_m) + u[song_m], sigma_line )
//     where f(t)=Σ f_k φ_k(t) is a population trajectory (RW2 P-spline) and
//     u[s] is song s's latent overall congruence (how far above/below the
//     population curve its lines sit).
//
//   Popularity model (N songs):
//       y[s] ~ Beta( mu_s φ_s , (1-mu_s) φ_s ),
//       logit(mu_s) = … + theta · u[s] + controls,
//     so popularity is linked to the SAME latent congruence the lines inform.
//
// Heaviest of the family to sample (a latent per song + a shared scale) — kept
// experimental. Only the popularity term is written to log_lik, so its LOO is
// directly comparable to the other v4 models. Tune sigma_line / tau_u priors and
// watch for u<->theta scale trade-offs before trusting it.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> N_genre;
  vector<lower=0, upper=1>[N] y;

  int<lower=0> K;
  matrix[N, K] X;
  array[N] int<lower=1, upper=N_genre> genre_id;
  vector<lower=0, upper=1>[N] orientation;
  vector<lower=0, upper=1>[N] orientation_known;

  int<lower=1> M;                       // total line observations
  array[M] int<lower=1, upper=N> song;  // song index for each line
  int<lower=1> Kb;
  matrix[M, Kb] LB;                     // line-position basis rows
  vector[M] lmc;                        // line-level LMC value (the measurement)

  int<lower=1> G;
  matrix[G, Kb] Bgrid;                  // basis on a grid (for f(t) output)
}

parameters {
  real mu_global;
  vector[K] beta_ctrl;
  real gamma_intercept;
  real theta;                           // latent congruence → popularity

  vector[Kb] f;                         // population trajectory f(t)
  real<lower=0> sigma_f;                // RW2 smoothing scale
  real<lower=0> sigma_line;            // line-level residual sd

  vector[N] z_u;                        // non-centred latent song congruence
  real<lower=0> tau_u;

  vector[N_genre] z_genre;
  real<lower=0> sigma_genre;
  real phi_intercept;
  vector[N_genre] z_phi_genre;
  real<lower=0> sigma_phi_genre;
}

transformed parameters {
  vector[N] u = tau_u * z_u;
  vector[N_genre] alpha_genre   = sigma_genre * z_genre;
  vector[N_genre] log_phi_genre = phi_intercept + sigma_phi_genre * z_phi_genre;

  vector[N] orient_k = orientation .* orientation_known;
  vector[N] ctrl = K > 0 ? X * beta_ctrl : rep_vector(0.0, N);

  vector[N] eta = mu_global
                  + gamma_intercept * orient_k
                  + alpha_genre[genre_id]
                  + theta * u
                  + ctrl;
  vector<lower=0>[N] phi = exp(log_phi_genre[genre_id]);
  vector<lower=0, upper=1>[N] mu = inv_logit(eta);
}

model {
  mu_global       ~ normal(0, 1.5);
  beta_ctrl       ~ normal(0, 0.5);
  gamma_intercept ~ normal(0, 0.5);
  theta           ~ normal(0, 1);

  // Population trajectory: RW2 P-spline.
  f[1] ~ normal(0, 1);
  if (Kb >= 2) f[2] ~ normal(0, 1);
  for (k in 3:Kb)
    target += normal_lpdf(f[k] - 2 * f[k - 1] + f[k - 2] | 0, sigma_f);
  sigma_f    ~ normal(0, 1);
  sigma_line ~ normal(0, 1);

  z_u   ~ std_normal();
  tau_u ~ normal(0, 1);

  z_genre     ~ std_normal();
  sigma_genre ~ normal(0, 0.5);
  phi_intercept   ~ normal(1.8, 0.5);
  z_phi_genre     ~ std_normal();
  sigma_phi_genre ~ normal(0, 0.5);

  // Line measurement model (shares u with popularity).
  lmc ~ normal(LB * f + u[song], sigma_line);

  // Popularity model.
  y ~ beta(mu .* phi, (1.0 - mu) .* phi);
}

generated quantities {
  vector[G] f_t = Bgrid * f;            // population congruence trajectory
  vector[N] y_rep;
  vector[N] log_lik;                    // popularity term only (LOO-comparable)
  for (i in 1:N) {
    real a = fmax(mu[i] * phi[i], 1e-6);
    real bb = fmax((1.0 - mu[i]) * phi[i], 1e-6);
    y_rep[i]   = beta_rng(a, bb);
    log_lik[i] = beta_lpdf(y[i] | a, bb);
  }
}
