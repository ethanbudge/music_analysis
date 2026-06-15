# =============================================================================
# bayesian_lmc_model.R
# Bayesian Hierarchical Beta Regression: LMC → Streaming Popularity
#
# Requires:
#   cmdstanr        (Stan interface)
#   posterior        (posterior summaries)
#   bayesplot        (MCMC diagnostics + traceplots)
#   loo              (LOO-CV model comparison)
#   tidyverse
#   splines2         (B-spline basis construction)
#   patchwork
#
# Install if needed:
#   install.packages(c("posterior","bayesplot","loo","splines2","patchwork"))
#   install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/",
#                                           getOption("repos")))
#   cmdstanr::install_cmdstan()
# =============================================================================

library(cmdstanr)
library(posterior)
library(bayesplot)
library(loo)
library(tidyverse)
library(splines2)
library(patchwork)
library(here)

# Match theme from main analysis.R
theme_mc <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 2),
      plot.subtitle    = element_text(color = "grey40", size = base_size - 1),
      axis.title       = element_text(face = "bold"),
      legend.position  = "bottom",
      panel.grid.minor = element_blank(),
      strip.text       = element_text(face = "bold"),
    )
}
theme_set(theme_mc())

ORIENTATION_COLORS <- c("narrative" = "#2A9D8F", "production" = "#E76F51")

out_dir <- here("analysis", "output")
tab_dir <- file.path(out_dir, "tables")
fig_dir <- file.path(out_dir, "figures")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

save_fig <- function(name, w = 9, h = 6) {
  ggsave(file.path(fig_dir, name), width = w, height = h,
         device = "pdf", dpi = 300)
  message("  Saved: ", name)
}


# =============================================================================
# 1.  DATA PREPARATION
# =============================================================================
message("Loading and preparing data...")

# ── Load timeline (line-level) data ──────────────────────────────────────────
tl_path <- here("results", "lyric_timeline", "lyric_timeline.csv")
if (!file.exists(tl_path)) stop("Run 10_lyric_timeline.py first.")

tl <- read_csv(tl_path, show_col_types = FALSE) %>%
  filter(match_confidence >= 0.40)

# ── Load master results (song-level) ──────────────────────────────────────────
master <- read_csv(here("results", "master_results.csv"),
                   show_col_types = FALSE)

# ── Extract song-level trajectory features ────────────────────────────────────
# (same function as in the timeline analysis section)
extract_traj <- function(d) {
  if (nrow(d) < 5) return(NULL)
  tryCatch({
    lo  <- loess(lmc ~ position_pct, data = d, span = 0.4, degree = 2)
    g   <- seq(0, 100, by = 1)
    sm  <- predict(lo, newdata = data.frame(position_pct = g))
    sm  <- ifelse(is.na(sm), mean(d$lmc), sm)
    tibble(
      mean_lmc      = mean(d$lmc),
      lmc_smooth_sd = sd(sm),
      n_lines       = nrow(d),
      # Store full smooth for basis projection below
      smooth_grid   = list(tibble(pos = g, lmc = sm))
    )
  }, error = function(e) NULL)
}

traj_song <- tl %>%
  group_by(song_id) %>%
  group_modify(~extract_traj(.x)) %>%
  ungroup()

# ── Build B-spline basis for functional trajectory term ───────────────────────
# Each song's LMC curve is projected onto K basis functions.
# The integral ∫ B_k(t) · LMC(t) dt approximates the inner product between
# the trajectory and each basis function.

K       <- 10                          # number of basis functions
knots   <- seq(0, 100, length.out = K - 2)   # interior knots
grid    <- 0:100                       # evaluation grid (101 points)

# Evaluate B-spline basis on the grid
basis_mat <- bSpline(grid, knots = knots[-c(1, length(knots))],
                     degree = 3, intercept = TRUE)
# basis_mat: 101 × K matrix

# For each song: project its smooth onto the basis via numerical integration
project_onto_basis <- function(smooth_grid_df) {
  # Interpolate smooth to common grid
  lmc_at_grid <- approx(smooth_grid_df$pos, smooth_grid_df$lmc,
                         xout = grid, rule = 2)$y
  # Trapezoidal integration: ∫ B_k(t) · LMC(t) dt ≈ Σ B_k[t] · LMC[t] · Δt
  # Δt = 1 (grid points are 1% apart), normalise by 100
  colSums(basis_mat * lmc_at_grid) / 100
}

B_matrix <- traj_song %>%
  filter(!is.null(smooth_grid)) %>%
  mutate(basis_proj = map(smooth_grid, project_onto_basis)) %>%
  pull(basis_proj) %>%
  do.call(rbind, .)   # N_songs × K matrix

# ── Assemble song-level modelling dataset ─────────────────────────────────────
song_df <- traj_song %>%
  filter(!is.null(smooth_grid)) %>%
  left_join(master %>% select(
    song_id, artist_code, genre, orientation,
    popularity, energy, danceability, valence, release_date
  ), by = "song_id") %>%
  filter(!is.na(popularity)) %>%
  mutate(
    release_year = as.numeric(substr(release_date, 1, 4)),
    song_age     = 2025 - release_year,
    orientation_num = as.integer(orientation == "narrative"),
    genre_cluster = case_when(
      genre %in% c("hip-hop")                           ~ "Hip-Hop",
      genre %in% c("folk", "folk-rock", "country")      ~ "Folk/Country",
      genre %in% c("pop")                               ~ "Pop",
      genre %in% c("electronic","psychedelic-electronic") ~ "Electronic",
      TRUE                                              ~ "Other"
    )
  ) %>%
  drop_na(energy, danceability, valence, song_age)

n_songs <- nrow(song_df)
message(sprintf("Analysis sample: %d songs", n_songs))

# ── Indexing: artist and genre ────────────────────────────────────────────────
artist_levels <- unique(song_df$artist_code)
genre_levels  <- unique(song_df$genre_cluster)

song_df <- song_df %>%
  mutate(
    artist_idx = as.integer(factor(artist_code, levels = artist_levels)),
    genre_idx  = as.integer(factor(genre_cluster, levels = genre_levels))
  )

# Artist → genre mapping
artist_genre_map <- song_df %>%
  distinct(artist_code, genre_cluster, artist_idx, genre_idx) %>%
  arrange(artist_idx) %>%
  pull(genre_idx)

# Artist orientation (aggregated from songs)
artist_orientation <- song_df %>%
  group_by(artist_idx) %>%
  summarise(orientation = mean(orientation_num), .groups = "drop") %>%
  arrange(artist_idx) %>%
  pull(orientation)

# ── Z-score all continuous predictors ────────────────────────────────────────
z <- function(x) as.numeric(scale(x))

# ── Scale outcome to (0,1) open interval ─────────────────────────────────────
# Smithson & Verkuilen (2006) transformation to avoid 0/1 boundary issues
N_s  <- n_songs
y_raw <- song_df$popularity / 100
y_scaled <- (y_raw * (N_s - 1) + 0.5) / N_s

# ── Align B matrix to song_df row order ──────────────────────────────────────
# (traj_song and song_df may differ in row order after joins)
B_aligned <- B_matrix[match(song_df$song_id,
                             traj_song$song_id[!sapply(traj_song$smooth_grid,
                                                        is.null)]), ]


# ── Assemble Stan data list ───────────────────────────────────────────────────
stan_data <- list(
  N              = n_songs,
  J              = length(artist_levels),
  G              = length(genre_levels),
  K              = K,
  y              = y_scaled,
  lmc_mean_z     = z(song_df$mean_lmc),
  lmc_consistency_z = z(song_df$lmc_smooth_sd),
  energy_z       = z(song_df$energy),
  valence_z      = z(song_df$valence),
  danceability_z = z(song_df$danceability),
  song_age_z     = z(song_df$song_age),
  orientation    = song_df$orientation_num,
  B              = B_aligned,
  artist_id      = song_df$artist_idx,
  genre_id       = song_df$genre_idx,
  artist_genre_id = artist_genre_map,
  artist_orientation = artist_orientation
)

message(sprintf("Stan data: N=%d, J=%d artists, G=%d genres, K=%d basis fns",
                stan_data$N, stan_data$J, stan_data$G, stan_data$K))


# =============================================================================
# 2.  COMPILE AND FIT STAN MODEL
# =============================================================================
message("\nCompiling Stan model...")

stan_file <- here("analysis", "lmc_popularity.stan")
model     <- cmdstan_model(stan_file, compile = TRUE)

message("Fitting model (4 chains × 2000 iterations)...")
message("  Expected runtime: 15–45 min depending on N and hardware.\n")

fit <- model$sample(
  data            = stan_data,
  seed            = 42,
  chains          = 4,
  parallel_chains = 4,       # uses all 4 cores; adjust if needed
  iter_warmup     = 2000,
  iter_sampling   = 2000,
  adapt_delta     = 0.95,    # higher than default; hierarchical models need this
  max_treedepth   = 12,
  refresh         = 200,
  output_dir      = here("analysis", "stan_output"),
)

message("\nFit complete.")


# =============================================================================
# 3.  CONVERGENCE DIAGNOSTICS
# =============================================================================
message("\n── Convergence diagnostics ──")

# ── R-hat and ESS ─────────────────────────────────────────────────────────────
diag_df <- fit$summary() %>%
  select(variable, mean, sd, q5, q95, rhat, ess_bulk, ess_tail) %>%
  filter(!str_detect(variable, "^log_lik|^y_rep|^mu_hat"))

# Flag problematic parameters
bad_rhat <- diag_df %>% filter(!is.na(rhat), rhat > 1.01)
low_ess  <- diag_df %>% filter(!is.na(ess_bulk), ess_bulk < 400)

if (nrow(bad_rhat) > 0) {
  message("  ⚠ Parameters with R-hat > 1.01:")
  print(bad_rhat %>% select(variable, rhat, ess_bulk))
} else {
  message("  ✓ All R-hat < 1.01")
}

if (nrow(low_ess) > 0) {
  message("  ⚠ Parameters with ESS_bulk < 400:")
  print(low_ess %>% select(variable, ess_bulk))
} else {
  message("  ✓ All ESS_bulk ≥ 400")
}

# ── Divergences ───────────────────────────────────────────────────────────────
n_div <- sum(fit$diagnostic_summary()$num_divergent)
message(sprintf("  Divergent transitions: %d", n_div))
if (n_div > 0) message("  → Consider increasing adapt_delta to 0.99")

# ── Save diagnostic table ─────────────────────────────────────────────────────
params_of_interest <- c(
  "alpha_global", "beta_lmc", "beta_lmc_orient", "beta_consistency",
  "beta_energy", "beta_valence", "beta_danceability", "beta_age",
  "beta_orientation", "gamma_orient", "phi", "tau",
  "sigma_artist[1]", "sigma_artist[2]", "sigma_genre", "rho_artist", "icc_artist"
)

diag_key <- diag_df %>%
  filter(variable %in% params_of_interest) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

write_csv(diag_key, file.path(tab_dir, "19_bayes_diagnostics.csv"))
print(diag_key)


# =============================================================================
# 4.  TRACEPLOTS
# =============================================================================
message("\n── Generating traceplots ──")

draws_array <- fit$draws(format = "array")

# ── Primary parameters ────────────────────────────────────────────────────────
p_trace_main <- mcmc_trace(
  draws_array,
  pars  = c("alpha_global", "beta_lmc", "beta_lmc_orient",
            "beta_consistency", "beta_orientation", "phi"),
  facet_args = list(ncol = 2, strip.position = "left")
) +
  scale_color_brewer(palette = "Set2", name = "Chain") +
  labs(
    title    = "Traceplots: Primary Fixed Effects",
    subtitle = "Well-mixed chains indicate convergence"
  ) +
  theme_mc()

print(p_trace_main)
save_fig("20_traceplots_main.pdf", w = 11, h = 9)

# ── Random effect SDs ─────────────────────────────────────────────────────────
p_trace_re <- mcmc_trace(
  draws_array,
  pars = c("sigma_artist[1]", "sigma_artist[2]",
           "sigma_genre", "gamma_orient", "tau", "rho_artist"),
  facet_args = list(ncol = 2, strip.position = "left")
) +
  scale_color_brewer(palette = "Set2", name = "Chain") +
  labs(title = "Traceplots: Variance Components and Hyperparameters") +
  theme_mc()

print(p_trace_re)
save_fig("21_traceplots_re.pdf", w = 11, h = 9)

# ── Functional trajectory coefficients ────────────────────────────────────────
theta_pars <- paste0("theta[", 1:K, "]")

p_trace_theta <- mcmc_trace(
  draws_array,
  pars = theta_pars,
  facet_args = list(ncol = 2, strip.position = "left")
) +
  scale_color_brewer(palette = "Set2", name = "Chain") +
  labs(title    = "Traceplots: B-Spline Trajectory Coefficients (θ)",
       subtitle = "RW1 smoothness prior — adjacent coefficients correlated") +
  theme_mc()

print(p_trace_theta)
save_fig("22_traceplots_theta.pdf", w = 11, h = 12)

# ── Rank plots (better than traceplots for detecting mixing problems) ──────────
p_rank <- mcmc_rank_overlay(
  draws_array,
  pars = c("beta_lmc", "beta_lmc_orient", "beta_consistency",
           "sigma_artist[1]", "sigma_genre")
) +
  labs(title = "Chain Rank Plots",
       subtitle = "Uniform distribution across ranks = good mixing") +
  theme_mc()

print(p_rank)
save_fig("23_rank_plots.pdf", w = 11, h = 7)


# =============================================================================
# 5.  POSTERIOR SUMMARIES
# =============================================================================
message("\n── Posterior summaries ──")

draws_df <- fit$draws(format = "df")

# ── Credible intervals for key parameters ─────────────────────────────────────
posterior_summary <- fit$summary(
  variables = params_of_interest,
  mean, sd,
  ~quantile(.x, probs = c(0.025, 0.10, 0.90, 0.975)),
  ~mean(.x > 0)   # P(effect > 0)
) %>%
  rename(
    q2.5  = `2.5%`,
    q10   = `10%`,
    q90   = `90%`,
    q97.5 = `97.5%`,
    p_pos = `mean(.x > 0)`
  ) %>%
  mutate(
    p_direction = pmax(p_pos, 1 - p_pos),  # P(correct sign)
    rope_excl   = q2.5 > -0.05 | q97.5 < 0.05,  # excludes ROPE [-0.05, 0.05]
    across(where(is.numeric), ~round(., 4))
  )

print(posterior_summary)
write_csv(posterior_summary,
          file.path(tab_dir, "20_posterior_summary.csv"))

# ── Forest plot of fixed effects ───────────────────────────────────────────────
fixed_params <- c(
  "beta_lmc"          = "LMC Mean",
  "beta_lmc_orient"   = "LMC × Narrative",
  "beta_consistency"  = "LMC Consistency",
  "beta_energy"       = "Energy",
  "beta_valence"      = "Valence",
  "beta_danceability" = "Danceability",
  "beta_age"          = "Song Age",
  "beta_orientation"  = "Narrative (main)"
)

forest_df <- posterior_summary %>%
  filter(variable %in% names(fixed_params)) %>%
  mutate(
    label     = recode(variable, !!!fixed_params),
    lmc_param = str_detect(variable, "lmc|consistency"),
    sig_90    = (q10 > 0 | q90 < 0)   # excludes zero at 80% CI
  )

p_forest <- forest_df %>%
  ggplot(aes(x = mean, y = reorder(label, mean), color = lmc_param)) +
  # ROPE shading
  annotate("rect", xmin = -0.05, xmax = 0.05,
           ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  # 95% CI
  geom_errorbarh(aes(xmin = q2.5, xmax = q97.5),
                 height = 0.25, linewidth = 0.8) +
  # 80% CI (thicker)
  geom_errorbarh(aes(xmin = q10, xmax = q90),
                 height = 0, linewidth = 2.5, alpha = 0.5) +
  geom_point(aes(shape = sig_90), size = 3.5) +
  scale_color_manual(
    values = c("TRUE" = "#2A9D8F", "FALSE" = "#B0B0B0"),
    labels = c("TRUE" = "LMC parameter", "FALSE" = "Control"),
    name   = NULL
  ) +
  scale_shape_manual(
    values = c("TRUE" = 16, "FALSE" = 1),
    labels = c("TRUE" = "Excludes zero (80% CI)", "FALSE" = "Includes zero"),
    name   = NULL
  ) +
  labs(
    title    = "Posterior Distributions: Fixed Effects",
    subtitle = "Grey band = ROPE [−0.05, 0.05]. Thick bars = 80% CI, thin = 95% CI.",
    x        = "Posterior Mean (log-odds scale)",
    y        = NULL
  )

print(p_forest)
save_fig("24_posterior_forest.pdf", w = 10, h = 6)


# =============================================================================
# 6.  FUNCTIONAL TRAJECTORY EFFECT: β(t)
# =============================================================================
message("\n── Functional trajectory effect ──")

# Extract posterior draws for theta (K-vector)
theta_draws <- draws_df %>%
  select(starts_with("theta[")) %>%
  as.matrix()   # (n_draws × K)

# Reconstruct β(t) = basis_mat %*% theta for each posterior draw
# basis_mat: 101 × K evaluated on 0:100 grid
beta_t_draws <- theta_draws %*% t(basis_mat)   # n_draws × 101

# Posterior summary of β(t) at each grid point
beta_t_summary <- tibble(
  position_pct = 0:100,
  mean         = colMeans(beta_t_draws),
  q5           = apply(beta_t_draws, 2, quantile, 0.05),
  q25          = apply(beta_t_draws, 2, quantile, 0.25),
  q75          = apply(beta_t_draws, 2, quantile, 0.75),
  q95          = apply(beta_t_draws, 2, quantile, 0.95),
  p_pos        = colMeans(beta_t_draws > 0)
)

# Where is β(t) reliably positive / negative?
beta_t_summary <- beta_t_summary %>%
  mutate(
    sig_pos = q5  > 0,   # 90% CI excludes 0 from below
    sig_neg = q95 < 0,   # 90% CI excludes 0 from above
    sig     = sig_pos | sig_neg
  )

p_beta_t <- beta_t_summary %>%
  ggplot(aes(x = position_pct)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  # 90% CI band
  geom_ribbon(aes(ymin = q5, ymax = q95), fill = "#2A9D8F", alpha = 0.15) +
  # 50% CI band
  geom_ribbon(aes(ymin = q25, ymax = q75), fill = "#2A9D8F", alpha = 0.30) +
  # Posterior mean
  geom_line(aes(y = mean), color = "#2A9D8F", linewidth = 1.2) +
  # Highlight significant regions
  geom_rug(
    data = beta_t_summary %>% filter(sig),
    aes(x = position_pct, color = sig_pos),
    sides = "b", linewidth = 1.2, alpha = 0.7
  ) +
  scale_color_manual(
    values = c("TRUE" = "#2A9D8F", "FALSE" = "#E63946"),
    labels = c("TRUE" = "Positive effect", "FALSE" = "Negative effect"),
    name   = "Effect direction"
  ) +
  scale_x_continuous(labels = function(x) paste0(x, "%"),
                     breaks = seq(0, 100, 25)) +
  labs(
    title    = "Functional Effect of LMC Trajectory on Popularity: β(t)",
    subtitle = paste0(
      "Positive β(t) = higher LMC at position t predicts higher popularity.\n",
      "Bands: 50% and 90% credible intervals. Rug = positions with 90% CI ≠ 0."
    ),
    x = "Position in Song (%)",
    y = "β(t): Marginal effect of LMC at position t"
  )

print(p_beta_t)
save_fig("25_functional_beta_t.pdf", w = 11, h = 6)

# Save β(t) summary
write_csv(beta_t_summary,
          file.path(tab_dir, "21_beta_t_functional.csv"))


# =============================================================================
# 7.  POSTERIOR PREDICTIVE CHECK
# =============================================================================
message("\n── Posterior predictive checks ──")

# Extract y_rep draws (n_draws × N matrix)
y_rep_draws <- fit$draws("y_rep", format = "matrix")

# Scale back to 0–100 for interpretability
y_obs_pop   <- song_df$popularity
y_rep_pop   <- y_rep_draws * 100

# ── Distribution overlap ──────────────────────────────────────────────────────
p_ppc_dens <- ppc_dens_overlay(
  y    = y_obs_pop,
  yrep = y_rep_pop[sample(nrow(y_rep_pop), 100), ]
) +
  labs(
    title    = "Posterior Predictive Check: Popularity Distribution",
    subtitle = "Dark line = observed. Light lines = 100 posterior predictive draws.",
    x        = "Popularity (0–100)",
    y        = "Density"
  ) +
  theme_mc()

print(p_ppc_dens)
save_fig("26_ppc_density.pdf", w = 9, h = 6)

# ── PPC statistics ─────────────────────────────────────────────────────────────
p_ppc_stat <- ppc_stat_2d(
  y    = y_obs_pop,
  yrep = y_rep_pop,
  stat = c("mean", "sd")
) +
  labs(
    title    = "PPC: Mean vs. SD of Popularity",
    subtitle = "Point = observed statistics. Ellipse = posterior predictive distribution."
  ) +
  theme_mc()

print(p_ppc_stat)
save_fig("27_ppc_stat2d.pdf", w = 7, h = 6)

# ── Residuals ─────────────────────────────────────────────────────────────────
mu_hat_draws <- fit$draws("mu_hat", format = "matrix") * 100  # scale to 0-100
mu_hat_mean  <- colMeans(mu_hat_draws)

resid_df <- song_df %>%
  mutate(
    predicted = mu_hat_mean,
    residual  = popularity - predicted
  )

p_resid <- resid_df %>%
  ggplot(aes(x = predicted, y = residual, color = genre_cluster)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(alpha = 0.65, size = 2.5) +
  geom_smooth(method = "loess", se = FALSE, color = "black",
              linewidth = 0.8, linetype = "dotted") +
  scale_color_brewer(palette = "Set2", name = "Genre") +
  labs(
    title    = "Posterior Mean Residuals vs. Predicted Popularity",
    subtitle = "Dotted LOESS = systematic pattern (should be flat if well-fitted)",
    x        = "Predicted Popularity",
    y        = "Residual (Actual − Predicted)"
  )

print(p_resid)
save_fig("28_residuals.pdf", w = 9, h = 6)


# =============================================================================
# 8.  LOO-CV MODEL COMPARISON
# =============================================================================
message("\n── LOO-CV model comparison ──")

log_lik_draws <- fit$draws("log_lik", format = "matrix")
loo_result    <- loo(log_lik_draws, cores = 4)

print(loo_result)
message(sprintf(
  "  LOO-IC: %.1f  (SE: %.1f)",
  loo_result$estimates["looic", "Estimate"],
  loo_result$estimates["looic", "SE"]
))

# Pareto k diagnostics — flag influential observations
pareto_k  <- loo_result$diagnostics$pareto_k
k_df <- song_df %>%
  mutate(pareto_k = pareto_k) %>%
  filter(pareto_k > 0.7) %>%
  select(song_id, artist_code, genre_cluster, popularity, mean_lmc, pareto_k) %>%
  arrange(desc(pareto_k))

if (nrow(k_df) > 0) {
  message("  ⚠ High Pareto-k songs (influential observations):")
  print(k_df)
} else {
  message("  ✓ No Pareto-k > 0.7 — LOO estimates are reliable")
}

# Pareto-k plot
p_pareto <- tibble(
  song_id  = song_df$song_id,
  genre    = song_df$genre_cluster,
  pareto_k = pareto_k
) %>%
  arrange(pareto_k) %>%
  mutate(idx = row_number()) %>%
  ggplot(aes(x = idx, y = pareto_k, color = genre)) +
  geom_hline(yintercept = c(0.5, 0.7), linetype = "dashed",
             color = c("orange", "red"), linewidth = 0.7) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_brewer(palette = "Set2", name = "Genre") +
  annotate("text", x = 5, y = 0.52, label = "k = 0.5 (caution)",
           color = "orange", hjust = 0, size = 3) +
  annotate("text", x = 5, y = 0.72, label = "k = 0.7 (problematic)",
           color = "red", hjust = 0, size = 3) +
  labs(
    title    = "LOO-CV Pareto-k Diagnostic",
    subtitle = "Values < 0.5 = reliable; 0.5–0.7 = mild concern; > 0.7 = problematic",
    x        = "Song (sorted)",
    y        = "Pareto-k"
  )

print(p_pareto)
save_fig("29_pareto_k.pdf", w = 9, h = 5)


# =============================================================================
# 9.  VARIANCE DECOMPOSITION (ICC)
# =============================================================================
message("\n── Variance decomposition ──")

icc_draws <- draws_df %>% pull(icc_artist)

icc_summary <- tibble(
  level   = c("Artist", "Genre", "Residual"),
  icc_est = c(
    mean(icc_draws),
    mean(draws_df$sigma_genre^2 /
           (draws_df$`sigma_artist[1]`^2 + draws_df$sigma_genre^2 + pi^2/3)),
    NA  # residual by subtraction
  )
) %>%
  mutate(icc_est = if_else(is.na(icc_est),
                            1 - sum(icc_est, na.rm = TRUE),
                            icc_est))

message("ICC (proportion of variance at each level):")
print(icc_summary %>% mutate(across(where(is.numeric), ~round(., 3))))

p_icc_draws <- ggplot(data.frame(icc = icc_draws), aes(x = icc)) +
  geom_density(fill = "#2A9D8F", alpha = 0.6) +
  geom_vline(xintercept = mean(icc_draws), color = "#2A9D8F",
             linewidth = 1, linetype = "dashed") +
  labs(
    title    = "Posterior Distribution: Artist-Level ICC",
    subtitle = "Proportion of popularity variance attributable to artist-level factors",
    x        = "ICC (Artist)",
    y        = "Density"
  )

print(p_icc_draws)
save_fig("30_icc_posterior.pdf", w = 8, h = 5)


# =============================================================================
# 10.  ARTIST-LEVEL RANDOM EFFECTS
# =============================================================================
message("\n── Artist random effects ──")

# Extract artist random effect draws
# u_artist[j,1] = intercept deviate; u_artist[j,2] = LMC slope deviate

artist_re_df <- map_dfr(seq_along(artist_levels), function(j) {
  int_col  <- paste0("u_artist[", j, ",1]")
  slp_col  <- paste0("u_artist[", j, ",2]")

  int_draws <- if (int_col %in% names(draws_df)) draws_df[[int_col]] else NA
  slp_draws <- if (slp_col %in% names(draws_df)) draws_df[[slp_col]] else NA

  tibble(
    artist_code   = artist_levels[j],
    re_int_mean   = mean(int_draws, na.rm = TRUE),
    re_int_q5     = quantile(int_draws, 0.05, na.rm = TRUE),
    re_int_q95    = quantile(int_draws, 0.95, na.rm = TRUE),
    re_slp_mean   = mean(slp_draws, na.rm = TRUE),
    re_slp_q5     = quantile(slp_draws, 0.05, na.rm = TRUE),
    re_slp_q95    = quantile(slp_draws, 0.95, na.rm = TRUE),
  )
}) %>%
  left_join(song_df %>% distinct(artist_code, genre_cluster, orientation),
            by = "artist_code")

# ── Caterpillar plot: artist intercept deviates ───────────────────────────────
p_cat_int <- artist_re_df %>%
  mutate(artist_code = fct_reorder(artist_code, re_int_mean)) %>%
  ggplot(aes(x = re_int_mean, y = artist_code, color = genre_cluster)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = re_int_q5, xmax = re_int_q95),
                 height = 0.4, linewidth = 0.8) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Set2", name = "Genre") +
  labs(
    title    = "Artist Random Intercepts (Posterior 90% CI)",
    subtitle = "Deviation from genre-adjusted baseline popularity",
    x        = "Random intercept deviate (log-odds)",
    y        = NULL
  )

print(p_cat_int)
save_fig("31_artist_re_intercepts.pdf", w = 9, h = 8)

# ── Artist LMC slope heterogeneity ───────────────────────────────────────────
p_cat_slp <- artist_re_df %>%
  mutate(artist_code = fct_reorder(artist_code, re_slp_mean)) %>%
  ggplot(aes(x = re_slp_mean, y = artist_code, color = orientation)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = re_slp_q5, xmax = re_slp_q95),
                 height = 0.4, linewidth = 0.8) +
  geom_point(size = 2.5) +
  scale_color_manual(values = ORIENTATION_COLORS, name = "Orientation") +
  labs(
    title    = "Artist-Varying LMC Slopes (Posterior 90% CI)",
    subtitle = "Does the LMC→popularity effect differ across artists?",
    x        = "Artist-specific LMC slope deviate",
    y        = NULL
  )

print(p_cat_slp)
save_fig("32_artist_lmc_slopes.pdf", w = 9, h = 8)

# ── Scatter: intercept vs. LMC slope (shows correlation structure) ───────────
p_re_scatter <- artist_re_df %>%
  ggplot(aes(x = re_int_mean, y = re_slp_mean,
             color = genre_cluster, label = artist_code)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey70") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey70") +
  geom_point(size = 3, alpha = 0.8) +
  ggrepel::geom_text_repel(size = 2.8, max.overlaps = 12) +
  scale_color_brewer(palette = "Set2", name = "Genre") +
  labs(
    title    = "Artist Random Effects: Intercept vs. LMC Slope",
    subtitle = sprintf("Posterior correlation ρ = %.3f",
                       mean(draws_df$rho_artist)),
    x        = "Random intercept (baseline popularity)",
    y        = "Random LMC slope (sensitivity to congruence)"
  )

print(p_re_scatter)
save_fig("33_re_intercept_vs_slope.pdf", w = 9, h = 7)


# =============================================================================
# 11.  SENSITIVITY: PRIOR VS. POSTERIOR
# =============================================================================
message("\n── Prior vs. posterior for beta_lmc ──")

prior_draws <- rnorm(8000, 0, 0.5)   # prior for beta_lmc
post_draws  <- draws_df$beta_lmc

p_prior_post <- ggplot() +
  geom_density(data = data.frame(x = prior_draws), aes(x = x),
               fill = "grey70", alpha = 0.6, color = "grey50") +
  geom_density(data = data.frame(x = post_draws), aes(x = x),
               fill = "#2A9D8F", alpha = 0.6, color = "#2A9D8F") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  annotate("text", x = -0.8, y = max(density(prior_draws)$y) * 0.7,
           label = "Prior\nN(0, 0.5)", color = "grey40", size = 3.5) +
  annotate("text", x = mean(post_draws) + 0.15,
           y = max(density(post_draws)$y) * 0.8,
           label = "Posterior", color = "#2A9D8F", size = 3.5) +
  labs(
    title    = "Prior vs. Posterior: LMC Main Effect (β_lmc)",
    subtitle = "If posterior substantially departs from prior, data is informative",
    x        = "Coefficient value (log-odds scale)",
    y        = "Density"
  )

print(p_prior_post)
save_fig("34_prior_posterior.pdf", w = 8, h = 5)

# Bayes Factor approximation via Savage-Dickey ratio
# BF10 = p(data | H1) / p(data | H0) ≈ prior(0) / posterior(0)
prior_at_0    <- dnorm(0, 0, 0.5)
posterior_at_0 <- density(post_draws, n = 2048)
post_density_at_0 <- approx(posterior_at_0$x, posterior_at_0$y, xout = 0)$y
BF10 <- prior_at_0 / post_density_at_0

message(sprintf("  Savage-Dickey BF10 (H1: beta_lmc ≠ 0): %.2f", BF10))
message(sprintf("  Interpretation: %.1fx more likely under H1 than H0", BF10))


# =============================================================================
# SUMMARY OUTPUT
# =============================================================================

message("\n", strrep("=", 60))
message("Bayesian model complete.")
message(sprintf("  R-hat max (key params): %.4f",
                max(diag_key$rhat, na.rm = TRUE)))
message(sprintf("  ESS_bulk min          : %.0f",
                min(diag_key$ess_bulk, na.rm = TRUE)))
message(sprintf("  Divergences           : %d", n_div))
message(sprintf("  LOO-IC                : %.1f",
                loo_result$estimates["looic", "Estimate"]))
message(sprintf("  β_lmc posterior mean  : %.4f  [%.4f, %.4f]",
                mean(post_draws),
                quantile(post_draws, 0.025),
                quantile(post_draws, 0.975)))
message(sprintf("  P(β_lmc > 0)          : %.3f", mean(post_draws > 0)))
message(sprintf("  Savage-Dickey BF10    : %.2f", BF10))
message(sprintf("  Artist ICC            : %.3f", mean(icc_draws)))
message("\n  Figures → ", fig_dir)
message("  Tables  → ", tab_dir)
message(strrep("=", 60))
