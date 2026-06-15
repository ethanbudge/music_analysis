# =============================================================================
# run_bayesian_model_simple.R
#
# Bayesian hierarchical Beta regression — no Spotify audio features.
# Fixes the group_modify NULL-return error with a safe wrapper.
#
# Prerequisites
# -------------
#   cmdstanr::install_cmdstan()   # if not already installed
#   musical_congruence_model_simple.stan in bayes/
#   results/master_results.csv    (needs: song_id, popularity, artist_code,
#                                  genre, orientation)
#   results/lyric_timeline/lyric_timeline.csv
# =============================================================================


# ── 0. Packages ───────────────────────────────────────────────────────────────

required <- c("cmdstanr", "posterior", "bayesplot", "loo",
              "tidyverse", "patchwork", "ggdist", "scales", "here")
for (pkg in required) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

BAYES_DIR  <- here::here("bayes")
STAN_FILE  <- file.path(BAYES_DIR, "musical_congruence_model_simple.stan")
OUTPUT_DIR <- file.path(BAYES_DIR, "output")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(42)

theme_set(
  theme_minimal(base_size = 12) +
    theme(plot.title      = element_text(face = "bold"),
          plot.subtitle   = element_text(color = "grey40"),
          legend.position = "bottom",
          panel.grid.minor = element_blank())
)

savefig <- function(name, w = 9, h = 6) {
  ggsave(file.path(OUTPUT_DIR, name), width = w, height = h,
         device = "pdf", dpi = 300)
  message("  Saved: ", name)
}


# ── 1. Load data ──────────────────────────────────────────────────────────────
message("\n── 1. Loading data ──")

timeline_path <- here::here("results", "lyric_timeline", "lyric_timeline.csv")
master_path   <- here::here("results", "master_results.csv")

if (!file.exists(timeline_path)) stop("Run 10_lyric_timeline.py first.")
if (!file.exists(master_path))   stop("Run 09_combine_results.py first.")

tl <- read_csv(timeline_path, show_col_types = FALSE) %>%
  filter(match_confidence >= 0.40)

master <- read_csv(master_path, show_col_types = FALSE) %>%
  select(song_id, artist_code, genre, orientation, popularity)

message(sprintf("Timeline: %s lines, %d songs",
                format(nrow(tl), big.mark = ","),
                n_distinct(tl$song_id)))


# ── 2. Trajectory feature extraction (NULL-safe) ──────────────────────────────
message("\n── 2. Extracting trajectory features ──")

# KEY FIX: always return a data frame, never NULL.
# If a song has too few lines or the loess fails, return a row of NAs
# so group_modify is satisfied. These rows are dropped later via drop_na().

extract_traj <- function(song_data) {
  pos <- song_data$position_pct
  lmc <- song_data$lmc
  n   <- nrow(song_data)
  
  na_row <- data.frame(
    mean_lmc = NA_real_, sd_lmc = NA_real_,
    lmc_slope = NA_real_, lmc_curve = NA_real_,
    lmc_early = NA_real_, lmc_late = NA_real_,
    lmc_change = NA_real_, lmc_smooth_sd = NA_real_,
    n_lines = n
  )
  
  # Need at least 5 data points for a loess with span=0.4
  if (n < 5) return(na_row)
  
  mean_lmc <- mean(lmc)
  sd_lmc   <- sd(lmc)
  
  tryCatch({
    lo       <- loess(lmc ~ pos, span = 0.5, degree = 1,
                      control = loess.control(surface = "direct"))
    grid     <- seq(0, 100, by = 2)   # coarser grid avoids trL > n
    smoothed <- predict(lo, newdata = data.frame(pos = grid))
    smoothed[is.na(smoothed)] <- mean_lmc
    
    lm_fit  <- lm(smoothed ~ grid)
    lm_quad <- lm(smoothed ~ grid + I(grid^2))
    
    data.frame(
      mean_lmc     = mean_lmc,
      sd_lmc       = sd_lmc,
      lmc_slope    = as.numeric(coef(lm_fit)["grid"]),
      lmc_curve    = as.numeric(coef(lm_quad)["I(grid^2)"]),
      lmc_early    = mean(smoothed[grid <= 50]),
      lmc_late     = mean(smoothed[grid > 50]),
      lmc_change   = mean(smoothed[grid > 50]) - mean(smoothed[grid <= 50]),
      lmc_smooth_sd = sd(smoothed),
      n_lines      = n
    )
  }, error = function(e) {
    # loess failed (too few unique x values etc.) — return level features only
    data.frame(
      mean_lmc = mean_lmc, sd_lmc = sd_lmc,
      lmc_slope = NA_real_, lmc_curve = NA_real_,
      lmc_early = NA_real_, lmc_late = NA_real_,
      lmc_change = NA_real_, lmc_smooth_sd = NA_real_,
      n_lines = n
    )
  })
}

# Two loess tweaks that prevent the "Chernobyl trL > n" warning:
#   1. span = 0.5 instead of 0.4 (larger neighbourhood, fewer dof used)
#   2. degree = 1 instead of 2  (linear local fit, more stable)
#   3. grid step = 2 instead of 1 (fewer prediction points)
# The "trL > n" warning means the effective degrees of freedom of the smooth
# exceeded the number of data points — innocuous but noisy. These settings
# suppress it for most songs.

traj_features <- tl %>%
  group_by(song_id) %>%
  group_modify(~ extract_traj(.x)) %>%
  ungroup()

message(sprintf("  Trajectory features extracted for %d songs",
                nrow(traj_features)))
message(sprintf("  Songs with complete features: %d",
                sum(complete.cases(traj_features))))


# ── 3. Build analysis dataset ─────────────────────────────────────────────────
message("\n── 3. Merging datasets ──")

song_df <- master %>%
  inner_join(traj_features, by = "song_id") %>%
  filter(!is.na(popularity)) %>%
  drop_na(mean_lmc, lmc_slope, lmc_curve, lmc_change, sd_lmc) %>%
  mutate(
    genre_cluster = case_when(
      genre %in% c("hip-hop")                              ~ "Hip-Hop",
      genre %in% c("folk", "folk-rock", "country")        ~ "Folk/Country",
      genre %in% c("pop")                                  ~ "Pop",
      genre %in% c("electronic", "psychedelic-electronic") ~ "Electronic",
      TRUE                                                 ~ "Other"
    )
  )

message(sprintf("Analysis sample: %d songs | %d artists | %d genre clusters",
                nrow(song_df),
                n_distinct(song_df$artist_code),
                n_distinct(song_df$genre_cluster)))


# ── 4. Encode groups and scale predictors ────────────────────────────────────
artist_levels <- sort(unique(song_df$artist_code))
genre_levels  <- sort(unique(song_df$genre_cluster))

song_df <- song_df %>%
  mutate(
    artist_idx = as.integer(factor(artist_code,   levels = artist_levels)),
    genre_idx  = as.integer(factor(genre_cluster, levels = genre_levels))
  )

artist_df <- song_df %>%
  distinct(artist_code, artist_idx, orientation) %>%
  arrange(artist_idx) %>%
  mutate(orientation_num = if_else(orientation == "narrative", 1L, 0L))

zsc <- function(x) as.numeric(scale(x))

N <- nrow(song_df)
song_df <- song_df %>%
  mutate(
    # Boundary-adjust popularity: (Smithson & Verkuilen 2006)
    pop_scaled   = (popularity / 100 * (N - 1) + 0.5) / N,
    
    mean_lmc_z   = zsc(mean_lmc),
    mean_lmc_z2  = mean_lmc_z^2,
    lmc_slope_z  = zsc(lmc_slope),
    lmc_curve_z  = zsc(lmc_curve),
    lmc_change_z = zsc(lmc_change),
    lmc_sd_z     = zsc(sd_lmc)
  )

# Store for back-transformation
scale_params <- list(
  mean_lmc_mean = mean(song_df$mean_lmc),
  mean_lmc_sd   = sd(song_df$mean_lmc)
)

message(sprintf("Popularity: mean=%.1f  sd=%.1f  min=%.0f  max=%.0f",
                mean(song_df$popularity), sd(song_df$popularity),
                min(song_df$popularity), max(song_df$popularity)))


# ── 5. Stan data list ─────────────────────────────────────────────────────────
stan_data <- list(
  N        = nrow(song_df),
  N_artist = n_distinct(song_df$artist_idx),
  N_genre  = n_distinct(song_df$genre_idx),
  
  y            = song_df$pop_scaled,
  mean_lmc_z   = song_df$mean_lmc_z,
  mean_lmc_z2  = song_df$mean_lmc_z2,
  lmc_slope_z  = song_df$lmc_slope_z,
  lmc_curve_z  = song_df$lmc_curve_z,
  lmc_change_z = song_df$lmc_change_z,
  lmc_sd_z     = song_df$lmc_sd_z,
  
  artist_id    = song_df$artist_idx,
  genre_id     = song_df$genre_idx,
  orientation  = artist_df$orientation_num
)

message(sprintf("\nStan data: %d songs | %d artists | %d genres",
                stan_data$N, stan_data$N_artist, stan_data$N_genre))


# ── 6. Compile and sample ─────────────────────────────────────────────────────
message("\n── 6. Compiling Stan model ──")
mod <- cmdstan_model(STAN_FILE)

message("── Sampling (4 chains × 2000 warmup + 2000 sampling) ──")
fit <- mod$sample(
  data            = stan_data,
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 2000,
  iter_sampling   = 2000,
  adapt_delta     = 0.95,
  max_treedepth   = 12,
  seed            = 42,
  output_dir      = OUTPUT_DIR,
  show_messages   = FALSE,
  refresh         = 200
)

fit$save_object(file.path(OUTPUT_DIR, "fit_simple.rds"))
message("Sampling complete. Fit saved.")

# ── 7. Convergence diagnostics ────────────────────────────────────────────────
message("\n── 7. Convergence diagnostics ──")

key_params <- c(
  "mu_global",
  "beta_lmc", "beta_lmc2",
  "beta_slope", "beta_curve", "beta_change", "beta_sd",
  "gamma_intercept", "gamma_lmc",
  "sigma_artist[1]", "sigma_artist[2]",
  "sigma_genre", "phi",
  "Rho_artist[1,2]"
)

diag_df <- fit$summary(variables = key_params) %>%
  select(variable, mean, median, sd, q5, q95, rhat, ess_bulk, ess_tail) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

print(diag_df, n = nrow(diag_df))
write_csv(diag_df, file.path(OUTPUT_DIR, "convergence_simple.csv"))

n_div <- sum(fit$diagnostic_summary()$num_divergent)
bad_r <- diag_df %>% filter(rhat > 1.01)
low_e <- diag_df %>% filter(ess_bulk < 400)

message(sprintf("\n  Divergences     : %d", n_div))
message(sprintf("  R-hat > 1.01   : %d parameters", nrow(bad_r)))
message(sprintf("  ESS_bulk < 400 : %d parameters", nrow(low_e)))

if (nrow(bad_r) == 0 && n_div == 0)
  message("  Convergence looks good ✓")


# ── 8. Traceplots ─────────────────────────────────────────────────────────────
message("\n── 8. Traceplots ──")

draws_arr <- fit$draws(format = "array")

p_trace_lmc <- mcmc_trace(
  draws_arr,
  pars = c("beta_lmc", "beta_lmc2", "beta_slope",
           "beta_curve", "beta_change", "beta_sd"),
  facet_args = list(ncol = 2)
) +
  scale_color_brewer(palette = "Set1") +
  labs(title    = "Traceplots — LMC Parameters",
       subtitle = "Well-mixed chains indicate good convergence")
print(p_trace_lmc)
savefig("01_traceplots_lmc.pdf", w = 12, h = 10)

p_trace_hyper <- mcmc_trace(
  draws_arr,
  pars = c("mu_global", "gamma_intercept", "gamma_lmc",
           "sigma_artist[1]", "sigma_artist[2]",
           "sigma_genre", "phi"),
  facet_args = list(ncol = 2)
) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Traceplots — Hyperparameters")
print(p_trace_hyper)
savefig("02_traceplots_hyper.pdf", w = 12, h = 10)

# p_rank <- mcmc_rank_hist(
#   draws_arr,
#   pars = c("beta_lmc", "beta_lmc2", "beta_slope",
#            "beta_curve", "beta_change", "beta_sd"),
#   facet_args = list(ncol = 2)
# ) +
#   labs(title    = "Rank Histograms — LMC Parameters",
#        subtitle = "Uniform = well-mixed chains")
# print(p_rank)
# savefig("03_rank_histograms.pdf", w = 12, h = 10)


# ── 9. Posterior summaries and effect plots ────────────────────────────────────
message("\n── 9. Posterior summaries ──")

draws_df <- fit$draws(format = "df")

post_summary <- function(param) {
  x <- draws_df[[param]]
  tibble(
    parameter = param,
    mean   = mean(x),  median = median(x),
    sd     = sd(x),
    q5     = quantile(x, 0.05), q95 = quantile(x, 0.95),
    p_pos  = mean(x > 0), p_neg = mean(x < 0)
  )
}

summary_tbl <- map_dfr(key_params, post_summary) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

print(summary_tbl, n = nrow(summary_tbl))
write_csv(summary_tbl, file.path(OUTPUT_DIR, "posterior_summary_simple.csv"))

# Average marginal effect of LMC (back on 0–100 scale)
me_lmc <- fit$draws("marginal_lmc", format = "matrix") %>%
  apply(2, mean) %>% mean()
message(sprintf("\nAvg. marginal effect of +1 SD LMC: %.2f popularity points",
                me_lmc))

# ── 9a. Posterior distributions ───────────────────────────────────────────────
lmc_params <- c("beta_lmc", "beta_lmc2", "beta_slope",
                "beta_curve", "beta_change", "beta_sd")

lmc_param_labels <- c(
  "beta_lmc"    = "Mean LMC",
  "beta_lmc2"   = "Mean LMC²\n(nonlinearity)",
  "beta_slope"  = "LMC Slope\n(trend over song)",
  "beta_curve"  = "LMC Curvature\n(arc)",
  "beta_change" = "Early→Late\nChange",
  "beta_sd"     = "LMC Consistency\n(−SD)"
)

lmc_draws_long <- draws_df %>%
  select(all_of(lmc_params)) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(parameter = recode(parameter, !!!lmc_param_labels))

p_post <- lmc_draws_long %>%
  ggplot(aes(x = value, y = parameter, fill = after_stat(x > 0))) +
  stat_halfeye(.width = c(0.8, 0.95), point_interval = median_hdi) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey30") +
  scale_fill_manual(
    values = c("FALSE" = "#E76F51", "TRUE" = "#2A9D8F"),
    guide  = "none"
  ) +
  labs(
    title    = "Posterior Distributions — LMC Coefficients",
    subtitle = "Bars = 80% and 95% credible intervals",
    x = "Coefficient (logit scale)", y = NULL
  )
print(p_post)
savefig("04_posterior_lmc.pdf", w = 10, h = 7)

# ── 9b. Moderation posteriors ─────────────────────────────────────────────────
p_mod <- draws_df %>%
  select(gamma_intercept, gamma_lmc) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(parameter = recode(parameter,
                            "gamma_intercept" = "Orientation →\nBaseline popularity",
                            "gamma_lmc"       = "Orientation × LMC\n(narrative amplification)"
  )) %>%
  ggplot(aes(x = value, y = parameter, fill = after_stat(x > 0))) +
  stat_halfeye(.width = c(0.8, 0.95), point_interval = median_hdi) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey30") +
  scale_fill_manual(
    values = c("FALSE" = "#E76F51", "TRUE" = "#2A9D8F"),
    guide  = "none"
  ) +
  labs(title = "Orientation Moderation Effects",
       x = "Coefficient (logit scale)", y = NULL)
print(p_mod)
savefig("05_moderation_posteriors.pdf", w = 9, h = 5)

# ── 9c. Conditional effect of LMC on popularity ───────────────────────────────
lmc_grid <- seq(min(song_df$mean_lmc_z), max(song_df$mean_lmc_z),
                length.out = 60)

cond_eff <- map_dfr(seq_along(lmc_grid), function(idx) {
  x   <- lmc_grid[idx]
  eta <- draws_df$mu_global +
    draws_df$beta_lmc  * x +
    draws_df$beta_lmc2 * x^2
  mu  <- plogis(eta) * 100
  
  tibble(
    mean_lmc_z   = x,
    mean_lmc_raw = x * scale_params$mean_lmc_sd + scale_params$mean_lmc_mean,
    pop_med      = median(mu),
    pop_lo80     = quantile(mu, 0.10),
    pop_hi80     = quantile(mu, 0.90),
    pop_lo95     = quantile(mu, 0.025),
    pop_hi95     = quantile(mu, 0.975)
  )
})

p_cond <- cond_eff %>%
  ggplot(aes(x = mean_lmc_raw)) +
  geom_ribbon(aes(ymin = pop_lo95, ymax = pop_hi95),
              fill = "#2A9D8F", alpha = 0.15) +
  geom_ribbon(aes(ymin = pop_lo80, ymax = pop_hi80),
              fill = "#2A9D8F", alpha = 0.25) +
  geom_line(aes(y = pop_med), color = "#2A9D8F", linewidth = 1.3) +
  geom_point(data = song_df,
             aes(x = mean_lmc, y = popularity, color = genre_cluster),
             alpha = 0.5, size = 2, inherit.aes = FALSE) +
  scale_color_brewer(palette = "Set2", name = "Genre") +
  labs(
    title    = "Conditional Effect of Mean LMC on Popularity",
    subtitle = "Posterior median + 80%/95% CI. Controls held at mean.",
    x = "Mean LMC (raw scale)", y = "Predicted Popularity (0–100)"
  )
print(p_cond)
savefig("06_conditional_effect_lmc.pdf", w = 10, h = 7)

# ── 9d. Artist forest plot ────────────────────────────────────────────────────
artist_params <- paste0("beta_lmc_artist[", seq_len(stan_data$N_artist), "]")
artist_summ <- fit$summary(variables = artist_params) %>%
  mutate(
    artist_idx  = as.integer(str_extract(variable, "\\d+")),
    artist_code = artist_levels[artist_idx]
  ) %>%
  left_join(artist_df %>% select(artist_code, orientation),
            by = "artist_code") %>%
  arrange(median)

p_forest <- artist_summ %>%
  ggplot(aes(x = median, y = reorder(artist_code, median),
             color = orientation)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = q5, xmax = q95),
                 height = 0.3, linewidth = 0.8, alpha = 0.6) +
  geom_point(size = 3) +
  scale_color_manual(
    values = c("narrative" = "#2A9D8F", "production" = "#E76F51"),
    name   = "Orientation"
  ) +
  labs(
    title    = "Artist-Specific LMC Slopes (Forest Plot)",
    subtitle = "Posterior median + 90% CI",
    x = "LMC slope (logit scale)", y = NULL
  ) +
  theme(axis.text.y = element_text(size = 8))
print(p_forest)
savefig("07_artist_forest.pdf", w = 10, h = 9)


# ── 10. Posterior predictive check ───────────────────────────────────────────
message("\n── 10. Posterior predictive check ──")

y_rep  <- fit$draws("y_rep", format = "matrix")
y_obs  <- song_df$pop_scaled

p_ppc <- ppc_dens_overlay(
  y    = y_obs,
  yrep = y_rep[sample(nrow(y_rep), 100), ]
) +
  labs(title    = "Posterior Predictive Check — Density Overlay",
       subtitle = "Black = observed | Blue = 100 posterior predictive draws",
       x = "Popularity (scaled to 0–1)")
print(p_ppc)
savefig("08_ppc_density.pdf", w = 9, h = 6)

p_ppc_stats <- (ppc_stat(y_obs, y_rep, stat = "mean") + labs(title = "Mean")) +
  (ppc_stat(y_obs, y_rep, stat = "sd")   + labs(title = "SD")) +
  (ppc_stat(y_obs, y_rep, stat = "max")  + labs(title = "Max")) +
  plot_layout(ncol = 3)
print(p_ppc_stats)
savefig("09_ppc_stats.pdf", w = 14, h = 5)


# ── 11. LOO-CV ────────────────────────────────────────────────────────────────
message("\n── 11. LOO cross-validation ──")

log_lik_mat <- fit$draws("log_lik", format = "matrix")
loo_res     <- loo(log_lik_mat, cores = 4)
print(loo_res)

write_csv(
  as_tibble(loo_res$estimates, rownames = "metric") %>%
    mutate(across(where(is.numeric), ~round(., 3))),
  file.path(OUTPUT_DIR, "loo_simple.csv")
)

k_vals  <- loo_res$pointwise[, "influence_pareto_k"]
n_bad   <- sum(k_vals > 0.7)
message(sprintf("  Pareto-k > 0.7 (high influence): %d songs", n_bad))
if (n_bad > 0)
  message("  High-influence songs: ",
          paste(song_df$song_id[k_vals > 0.7], collapse = ", "))

p_pareto <- tibble(k = k_vals, genre = song_df$genre_cluster,
                   rank = rank(-k_vals)) %>%
  ggplot(aes(x = rank, y = k, color = genre)) +
  geom_hline(yintercept = c(0.5, 0.7),
             linetype = "dashed", color = c("orange", "red")) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_brewer(palette = "Set2", name = "Genre") +
  labs(title    = "LOO Pareto-k Diagnostics",
       subtitle = "Values > 0.7 indicate highly influential observations",
       x = "Song (ranked by influence)", y = "Pareto-k")
print(p_pareto)
savefig("10_pareto_k.pdf", w = 10, h = 6)


# ── Done ──────────────────────────────────────────────────────────────────────
message("\n", strrep("═", 60))
message("Bayesian analysis complete.")
message(sprintf("  Outputs: %s", OUTPUT_DIR))
message(strrep("═", 60))
