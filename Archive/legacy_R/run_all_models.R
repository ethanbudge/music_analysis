# =============================================================================
# run_all_models.R
#
# Fits four Bayesian hierarchical Beta regression models, one per LMC
# operationalisation, and compares them via LOO-CV:
#
#   M1  Track-level MuLan       — full-song audio × full lyrics
#   M2  Track-level CLAP        — full-song audio × full lyrics (different model)
#   M3  Segment-level MuLan     — chorus/verse/consistency decomposition
#   M4  Timeline-level MuLan    — trajectory shape (slope, curve, change, SD)
#
# All models share:
#   - Beta regression likelihood
#   - Partial pooling by artist (random intercepts) and genre (random intercepts)
#   - Release date control (song_age_z)
#   - Orientation moderation (γ terms)
#   - No Spotify audio features
#
# The key scientific question is which operationalisation of LMC best
# predicts popularity — i.e., does temporal or structural information
# about congruence matter over and above the aggregate score?
# =============================================================================


# ── 0. Setup ──────────────────────────────────────────────────────────────────

required <- c("cmdstanr", "posterior", "bayesplot", "loo",
              "tidyverse", "patchwork", "ggdist", "here", "scales")
for (pkg in required) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

BAYES_DIR  <- here::here("bayes")
OUTPUT_DIR <- file.path(BAYES_DIR, "output")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(42)

theme_set(
  theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold"),
          plot.subtitle = element_text(color = "grey40"),
          legend.position = "bottom",
          panel.grid.minor = element_blank())
)

savefig <- function(name, w = 9, h = 6)
  ggsave(file.path(OUTPUT_DIR, name), width = w, height = h,
         device = "pdf", dpi = 300)

savetable <- function(x, name) {
  sink(file.path(OUTPUT_DIR, name)); print(x); sink()
}

zsc <- function(x) as.numeric(scale(x))

# Smithson & Verkuilen (2006) boundary adjustment
boundary_adjust <- function(y01, N) (y01 * (N - 1) + 0.5) / N


# ── 1. Load data ──────────────────────────────────────────────────────────────
message("\n══ 1. Loading data ══")

master <- read_csv(here::here("results", "master_results.csv"),
                   show_col_types = FALSE)

# ── Song age from release date ─────────────────────────────────────────────
master <- master %>%
  mutate(
    release_year = as.numeric(substr(release_date, 1, 4)),
    song_age     = 2025 - release_year,
    genre_cluster = case_when(
      genre %in% c("hip-hop")                              ~ "Hip-Hop",
      genre %in% c("folk", "folk-rock", "country")        ~ "Folk/Country",
      genre %in% c("pop")                                  ~ "Pop",
      genre %in% c("electronic", "psychedelic-electronic") ~ "Electronic",
      TRUE                                                 ~ "Other"
    )
  )

# ── Timeline data ─────────────────────────────────────────────────────────
tl_path <- here::here("results", "lyric_timeline", "lyric_timeline.csv")
has_timeline <- file.exists(tl_path)

if (has_timeline) {
  tl <- read_csv(tl_path, show_col_types = FALSE) %>%
    filter(match_confidence >= 0.40)
} else {
  warning("lyric_timeline.csv not found. Model 4 will be skipped.")
}

# ── Segment data ──────────────────────────────────────────────────────────
seg_path <- here::here("results", "segment_analysis", "segment_summary.csv")
has_segments <- file.exists(seg_path)

if (has_segments) {
  seg <- read_csv(seg_path, show_col_types = FALSE)
} else {
  warning("segment_summary.csv not found. Model 3 will be skipped.")
}

# ── Trajectory feature extractor (NULL-safe) ──────────────────────────────
extract_traj <- function(song_data) {
  pos <- song_data$position_pct
  lmc <- song_data$lmc
  n   <- nrow(song_data)

  na_row <- data.frame(
    mean_lmc_tl = NA_real_, sd_lmc_tl = NA_real_,
    lmc_slope = NA_real_, lmc_curve = NA_real_,
    lmc_change = NA_real_, lmc_smooth_sd = NA_real_,
    n_lines = n
  )

  if (n < 5) return(na_row)

  mean_lmc <- mean(lmc)
  sd_lmc   <- sd(lmc)

  tryCatch({
    lo       <- loess(lmc ~ pos, span = 0.5, degree = 1,
                      control = loess.control(surface = "direct"))
    grid     <- seq(0, 100, by = 2)
    smoothed <- predict(lo, newdata = data.frame(pos = grid))
    smoothed[is.na(smoothed)] <- mean_lmc

    lm_fit  <- lm(smoothed ~ grid)
    lm_quad <- lm(smoothed ~ grid + I(grid^2))

    data.frame(
      mean_lmc_tl   = mean_lmc,
      sd_lmc_tl     = sd_lmc,
      lmc_slope     = as.numeric(coef(lm_fit)["grid"]),
      lmc_curve     = as.numeric(coef(lm_quad)["I(grid^2)"]),
      lmc_change    = mean(smoothed[grid > 50]) - mean(smoothed[grid <= 50]),
      lmc_smooth_sd = sd(smoothed),
      n_lines       = n
    )
  }, error = function(e) na_row)
}

if (has_timeline) {
  traj_features <- tl %>%
    group_by(song_id) %>%
    group_modify(~ extract_traj(.x)) %>%
    ungroup()
}


# =============================================================================
# 2. Prepare model-specific datasets
# =============================================================================
message("\n══ 2. Preparing datasets ══")

# ── Helper: build common columns + group indices ──────────────────────────
build_base <- function(df) {
  artist_levels <- sort(unique(df$artist_code))
  genre_levels  <- sort(unique(df$genre_cluster))

  df <- df %>%
    mutate(
      artist_idx = as.integer(factor(artist_code,  levels = artist_levels)),
      genre_idx  = as.integer(factor(genre_cluster, levels = genre_levels)),
      pop_scaled = boundary_adjust(popularity / 100, n()),
      song_age_z = zsc(song_age)
    )

  artist_tbl <- df %>%
    distinct(artist_code, artist_idx, orientation) %>%
    arrange(artist_idx) %>%
    mutate(orientation_num = if_else(orientation == "narrative", 1, 0))

  list(df = df, artist_levels = artist_levels, genre_levels = genre_levels,
       artist_tbl = artist_tbl)
}


# ── M1: Track-level MuLan ─────────────────────────────────────────────────
df_m1_raw <- master %>%
  filter(!is.na(popularity), !is.na(lmc_mulan), !is.na(song_age)) %>%
  drop_na(artist_code, genre_cluster)

b1       <- build_base(df_m1_raw)
df_m1    <- b1$df
m1_data  <- list(
  N = nrow(df_m1), N_artist = max(df_m1$artist_idx),
  N_genre = max(df_m1$genre_idx),
  y = df_m1$pop_scaled,
  lmc_z = zsc(df_m1$lmc_mulan), lmc_z2 = zsc(df_m1$lmc_mulan)^2,
  song_age_z = df_m1$song_age_z,
  artist_id = df_m1$artist_idx, genre_id = df_m1$genre_idx,
  orientation = b1$artist_tbl$orientation_num
)
# Fix: lmc_z2 should be square of lmc_z, not independently z-scored
m1_data$lmc_z2 <- m1_data$lmc_z^2

message(sprintf("  M1 (MuLan track):  %d songs, %d artists",
                m1_data$N, m1_data$N_artist))


# ── M2: Track-level CLAP ──────────────────────────────────────────────────
df_m2_raw <- master %>%
  filter(!is.na(popularity), !is.na(lmc_clap), !is.na(song_age)) %>%
  drop_na(artist_code, genre_cluster)

b2       <- build_base(df_m2_raw)
df_m2    <- b2$df
m2_data  <- list(
  N = nrow(df_m2), N_artist = max(df_m2$artist_idx),
  N_genre = max(df_m2$genre_idx),
  y = df_m2$pop_scaled,
  lmc_z = zsc(df_m2$lmc_clap), lmc_z2 = zsc(df_m2$lmc_clap)^2,
  song_age_z = df_m2$song_age_z,
  artist_id = df_m2$artist_idx, genre_id = df_m2$genre_idx,
  orientation = b2$artist_tbl$orientation_num
)
m2_data$lmc_z2 <- m2_data$lmc_z^2

message(sprintf("  M2 (CLAP track):   %d songs, %d artists",
                m2_data$N, m2_data$N_artist))

# ── M3: Segment-level MuLan ───────────────────────────────────────────────
if (has_segments) {
  df_m3_raw <- master %>%
    inner_join(seg, by = "song_id") %>%
    filter(!is.na(popularity), !is.na(song_age)) %>%
    mutate(
      # Fill missing chorus/verse with overall mean LMC for that song
      seg_chorus = if_else(is.na(mean_lmc_chorus), mean_lmc_all, mean_lmc_chorus),
      seg_verse  = if_else(is.na(mean_lmc_verse),  mean_lmc_all, mean_lmc_verse),
      seg_sd     = if_else(is.na(sd_lmc),           0,            sd_lmc)
    ) %>%
    drop_na(artist_code, genre_cluster, seg_chorus, seg_verse, seg_sd)

  b3       <- build_base(df_m3_raw)
  df_m3    <- b3$df
  m3_data  <- list(
    N = nrow(df_m3), N_artist = max(df_m3$artist_idx),
    N_genre = max(df_m3$genre_idx),
    y = df_m3$pop_scaled,
    chorus_lmc_z = zsc(df_m3$seg_chorus),
    verse_lmc_z  = zsc(df_m3$seg_verse),
    lmc_sd_z     = zsc(df_m3$seg_sd),
    song_age_z   = df_m3$song_age_z,
    artist_id    = df_m3$artist_idx, genre_id = df_m3$genre_idx,
    orientation  = b3$artist_tbl$orientation_num
  )

  message(sprintf("  M3 (Segment):      %d songs, %d artists",
                  m3_data$N, m3_data$N_artist))
} else {
  message("  M3 (Segment):      SKIPPED — no segment data")
}


# ── M4: Timeline-level MuLan ──────────────────────────────────────────────
if (has_timeline) {
  df_m4_raw <- master %>%
    inner_join(traj_features, by = "song_id") %>%
    filter(!is.na(popularity), !is.na(song_age)) %>%
    drop_na(artist_code, genre_cluster,
            mean_lmc_tl, lmc_slope, lmc_curve, lmc_change, sd_lmc_tl)

  b4       <- build_base(df_m4_raw)
  df_m4    <- b4$df
  m4_data  <- list(
    N = nrow(df_m4), N_artist = max(df_m4$artist_idx),
    N_genre = max(df_m4$genre_idx),
    y = df_m4$pop_scaled,
    mean_lmc_z   = zsc(df_m4$mean_lmc_tl),
    mean_lmc_z2  = zsc(df_m4$mean_lmc_tl)^2,
    lmc_slope_z  = zsc(df_m4$lmc_slope),
    lmc_curve_z  = zsc(df_m4$lmc_curve),
    lmc_change_z = zsc(df_m4$lmc_change),
    lmc_sd_z     = zsc(df_m4$sd_lmc_tl),
    song_age_z   = df_m4$song_age_z,
    artist_id    = df_m4$artist_idx, genre_id = df_m4$genre_idx,
    orientation  = b4$artist_tbl$orientation_num
  )
  m4_data$mean_lmc_z2 <- m4_data$mean_lmc_z^2

  message(sprintf("  M4 (Timeline):     %d songs, %d artists",
                  m4_data$N, m4_data$N_artist))
} else {
  message("  M4 (Timeline):     SKIPPED — no timeline data")
}


# =============================================================================
# 3. Compile and sample all models
# =============================================================================
message("\n══ 3. Compiling models ══")

stan_track    <- cmdstan_model(file.path(BAYES_DIR, "model_track.stan"))
stan_segment  <- if (has_segments) cmdstan_model(file.path(BAYES_DIR, "model_segment.stan"))
stan_timeline <- if (has_timeline) cmdstan_model(file.path(BAYES_DIR, "model_timeline.stan"))

# ── Shared sampling settings ──────────────────────────────────────────────
sample_args <- list(
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 2000,
  iter_sampling   = 2000,
  adapt_delta     = 0.95,
  max_treedepth   = 12,
  seed            = 42,
  show_messages   = FALSE,
  refresh         = 500
)

run_model <- function(stan_mod, data, label) {
  message(sprintf("\n── Sampling %s ──", label))
  args <- c(list(object = stan_mod, data = data,
                 output_dir = OUTPUT_DIR), sample_args)
  fit  <- do.call(stan_mod$sample, args[-1])
  fit$save_object(file.path(OUTPUT_DIR, paste0("fit_", label, ".rds")))
  message(sprintf("  %s sampling complete.", label))
  fit
}

fit_m1 <- run_model(stan_track,    m1_data, "M1_mulan_track")
fit_m2 <- run_model(stan_track,    m2_data, "M2_clap_track")
fit_m3 <- if (has_segments) run_model(stan_segment,  m3_data, "M3_segment")
fit_m4 <- if (has_timeline) run_model(stan_timeline, m4_data, "M4_timeline")


# =============================================================================
# 4. Convergence diagnostics for all models
# =============================================================================
message("\n══ 4. Convergence diagnostics ══")

diagnose_fit <- function(fit, label, params) {
  diag <- fit$summary(variables = params) %>%
    select(variable, mean, median, sd, q5, q95, rhat, ess_bulk, ess_tail) %>%
    mutate(across(where(is.numeric), ~round(., 4)),
           model = label)

  n_div  <- sum(fit$diagnostic_summary()$num_divergent)
  bad_r  <- sum(diag$rhat > 1.01, na.rm = TRUE)
  low_e  <- sum(diag$ess_bulk < 400, na.rm = TRUE)

  message(sprintf("  %-20s | Divergences: %d | R-hat>1.01: %d | ESS<400: %d",
                  label, n_div, bad_r, low_e))
  diag
}

# Parameter lists for each model
track_params   <- c("mu_global", "beta_lmc", "beta_lmc2", "beta_age",
                     "gamma_intercept", "gamma_lmc",
                     "sigma_artist", "sigma_genre", "phi")
segment_params <- c("mu_global", "beta_chorus", "beta_verse", "beta_sd",
                     "beta_age", "gamma_intercept", "gamma_chorus",
                     "sigma_artist", "sigma_genre", "phi",
                     "chorus_vs_verse")
timeline_params <- c("mu_global", "beta_lmc", "beta_lmc2",
                      "beta_slope", "beta_curve", "beta_change", "beta_sd",
                      "beta_age", "gamma_intercept", "gamma_lmc",
                      "sigma_artist[1]", "sigma_artist[2]",
                      "sigma_genre", "phi", "Rho_artist[1,2]")

all_diag <- bind_rows(
  diagnose_fit(fit_m1, "M1: MuLan Track",   track_params),
  diagnose_fit(fit_m2, "M2: CLAP Track",     track_params),
  if (has_segments) diagnose_fit(fit_m3, "M3: Segment",  segment_params),
  if (has_timeline) diagnose_fit(fit_m4, "M4: Timeline", timeline_params)
)

write_csv(all_diag, file.path(OUTPUT_DIR, "all_convergence.csv"))


# =============================================================================
# 5. Traceplots
# =============================================================================
message("\n══ 5. Traceplots ══")

plot_traces <- function(fit, params, label, filename) {
  p <- mcmc_trace(fit$draws(format = "array"), pars = params,
                  facet_args = list(ncol = 2)) +
    scale_color_brewer(palette = "Set1") +
    labs(title = paste0("Traceplots — ", label))
  print(p)
  savefig(filename, w = 12, h = max(4, length(params) * 1.5))
}

plot_traces(fit_m1,
            c("beta_lmc", "beta_lmc2", "beta_age", "gamma_lmc"),
            "M1: MuLan Track", "traces_M1.pdf")
plot_traces(fit_m2,
            c("beta_lmc", "beta_lmc2", "beta_age", "gamma_lmc"),
            "M2: CLAP Track", "traces_M2.pdf")
if (has_segments)
  plot_traces(fit_m3,
              c("beta_chorus", "beta_verse", "beta_sd", "gamma_chorus"),
              "M3: Segment", "traces_M3.pdf")
if (has_timeline)
  plot_traces(fit_m4,
              c("beta_lmc", "beta_slope", "beta_curve",
                "beta_change", "beta_sd", "gamma_lmc"),
              "M4: Timeline", "traces_M4.pdf")


# =============================================================================
# 6. Posterior summaries — all models
# =============================================================================
message("\n══ 6. Posterior summaries ══")

extract_posteriors <- function(fit, params, label) {
  draws <- fit$draws(format = "df")
  map_dfr(params, function(p) {
    x <- draws[[p]]
    if (is.null(x)) return(NULL)
    tibble(
      model     = label,
      parameter = p,
      mean      = mean(x),
      median    = median(x),
      sd        = sd(x),
      q5        = quantile(x, 0.05),
      q95       = quantile(x, 0.95),
      p_pos     = mean(x > 0),
      p_neg     = mean(x < 0)
    )
  }) %>% mutate(across(where(is.numeric), ~round(., 4)))
}

all_posteriors <- bind_rows(
  extract_posteriors(fit_m1, track_params,    "M1: MuLan Track"),
  extract_posteriors(fit_m2, track_params,    "M2: CLAP Track"),
  if (has_segments) extract_posteriors(fit_m3, segment_params,  "M3: Segment"),
  if (has_timeline) extract_posteriors(fit_m4, timeline_params, "M4: Timeline")
)

write_csv(all_posteriors, file.path(OUTPUT_DIR, "all_posteriors.csv"))


# ── Key comparison: LMC main effect across models ────────────────────────────
lmc_coefs <- all_posteriors %>%
  filter(parameter %in% c("beta_lmc", "beta_chorus"))

message("\nLMC main effect comparison:")
print(lmc_coefs %>% select(model, parameter, median, q5, q95, p_pos))


# =============================================================================
# 7. Effect visualisations
# =============================================================================
message("\n══ 7. Effect plots ══")

# ── 7a. LMC coefficient forest plot across all models ─────────────────────────
coef_compare <- bind_rows(
  fit_m1$summary("beta_lmc") %>% mutate(model = "M1: MuLan Track",  param = "LMC (mean)"),
  fit_m2$summary("beta_lmc") %>% mutate(model = "M2: CLAP Track",   param = "LMC (mean)"),
  if (has_segments) fit_m3$summary("beta_chorus") %>%
    mutate(model = "M3: Segment", param = "LMC (chorus)"),
  if (has_timeline) fit_m4$summary("beta_lmc") %>%
    mutate(model = "M4: Timeline", param = "LMC (mean)")
)

p_forest_models <- coef_compare %>%
  ggplot(aes(x = median, y = reorder(model, median),
             xmin = q5, xmax = q95)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(height = 0.3, linewidth = 1, color = "#2A9D8F") +
  geom_point(size = 4, color = "#2A9D8F") +
  labs(
    title    = "LMC Effect on Popularity Across Models",
    subtitle = "Posterior median + 90% CI (logit scale)",
    x = "LMC coefficient", y = NULL
  )
print(p_forest_models)
savefig("cross_model_lmc_forest.pdf", w = 9, h = 5)


# ── 7b. M3-specific: Chorus vs. Verse comparison ─────────────────────────────
if (has_segments) {
  m3_draws <- fit_m3$draws(format = "df")

  p_chorus_verse <- tibble(
    `Chorus LMC`        = m3_draws$beta_chorus,
    `Verse LMC`         = m3_draws$beta_verse,
    `Chorus − Verse`    = m3_draws$chorus_vs_verse
  ) %>%
    pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
    ggplot(aes(x = value, y = parameter, fill = after_stat(x > 0))) +
    stat_halfeye(.width = c(0.8, 0.95), point_interval = median_hdi) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey30") +
    scale_fill_manual(values = c("FALSE" = "#E76F51", "TRUE" = "#2A9D8F"),
                      guide = "none") +
    labs(
      title    = "M3: Chorus vs. Verse LMC Effects",
      subtitle = "Positive 'Chorus − Verse' means chorus congruence matters more",
      x = "Coefficient (logit scale)", y = NULL
    )
  print(p_chorus_verse)
  savefig("M3_chorus_vs_verse.pdf", w = 9, h = 5)
}


# ── 7c. M4-specific: Trajectory feature posteriors ────────────────────────────
if (has_timeline) {
  m4_labels <- c(
    "beta_lmc"    = "Mean LMC",
    "beta_lmc2"   = "Mean LMC²",
    "beta_slope"  = "Slope (trend)",
    "beta_curve"  = "Curvature (arc)",
    "beta_change" = "Early→Late",
    "beta_sd"     = "Consistency (−SD)"
  )

  p_m4_post <- fit_m4$draws(format = "df") %>%
    select(all_of(names(m4_labels))) %>%
    pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
    mutate(parameter = recode(parameter, !!!m4_labels)) %>%
    ggplot(aes(x = value, y = parameter, fill = after_stat(x > 0))) +
    stat_halfeye(.width = c(0.8, 0.95), point_interval = median_hdi) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey30") +
    scale_fill_manual(values = c("FALSE" = "#E76F51", "TRUE" = "#2A9D8F"),
                      guide = "none") +
    labs(
      title    = "M4: Trajectory Feature Posteriors",
      subtitle = "Which aspects of the LMC arc predict popularity?",
      x = "Coefficient (logit scale)", y = NULL
    )
  print(p_m4_post)
  savefig("M4_trajectory_posteriors.pdf", w = 10, h = 7)
}


# ── 7d. Marginal LMC effects in popularity points ────────────────────────────
marginal_compare <- bind_rows(
  tibble(model = "M1: MuLan Track",
         me    = apply(fit_m1$draws("marginal_lmc", format = "matrix"), 2, mean)),
  tibble(model = "M2: CLAP Track",
         me    = apply(fit_m2$draws("marginal_lmc", format = "matrix"), 2, mean)),
  if (has_segments) tibble(
    model = "M3: Segment (chorus)",
    me    = apply(fit_m3$draws("marginal_chorus", format = "matrix"), 2, mean)),
  if (has_timeline) tibble(
    model = "M4: Timeline",
    me    = apply(fit_m4$draws("marginal_lmc", format = "matrix"), 2, mean))
)

p_marginal <- marginal_compare %>%
  ggplot(aes(x = me, fill = model)) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_brewer(palette = "Set2", name = "Model") +
  labs(
    title    = "Marginal Effect of +1 SD LMC on Popularity (All Models)",
    subtitle = "Distribution across songs. Units = popularity points (0–100)",
    x = "Marginal effect (popularity points)", y = "Density"
  )
print(p_marginal)
savefig("cross_model_marginal_effects.pdf", w = 10, h = 6)


# =============================================================================
# 8. Posterior predictive checks
# =============================================================================
message("\n══ 8. Posterior predictive checks ══")

ppc_density_plot <- function(fit, y_obs, label, filename) {
  y_rep <- fit$draws("y_rep", format = "matrix")
  p <- ppc_dens_overlay(y_obs, y_rep[sample(nrow(y_rep), 100), ]) +
    labs(title = paste0("PPC: ", label), x = "Popularity (scaled)")
  print(p)
  savefig(filename, w = 9, h = 5)
}

ppc_density_plot(fit_m1, df_m1$pop_scaled, "M1: MuLan Track", "ppc_M1.pdf")
ppc_density_plot(fit_m2, df_m2$pop_scaled, "M2: CLAP Track",  "ppc_M2.pdf")
if (has_segments)
  ppc_density_plot(fit_m3, df_m3$pop_scaled, "M3: Segment",  "ppc_M3.pdf")
if (has_timeline)
  ppc_density_plot(fit_m4, df_m4$pop_scaled, "M4: Timeline", "ppc_M4.pdf")


# =============================================================================
# 9. LOO-CV comparison
# =============================================================================
message("\n══ 9. LOO-CV model comparison ══")

compute_loo <- function(fit, label) {
  ll  <- fit$draws("log_lik", format = "matrix")
  res <- loo(ll, cores = 4)
  n_bad_k <- sum(res$pointwise[, "influence_pareto_k"] > 0.7)
  message(sprintf("  %-20s | ELPD: %7.1f ± %.1f | p_eff: %.1f | Pareto-k>0.7: %d",
                  label,
                  res$estimates["elpd_loo", "Estimate"],
                  res$estimates["elpd_loo", "SE"],
                  res$estimates["p_loo", "Estimate"],
                  n_bad_k))
  res
}

loo_m1 <- compute_loo(fit_m1, "M1: MuLan Track")
loo_m2 <- compute_loo(fit_m2, "M2: CLAP Track")
loo_m3 <- if (has_segments) compute_loo(fit_m3, "M3: Segment")
loo_m4 <- if (has_timeline) compute_loo(fit_m4, "M4: Timeline")

# ── Pairwise comparisons ──────────────────────────────────────────────────
# NOTE: loo_compare requires models fit on the same data.
# M1 and M2 may have different N (not all songs have both MuLan and CLAP).
# M3/M4 will almost certainly have fewer songs.
# We compare directly only when N matches; otherwise report ELPDs side by side.

message("\n── LOO summary table ──")

loo_summary <- tibble(
  Model  = c("M1: MuLan Track", "M2: CLAP Track",
             if (has_segments) "M3: Segment",
             if (has_timeline) "M4: Timeline"),
  N      = c(m1_data$N, m2_data$N,
             if (has_segments) m3_data$N,
             if (has_timeline) m4_data$N),
  ELPD   = c(loo_m1$estimates["elpd_loo", "Estimate"],
             loo_m2$estimates["elpd_loo", "Estimate"],
             if (has_segments) loo_m3$estimates["elpd_loo", "Estimate"],
             if (has_timeline) loo_m4$estimates["elpd_loo", "Estimate"]),
  SE     = c(loo_m1$estimates["elpd_loo", "SE"],
             loo_m2$estimates["elpd_loo", "SE"],
             if (has_segments) loo_m3$estimates["elpd_loo", "SE"],
             if (has_timeline) loo_m4$estimates["elpd_loo", "SE"]),
  p_eff  = c(loo_m1$estimates["p_loo", "Estimate"],
             loo_m2$estimates["p_loo", "Estimate"],
             if (has_segments) loo_m3$estimates["p_loo", "Estimate"],
             if (has_timeline) loo_m4$estimates["p_loo", "Estimate"]),
  LOOIC  = c(loo_m1$estimates["looic", "Estimate"],
             loo_m2$estimates["looic", "Estimate"],
             if (has_segments) loo_m3$estimates["looic", "Estimate"],
             if (has_timeline) loo_m4$estimates["looic", "Estimate"])
) %>%
  mutate(across(where(is.numeric), ~round(., 2))) %>%
  # Per-song ELPD for apples-to-apples comparison when N differs
  mutate(ELPD_per_song = round(ELPD / N, 4)) %>%
  arrange(desc(ELPD_per_song))

print(loo_summary)
write_csv(loo_summary, file.path(OUTPUT_DIR, "loo_comparison.csv"))
savetable(loo_summary, "loo_comparison.txt")

# ── Direct pairwise comparison if M1 and M2 have the same N ──────────────
if (m1_data$N == m2_data$N) {
  message("\nM1 vs M2 (same N — direct comparison):")
  print(loo_compare(loo_m1, loo_m2))
}


# ── LOO comparison visualisation ──────────────────────────────────────────
p_loo <- loo_summary %>%
  ggplot(aes(x = ELPD_per_song, y = reorder(Model, ELPD_per_song))) +
  geom_errorbarh(aes(xmin = ELPD_per_song - 2 * SE / N,
                     xmax = ELPD_per_song + 2 * SE / N),
                 height = 0.3, linewidth = 1, color = "#2A9D8F") +
  geom_point(size = 4, color = "#2A9D8F") +
  labs(
    title    = "LOO-CV Model Comparison",
    subtitle = "ELPD per song (higher = better predictive performance). ±2 SE bars.",
    x = "ELPD per song", y = NULL
  )
print(p_loo)
savefig("loo_comparison.pdf", w = 9, h = 5)


# =============================================================================
# 10. Orientation moderation comparison
# =============================================================================
message("\n══ 10. Orientation moderation comparison ══")

mod_draws <- bind_rows(
  tibble(model = "M1: MuLan Track",
         gamma = fit_m1$draws(format = "df")$gamma_lmc),
  tibble(model = "M2: CLAP Track",
         gamma = fit_m2$draws(format = "df")$gamma_lmc),
  if (has_segments) tibble(
    model = "M3: Segment",
    gamma = fit_m3$draws(format = "df")$gamma_chorus),
  if (has_timeline) tibble(
    model = "M4: Timeline",
    gamma = fit_m4$draws(format = "df")$gamma_lmc)
)

p_mod_compare <- mod_draws %>%
  ggplot(aes(x = gamma, fill = model)) +
  geom_density(alpha = 0.45) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_brewer(palette = "Set2", name = "Model") +
  labs(
    title    = "Orientation Moderation of LMC Effect Across Models",
    subtitle = "Positive = narrative artists show stronger LMC-popularity link",
    x = "γ_lmc (logit scale)", y = "Density"
  )
print(p_mod_compare)
savefig("cross_model_orientation_moderation.pdf", w = 10, h = 6)


# =============================================================================
# 11. Summary report
# =============================================================================
message("\n══ 11. Summary ══")

# Assemble key findings
key_findings <- all_posteriors %>%
  filter(parameter %in% c("beta_lmc", "beta_chorus",
                           "gamma_lmc", "gamma_chorus")) %>%
  select(model, parameter, median, q5, q95, p_pos) %>%
  mutate(
    sig_90   = if_else(q5 > 0 | q95 < 0, "Yes", "No"),
    credible = case_when(
      p_pos > 0.95 ~ "Strong positive",
      p_pos > 0.90 ~ "Weak positive",
      p_neg > 0.95 ~ "Strong negative",
      p_neg > 0.90 ~ "Weak negative",
      TRUE         ~ "Inconclusive"
    )
  )

message("\nKey findings:")
print(key_findings)
savetable(key_findings, "key_findings.txt")

# Best model
best <- loo_summary %>% slice(1)
message(sprintf("\nBest model (by ELPD/song): %s (ELPD/song = %.4f)",
                best$Model, best$ELPD_per_song))

message("\n", strrep("═", 60))
message("All models complete.")
message(sprintf("  Outputs: %s", OUTPUT_DIR))
message(strrep("═", 60))

