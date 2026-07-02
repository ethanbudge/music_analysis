# =============================================================================
# report_helpers.R — Shared logic for analysis/lmc_report.qmd (v4 model family).
#
# Turns the saved cmdstanr fits in stan/output/ into labelled, human-readable
# summaries: covariates by NAME (including the GENERIC control block — mood / MERT
# PCs / song age, whichever was toggled in), genre-varying effects by GENRE NAME,
# and the functional coefficient β(t) for the trajectory models. Kept separate
# from the .qmd so the wrangling can be unit-run with Rscript.
#
# Handles tags: track_<emb>_<measure>, segment_<emb>, curve_<emb>,
#               segcurve_<emb>, linecurve_<emb>.
# =============================================================================

suppressPackageStartupMessages({
  library(cmdstanr); library(posterior); library(loo)
  library(tidyverse); library(here)
})

STAN_DIR <- here::here("stan")
OUTPUT   <- file.path(STAN_DIR, "output")

# run_models.R only auto-runs when called as a script, so sourcing just loads the
# data builders + build_corpus() we need to recover labels / observed y.
source(file.path(STAN_DIR, "run_models.R"))

# ─── Labels ──────────────────────────────────────────────────────────────────
LABELS <- c(
  mu_global = "Global intercept (logit)",
  beta_lmc = "LMC — overall slope", gamma_lmc = "Orientation × LMC slope",
  gamma_intercept = "Orientation → intercept (narrative vs. production)",
  sigma_genre = "SD of genre intercepts", sigma_lmc_genre = "SD of genre LMC slopes",
  phi_intercept = "Precision: log-φ intercept", sigma_phi_genre = "SD of genre log-φ",
  beta_chorus = "Chorus LMC — overall slope", beta_nonchorus = "Non-chorus LMC slope",
  gamma_chorus = "Orientation × chorus LMC", sigma_chorus_genre = "SD of genre chorus slopes",
  chorus_vs_nonchorus = "Chorus − non-chorus (Δ slope)",
  sigma_b = "β(t) smoothing scale", sigma_bc = "β_chorus(t) smoothing scale",
  sigma_bnc = "β_nonchorus(t) smoothing scale",
  theta = "Latent congruence → popularity", sigma_f = "f(t) smoothing scale",
  sigma_line = "Line-level residual SD", tau_u = "SD of latent song congruence")

# Pretty-print a control-column name (mert_pc03 → "MERT PC3", mood_happy → "Mood: happy").
pretty_control <- function(p) dplyr::case_when(
  grepl("^mert_pc", p) ~ paste0("MERT PC", suppressWarnings(as.integer(str_extract(p, "\\d+")))),
  p == "song_age"      ~ "Song age",
  grepl("^mood_", p)   ~ paste0("Mood: ", sub("mood_", "", p)),
  p == "danceability"  ~ "Danceability",
  p == "voice_instrumental" ~ "Voice vs. instrumental",
  TRUE ~ p)
pretty_label <- function(p) ifelse(p %in% names(LABELS), LABELS[p], p)

# Scalar "effect" parameters per family (the controls are added separately, by name).
FAM_EFFECTS <- list(
  track    = c("beta_lmc", "gamma_lmc", "gamma_intercept"),
  segment  = c("beta_chorus", "beta_nonchorus", "chorus_vs_nonchorus", "gamma_chorus", "gamma_intercept"),
  curve    = c("gamma_intercept"),
  segcurve = c("gamma_intercept"),
  polycurve = c("gamma_intercept"),
  hiercurve = c("gamma_intercept"),
  linecurve = c("theta", "gamma_intercept"))
# Hierarchical scale / smoothing hyperparameters per family.
FAM_SCALES <- list(
  track    = c("sigma_genre", "sigma_lmc_genre", "phi_intercept", "sigma_phi_genre"),
  segment  = c("sigma_genre", "sigma_chorus_genre", "phi_intercept", "sigma_phi_genre"),
  curve    = c("sigma_b", "sigma_genre", "phi_intercept", "sigma_phi_genre"),
  segcurve = c("sigma_bc", "sigma_bnc", "sigma_genre", "phi_intercept", "sigma_phi_genre"),
  polycurve = c("sigma_genre", "phi_intercept", "sigma_phi_genre"),
  hiercurve = c("sigma_b", "tau_delta", "sigma_genre", "phi_intercept", "sigma_phi_genre"),
  linecurve = c("sigma_f", "sigma_line", "tau_u", "sigma_genre", "phi_intercept", "sigma_phi_genre"))
# Genre-varying slope vector per family (NA where the model has none).
SLOPE_GENRE <- c(track = "beta_lmc_genre", segment = "beta_chorus_genre",
                 curve = NA, segcurve = NA, polycurve = NA, hiercurve = NA, linecurve = NA)
SLOPE_GENRE_LABEL <- c(track = "LMC slope", segment = "Chorus LMC slope")
# Functional-coefficient (population) generated quantities per family.
FUNC_OUTPUTS <- list(curve = "beta_t",
                     segcurve = c("beta_chorus_t", "beta_nonchorus_t"),
                     polycurve = "beta_t", hiercurve = "beta_t",
                     linecurve = "f_t")
FUNC_LABEL <- c(beta_t = "β(t)", beta_chorus_t = "β chorus(t)",
                beta_nonchorus_t = "β non-chorus(t)", f_t = "f(t) population trajectory")

MEASURE_LABEL <- c(song = "Song-wide", line_exact = "Line ±0 s", line_buf1 = "Line ±1 s",
                   line_buf5 = "Line ±5 s", line_buf10 = "Line ±10 s")

fit_family <- function(tag) dplyr::case_when(
  startsWith(tag, "track_")     ~ "track",
  startsWith(tag, "segment_")   ~ "segment",
  startsWith(tag, "segcurve_")  ~ "segcurve",
  startsWith(tag, "polycurve_") ~ "polycurve",
  startsWith(tag, "hiercurve_") ~ "hiercurve",
  startsWith(tag, "curve_")     ~ "curve",
  startsWith(tag, "linecurve_") ~ "linecurve",
  TRUE ~ NA_character_)

tag_embedding <- function(tag)
  str_match(tag, "^(?:track|segment|segcurve|polycurve|hiercurve|curve|linecurve)_([a-z]+)")[, 2]
tag_measure <- function(tag) if (startsWith(tag, "track_")) sub("^track_[a-z]+_", "", tag) else NA_character_

# Rebuild a fit's observed y (for PPC). Mirrors the v4 builders; controls toggle is
# read from the saved label sidecar so the corpus matches the fit.
rebuild_data <- function(tag, df, controls) {
  fam <- fit_family(tag); emb <- tag_embedding(tag)
  switch(fam,
    track    = track_data(df, paste0(emb, "_", tag_measure(tag)), controls),
    segment  = segment_data(df, emb, controls),
    curve    = curve_data(df, emb, controls = controls),
    segcurve = segment_curve_data(df, emb, controls = controls),
    polycurve = poly_curve_data(df, emb, controls = controls),
    hiercurve = curve_data(df, emb, controls = controls),
    linecurve = line_curve_data(df, emb, controls = controls),
    NULL)
}

genre_levels <- function(embeddings = "mulan", N = NULL, seed = 42, controls = "mert")
  levels(factor(build_corpus(embeddings, N = N, seed = seed, controls = controls)$df$genre))

# ─── Per-fit extraction ──────────────────────────────────────────────────────
extract_fit <- function(tag, corpus_df = NULL, genre_names = NULL, n_draws = 800) {
  path <- file.path(OUTPUT, paste0(tag, ".rds"))
  if (!file.exists(path)) return(NULL)
  fit <- readRDS(path)
  fam <- fit_family(tag)
  mp  <- fit$metadata()$model_params
  scalars <- mp[!grepl("\\[", mp)]

  lab <- NULL
  labpath <- file.path(OUTPUT, paste0(tag, ".labels.rds"))
  if (file.exists(labpath)) lab <- readRDS(labpath)
  if (!is.null(lab$genre_levels)) genre_names <- lab$genre_levels
  ctrl_names <- lab$control_names
  q05 <- ~quantile(.x, 0.05, names = FALSE); q95 <- ~quantile(.x, 0.95, names = FALSE)
  p_pos <- ~mean(.x > 0)

  # --- population scalar effects + scales -----------------------------------
  eff <- intersect(FAM_EFFECTS[[fam]], scalars)
  scl <- intersect(FAM_SCALES[[fam]], scalars)
  summ <- fit$summary(c(eff, scl), median = median, q05 = q05, q95 = q95,
                      p_pos = p_pos, rhat = rhat, ess_bulk = ess_bulk) %>%
    rename(param = variable) %>%
    mutate(group = if_else(param %in% eff, "effect", "scale"),
           label = pretty_label(param))

  # --- generic control block (beta_ctrl[k] → control name) ------------------
  ctrl_tbl <- NULL
  if (any(grepl("^beta_ctrl\\[", mp))) {
    ctrl_tbl <- fit$summary("beta_ctrl", median = median, q05 = q05, q95 = q95,
                            p_pos = p_pos, rhat = rhat, ess_bulk = ess_bulk) %>%
      rename(param = variable) %>%
      mutate(idx = as.integer(str_extract(param, "(?<=\\[)\\d+(?=\\])")),
             label = if (!is.null(ctrl_names)) pretty_control(ctrl_names[idx]) else param,
             group = "control")
  }

  # --- genre-varying slope, by name -----------------------------------------
  gtag <- SLOPE_GENRE[[fam]]; gsumm <- NULL; gslope_long <- NULL
  if (!is.na(gtag) && any(grepl(paste0("^", gtag, "\\["), mp))) {
    gsumm <- fit$summary(c("alpha_genre", gtag), median = median, q05 = q05, q95 = q95) %>%
      rename(param = variable) %>%
      mutate(idx = as.integer(str_extract(param, "(?<=\\[)\\d+(?=\\])")),
             base = str_remove(param, "\\[\\d+\\]"), genre = genre_names[idx])
    gdd <- fit$draws(gtag, format = "draws_df")
    if (nrow(gdd) > n_draws) gdd <- gdd[sample(nrow(gdd), n_draws), ]
    gslope_long <- gdd %>% select(starts_with(gtag)) %>%
      pivot_longer(everything(), names_to = "param", values_to = "value") %>%
      mutate(idx = as.integer(str_extract(param, "(?<=\\[)\\d+(?=\\])")), genre = genre_names[idx])
  }

  # --- functional coefficient β(t), summarised over the saved grid ----------
  func <- NULL
  fouts <- FUNC_OUTPUTS[[fam]]
  if (!is.null(fouts) && !is.null(lab$func_grid)) {
    func <- map_dfr(fouts, function(v) {
      s <- fit$summary(v, median = median, q05 = q05, q95 = q95)
      tibble(which = FUNC_LABEL[[v]], t = lab$func_grid,
             median = s$median, lo = s$q05, hi = s$q95)
    })
  }

  # --- per-genre coefficient function β_g(t) (genre-varying curve models) ----
  func_genre <- NULL
  if ("beta_t_genre" %in% fit$metadata()$stan_variables && !is.null(lab$func_grid)) {
    G <- length(lab$func_grid)
    s <- fit$summary("beta_t_genre", median = median, q05 = q05, q95 = q95) %>%
      mutate(gi = as.integer(str_match(variable, "\\[(\\d+),")[, 2]),   # grid index (row)
             gj = as.integer(str_match(variable, ",(\\d+)\\]")[, 2]))    # genre index (col)
    func_genre <- tibble(t = lab$func_grid[s$gi], genre = genre_names[s$gj],
                         median = s$median, lo = s$q05, hi = s$q95)
  }

  # --- diagnostics + LOO ----------------------------------------------------
  ds <- fit$diagnostic_summary(quiet = TRUE)
  diag <- tibble(tag = tag, family = fam, embedding = tag_embedding(tag),
                 measure = tag_measure(tag), n_div = sum(ds$num_divergent),
                 ebfmi_min = min(ds$ebfmi), rhat_max = max(summ$rhat, na.rm = TRUE),
                 ess_bulk_min = min(summ$ess_bulk, na.rm = TRUE))
  loo_obj <- tryCatch(fit$loo(), error = function(e) NULL)

  # --- PPC ------------------------------------------------------------------
  ppc <- NULL
  if (!is.null(corpus_df) && !is.null(lab$controls)) {
    sd_data <- tryCatch(rebuild_data(tag, corpus_df, lab$controls), error = function(e) NULL)
    yrep <- tryCatch(fit$draws("y_rep", format = "draws_matrix"), error = function(e) NULL)
    if (!is.null(sd_data) && !is.null(yrep) && length(sd_data$y) == ncol(yrep)) {
      idx <- sample(nrow(yrep), min(60, nrow(yrep)))
      ppc <- list(y = as.numeric(sd_data$y),
                  yrep = matrix(as.numeric(yrep), nrow = nrow(yrep))[idx, , drop = FALSE])
    }
  }

  rm(fit); gc(verbose = FALSE)
  list(tag = tag, family = fam, embedding = tag_embedding(tag), measure = tag_measure(tag),
       controls = lab$controls, summ = summ, ctrl_tbl = ctrl_tbl, gsumm = gsumm,
       gslope_long = gslope_long, func = func, func_genre = func_genre,
       diag = diag, loo = loo_obj, ppc = ppc)
}

# ─── Plot helpers ────────────────────────────────────────────────────────────
theme_report <- function() theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), plot.title.position = "plot")

# Forest plot of population effects + control block, by name.
plot_effects <- function(m, include_controls = TRUE) {
  d <- bind_rows(m$summ %>% filter(group == "effect", param != "mu_global"),
                 if (include_controls) m$ctrl_tbl else NULL) %>%
    mutate(label = factor(label, levels = rev(unique(label))),
           sig = p_pos > 0.95 | p_pos < 0.05)
  ggplot(d, aes(median, label, colour = sig)) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey60") +
    geom_pointrange(aes(xmin = q05, xmax = q95)) +
    scale_colour_manual(values = c(`TRUE` = "#1f6feb", `FALSE` = "grey55"), guide = "none") +
    labs(x = "Posterior median and 90% credible interval (logit scale)", y = NULL) +
    theme_report()
}

# Genre-varying slope, by genre name.
plot_genre_slope <- function(m) {
  if (is.null(m$gsumm)) return(NULL)
  gt <- SLOPE_GENRE[[m$family]]
  ord <- m$gsumm %>% filter(base == gt) %>% arrange(median) %>% pull(genre)
  d <- m$gslope_long %>% mutate(genre = factor(genre, levels = ord))
  s <- m$gsumm %>% filter(base == gt) %>% mutate(genre = factor(genre, levels = ord))
  ggplot() +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey60") +
    geom_violin(data = d, aes(value, genre), fill = "#1f6feb", alpha = 0.2, colour = NA, scale = "width") +
    geom_pointrange(data = s, aes(median, genre, xmin = q05, xmax = q95)) +
    labs(x = paste0(SLOPE_GENRE_LABEL[[m$family]], " (logit scale)"), y = NULL) + theme_report()
}

# Functional coefficient β(t) over normalised song position, with 90% band.
plot_beta_t <- function(m) {
  if (is.null(m$func)) return(NULL)
  ggplot(m$func, aes(t, median)) +
    geom_hline(yintercept = 0, linetype = 2, colour = "grey60") +
    geom_ribbon(aes(ymin = lo, ymax = hi, fill = which), alpha = 0.2, colour = NA) +
    geom_line(aes(colour = which), linewidth = 0.9) +
    scale_x_continuous(labels = scales::percent) +
    labs(x = "Song position", y = "Coefficient on popularity (logit)",
         colour = NULL, fill = NULL) +
    theme_report() + theme(legend.position = "bottom")
}

# Per-genre congruence-effect trajectory β_g(t): one small panel per genre, with
# 90% band. Shows how the effect of congruence-at-position-t on popularity differs
# across genres (genre-varying curvature models: polycurve / hiercurve).
plot_beta_t_genre <- function(m, ncol = 3) {
  if (is.null(m$func_genre)) return(NULL)
  ggplot(m$func_genre, aes(t, median)) +
    geom_hline(yintercept = 0, linetype = 2, colour = "grey60") +
    geom_ribbon(aes(ymin = lo, ymax = hi), fill = "#1f6feb", alpha = 0.2) +
    geom_line(colour = "#1f6feb", linewidth = 0.8) +
    facet_wrap(~ genre, ncol = ncol) +
    scale_x_continuous(labels = scales::percent) +
    labs(x = "Song position", y = "Congruence effect on popularity (logit)") +
    theme_report()
}

# ─── Cross-embedding comparison (MuLan vs CLAP) ──────────────────────────────
EMB_COLOURS <- c(mulan = "#1f6feb", clap = "#d1242f")

# One scalar param's posterior summary across a list of fit objects (per embedding).
effect_across <- function(mods, which_param) {
  purrr::map_dfr(Filter(Negate(is.null), mods), function(m)
    m$summ %>% dplyr::filter(param == which_param) %>%
      transmute(embedding = m$embedding, label = label,
                median, q05, q95, p_pos))
}
fmt_ci <- function(med, lo, hi, p) sprintf("%+.3f [%+.3f, %+.3f] %.0f%%", med, lo, hi, 100 * p)

# β(t) (population functional models) across embeddings → long df with `embedding`.
func_across <- function(mods) purrr::map_dfr(Filter(Negate(is.null), mods),
  function(m) if (!is.null(m$func)) m$func %>% mutate(embedding = m$embedding))
func_genre_across <- function(mods) purrr::map_dfr(Filter(Negate(is.null), mods),
  function(m) if (!is.null(m$func_genre)) m$func_genre %>% mutate(embedding = m$embedding))

# Overlay β(t) coloured by embedding (facets over chorus/non-chorus if present).
plot_beta_t_compare <- function(mods) {
  d <- func_across(mods); if (!nrow(d)) return(NULL)
  ggplot(d, aes(t, median, colour = embedding, fill = embedding)) +
    geom_hline(yintercept = 0, linetype = 2, colour = "grey60") +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.9) +
    facet_wrap(~ which) +
    scale_colour_manual(values = EMB_COLOURS, aesthetics = c("colour", "fill")) +
    scale_x_continuous(labels = scales::percent) +
    labs(x = "Song position", y = "Congruence effect on popularity (logit)",
         colour = NULL, fill = NULL) +
    theme_report() + theme(legend.position = "bottom")
}

# Per-genre β_g(t) overlaid by embedding (one panel per genre).
plot_beta_t_genre_compare <- function(mods, ncol = 3) {
  d <- func_genre_across(mods); if (!nrow(d)) return(NULL)
  ggplot(d, aes(t, median, colour = embedding, fill = embedding)) +
    geom_hline(yintercept = 0, linetype = 2, colour = "grey60") +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.12, colour = NA) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~ genre, ncol = ncol) +
    scale_colour_manual(values = EMB_COLOURS, aesthetics = c("colour", "fill")) +
    scale_x_continuous(labels = scales::percent) +
    labs(x = "Song position", y = "Congruence effect (logit)", colour = NULL, fill = NULL) +
    theme_report() + theme(legend.position = "bottom")
}
