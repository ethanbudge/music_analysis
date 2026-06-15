# =============================================================================
# run_models.R — Fit the LMC → popularity Bayesian models (cmdstanr).
#
# Adapts the v2 Stan family to the broad LRCLIB sample:
#   • genre is the primary grouping factor; artist random intercepts shrink,
#   • orientation is a recovered song-level moderator,
#   • a sample_corpus() helper draws a random N of songs so you can fit on a
#     flexible subset of whatever has been gathered so far.
#
# Source the file to get the helper functions, or run it top-to-bottom to fit
# the default battery (track models for each LMC measure + segment + timeline)
# for a chosen embedding model and save fits to stan/output/.
#
# Usage:
#   Rscript stan/run_models.R              # defaults: model=mulan, N=all
#   Rscript stan/run_models.R clap 500 1   # embedding=clap, N=500, seed=1
# =============================================================================

suppressPackageStartupMessages({
  library(cmdstanr); library(tidyverse); library(loo); library(here)
})

STAN_DIR   <- here::here("stan")
OUTPUT_DIR <- file.path(STAN_DIR, "output")
RESULTS    <- here::here("results")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

zsc <- function(x) as.numeric(scale(x))
boundary_adjust <- function(y, n) (y * (n - 1) + 0.5) / n   # Smithson & Verkuilen

CONTEXT_WINDOWS <- c("exact", "buf1", "buf5", "buf10")

# ─── Data loading ────────────────────────────────────────────────────────────
load_master <- function() {
  read_csv(file.path(RESULTS, "master_results.csv"), show_col_types = FALSE)
}

# Randomly draw N complete-case songs from the gathered corpus.
# Pass N = NULL (default) to use everything available.
sample_corpus <- function(df, N = NULL, seed = 42, required = NULL) {
  if (!is.null(required)) df <- df %>% drop_na(any_of(required))
  set.seed(seed)
  if (is.null(N) || N >= nrow(df)) return(df)
  df[sample(nrow(df), N), , drop = FALSE]
}

orient_num <- function(o) dplyr::recode(o, narrative = 1, production = 0, .default = 0.5)

# Common Stan data block (controls + grouping) shared by every model.
base_stan_data <- function(df) {
  df <- df %>%
    mutate(
      artist_id = as.integer(factor(artist)),
      genre_id  = as.integer(factor(genre)),
      y         = boundary_adjust(spotify_popularity / 100, nrow(df))
    )
  list(
    N = nrow(df), N_artist = max(df$artist_id), N_genre = max(df$genre_id),
    y = df$y,
    song_age_z        = zsc(replace_na(df$song_age_years, median(df$song_age_years, na.rm = TRUE))),
    mood_happy_z      = zsc(df$mood_happy),
    mood_sad_z        = zsc(df$mood_sad),
    mood_relaxed_z    = zsc(df$mood_relaxed),
    mood_aggressive_z = zsc(df$mood_aggressive),
    mood_party_z      = zsc(df$mood_party),
    danceability_z    = zsc(df$danceability),
    voice_instr_z     = zsc(df$voice_instrumental),
    artist_id = df$artist_id, genre_id = df$genre_id,
    orientation = orient_num(df$orientation)
  )
}

# ─── Per-model data prep ───────────────────────────────────────────────────────
track_data <- function(df, lmc_col) {
  stopifnot(lmc_col %in% names(df))
  df <- df %>% drop_na(spotify_popularity, all_of(lmc_col),
                       mood_happy, mood_sad, mood_relaxed,
                       mood_aggressive, mood_party, danceability, voice_instrumental)
  c(base_stan_data(df), list(lmc_z = zsc(df[[lmc_col]])))
}

segment_data <- function(df, model) {
  cc <- paste0(model, "_seg_chorus"); nc <- paste0(model, "_seg_nonchorus")
  df <- df %>% drop_na(spotify_popularity, all_of(c(cc, nc)),
                       mood_happy, mood_sad, mood_relaxed,
                       mood_aggressive, mood_party, danceability, voice_instrumental)
  c(base_stan_data(df),
    list(chorus_lmc_z = zsc(df[[cc]]), nonchorus_lmc_z = zsc(df[[nc]])))
}

# Derive per-song timeline shape features from the long line-level table.
timeline_features <- function(model, window = "buf5") {
  lines <- read_csv(file.path(RESULTS, "lmc_lines.csv"), show_col_types = FALSE) %>%
    filter(model == !!model, window == !!window) %>%
    arrange(track_id, line_idx)
  lines %>% group_by(track_id) %>% summarise(
    mean_lmc   = mean(lmc, na.rm = TRUE),
    lmc_sd     = sd(lmc, na.rm = TRUE),
    lmc_slope  = if (n() > 2) coef(lm(lmc ~ position_pct))[2] else 0,
    lmc_curve  = if (n() > 3) coef(lm(lmc ~ poly(position_pct, 2)))[3] else 0,
    lmc_change = mean(abs(diff(lmc))),
    .groups = "drop"
  )
}

timeline_data <- function(df, model, window = "buf5") {
  feats <- timeline_features(model, window)
  df <- df %>% inner_join(feats, by = "track_id") %>%
    drop_na(spotify_popularity, mean_lmc, lmc_sd, lmc_slope, lmc_curve, lmc_change,
            mood_happy, mood_sad, mood_relaxed, mood_aggressive,
            mood_party, danceability, voice_instrumental)
  c(base_stan_data(df),
    list(mean_lmc_z = zsc(df$mean_lmc), lmc_slope_z = zsc(df$lmc_slope),
         lmc_curve_z = zsc(df$lmc_curve), lmc_change_z = zsc(df$lmc_change),
         lmc_sd_z = zsc(df$lmc_sd)))
}

# ─── Fitting ───────────────────────────────────────────────────────────────────
.compiled <- new.env()
get_model <- function(name) {
  if (is.null(.compiled[[name]]))
    .compiled[[name]] <- cmdstan_model(file.path(STAN_DIR, paste0(name, ".stan")))
  .compiled[[name]]
}

fit_one <- function(stan_name, data, tag, chains = 4, iter = 1500, ...) {
  message(sprintf("── fitting %s  [%s]  (N=%d, genres=%d)", stan_name, tag,
                  data$N, data$N_genre))
  fit <- get_model(stan_name)$sample(
    data = data, chains = chains, parallel_chains = chains,
    iter_warmup = iter, iter_sampling = iter, refresh = 0,
    adapt_delta = 0.95, max_treedepth = 12, seed = 42, ...)
  fit$save_object(file.path(OUTPUT_DIR, paste0(tag, ".rds")))
  fit
}

# ─── Default battery ───────────────────────────────────────────────────────────
run_all <- function(model = "mulan", N = NULL, seed = 42) {
  master <- load_master()
  df <- sample_corpus(master, N = N, seed = seed,
                      required = c("spotify_popularity", "genre"))
  message(sprintf("Corpus for fitting: %d songs (embedding model: %s)", nrow(df), model))

  measures <- c("song", paste0("line_", CONTEXT_WINDOWS))
  loos <- list()
  for (m in measures) {
    col <- paste0(model, "_", m)
    if (!col %in% names(df)) next
    fit <- fit_one("model_track", track_data(df, col), paste0("track_", model, "_", m))
    loos[[m]] <- fit$loo()
  }
  fit_one("model_segment",  segment_data(df, model),  paste0("segment_", model))
  fit_one("model_timeline", timeline_data(df, model), paste0("timeline_", model, "_buf5"))

  if (length(loos) > 1) {
    cmp <- loo::loo_compare(loos)
    print(cmp)
    saveRDS(cmp, file.path(OUTPUT_DIR, paste0("loo_compare_", model, ".rds")))
  }
  message("Done. Fits saved to ", OUTPUT_DIR)
}

# Run the battery when invoked as a script (not when sourced).
if (sys.nframe() == 0) {
  a <- commandArgs(trailingOnly = TRUE)
  run_all(model = if (length(a) >= 1) a[1] else "mulan",
          N     = if (length(a) >= 2) as.integer(a[2]) else NULL,
          seed  = if (length(a) >= 3) as.integer(a[3]) else 42)
}
