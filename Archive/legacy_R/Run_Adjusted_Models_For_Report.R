# =============================================================================
# run_all_models_v2.R
#
# Runs M1–M4 using the v2 Stan models (mood controls, no quadratic).
# Saves fit objects to bayes/output/ for the QMD report to load.
#
# Run this first, then render report.qmd.
# =============================================================================

library(cmdstanr)
library(tidyverse)
library(loo)
library(here)

BAYES_DIR  <- here::here("bayes")
OUTPUT_DIR <- file.path(BAYES_DIR, "output")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(42)
zsc               <- function(x) as.numeric(scale(x))
boundary_adjust   <- function(y, N) (y * (N - 1) + 0.5) / N


# =============================================================================
# 1. Load data
# =============================================================================
message("\n── 1. Loading data ──")

master <- read_csv(here::here("results", "master_results.csv"),
                   show_col_types = FALSE) %>%
  mutate(
    release_year  = as.numeric(substr(release_date, 1, 4)),
    song_age      = 2025 - release_year,
    genre_cluster = case_when(
      genre %in% c("hip-hop")                              ~ "Hip-Hop",
      genre %in% c("folk", "folk-rock", "country")        ~ "Folk/Country",
      genre %in% c("pop")                                  ~ "Pop",
      genre %in% c("electronic", "psychedelic-electronic") ~ "Electronic",
      TRUE                                                 ~ "Other"
    )
  ) |> 
  select(-c("tempo", "danceability", "energy", "valence", "acousticness",
            "instrumentalness", "liveness", "speechiness", "loudness"))

# Mood features
mood_path <- here::here("results", "essentia_mood.csv")
if (!file.exists(mood_path)) stop("essentia_mood.csv not found. Run 11_extract_mood.py first.")

mood <- read_csv(mood_path, show_col_types = FALSE) %>%
  select(song_id, mood_happy, mood_sad, mood_relaxed,
         mood_aggressive, mood_party, danceability, voice_instrumental)

master <- master %>% left_join(mood, by = "song_id")

# Segment data
seg_path <- here::here("results", "segment_analysis", "segment_summary.csv")
has_seg  <- file.exists(seg_path)
if (has_seg) {
  seg <- read_csv(seg_path, show_col_types = FALSE)
} else {
  message("  No segment data — M3 will be skipped.")
}

# Timeline / trajectory data
tl_path <- here::here("results", "lyric_timeline", "lyric_timeline.csv")
has_tl  <- file.exists(tl_path)

if (has_tl) {
  tl <- read_csv(tl_path, show_col_types = FALSE) %>%
    filter(match_confidence >= 0.40)
  
  extract_traj <- function(d) {
    pos <- d$position_pct; lmc <- d$lmc; n <- nrow(d)
    na_row <- data.frame(mean_lmc_tl = NA, sd_lmc_tl = NA,
                         lmc_slope = NA, lmc_curve = NA,
                         lmc_change = NA, lmc_smooth_sd = NA, n_lines = n)
    if (n < 5) return(na_row)
    tryCatch({
      lo <- loess(lmc ~ pos, span = 0.5, degree = 1,
                  control = loess.control(surface = "direct"))
      g  <- seq(0, 100, by = 2)
      sm <- predict(lo, data.frame(pos = g))
      sm[is.na(sm)] <- mean(lmc)
      data.frame(
        mean_lmc_tl   = mean(lmc),
        sd_lmc_tl     = sd(lmc),
        lmc_slope     = as.numeric(coef(lm(sm ~ g))["g"]),
        lmc_curve     = as.numeric(coef(lm(sm ~ g + I(g^2)))["I(g^2)"]),
        lmc_change    = mean(sm[g > 50]) - mean(sm[g <= 50]),
        lmc_smooth_sd = sd(sm),
        n_lines       = n
      )
    }, error = function(e) na_row)
  }
  
  traj <- tl %>%
    group_by(song_id) %>%
    group_modify(~ extract_traj(.x)) %>%
    ungroup()
} else {
  message("  No timeline data — M4 will be skipped.")
}

# =============================================================================
# 2. Helper: build Stan data list from a song-level data frame
# =============================================================================

build_stan_data_base <- function(df) {
  # Encode group indices
  artist_levels <- sort(unique(df$artist_code))
  genre_levels  <- sort(unique(df$genre_cluster))
  
  df <- df %>%
    mutate(
      artist_idx = as.integer(factor(artist_code,   levels = artist_levels)),
      genre_idx  = as.integer(factor(genre_cluster, levels = genre_levels)),
      pop_scaled = boundary_adjust(popularity / 100, n())
    )
  
  artist_tbl <- df %>%
    distinct(artist_code, artist_idx, orientation) %>%
    arrange(artist_idx) %>%
    mutate(orientation_num = if_else(orientation == "narrative", 1, 0))
  
  list(
    df          = df,
    artist_tbl  = artist_tbl,
    artist_levels = artist_levels,
    genre_levels  = genre_levels
  )
}

# Mood control helper — z-score with fallback to 0 if all NA
zsc_safe <- function(x) {
  if (all(is.na(x))) return(rep(0, length(x)))
  as.numeric(scale(x))
}

mood_standata <- function(df) {
  list(
    song_age_z      = zsc_safe(df$song_age),
    mood_happy_z    = zsc_safe(df$mood_happy),
    mood_sad_z      = zsc_safe(df$mood_sad),
    mood_relaxed_z  = zsc_safe(df$mood_relaxed),
    mood_aggressive_z = zsc_safe(df$mood_aggressive),
    mood_party_z    = zsc_safe(df$mood_party),
    danceability_z  = zsc_safe(df$danceability),
    voice_instr_z   = zsc_safe(df$voice_instrumental)
  )
}


# =============================================================================
# 3. Compile Stan models
# =============================================================================
message("\n── 2. Compiling Stan models ──")

stan_track    <- cmdstan_model(file.path(BAYES_DIR, "model_track_v2.stan"))
stan_segment  <- if (has_seg) cmdstan_model(file.path(BAYES_DIR, "model_segment_v2.stan"))
stan_timeline <- if (has_tl)  cmdstan_model(file.path(BAYES_DIR, "model_timeline_v2.stan"))
message("  Compilation complete.")


# =============================================================================
# 4. Shared sampling settings
# =============================================================================

SAMPLE_ARGS <- list(
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

run_and_save <- function(stan_mod, data_list, label) {
  message(sprintf("\n── Sampling %s ──", label))
  fit <- do.call(
    stan_mod$sample,
    c(list(data = data_list, output_dir = OUTPUT_DIR), SAMPLE_ARGS)
  )
  
  # Convergence check
  diag  <- fit$diagnostic_summary()
  n_div <- sum(diag$num_divergent)
  summ  <- fit$summary() %>%
    filter(!is.na(rhat)) %>%
    summarise(max_rhat = max(rhat), min_ess = min(ess_bulk))
  
  message(sprintf("  %s done | Divergences: %d | Max R-hat: %.4f | Min ESS: %.0f",
                  label, n_div, summ$max_rhat, summ$min_ess))
  if (n_div > 0)
    message(sprintf("  ⚠ %d divergent transitions — consider raising adapt_delta", n_div))
  if (summ$max_rhat > 1.01)
    message(sprintf("  ⚠ R-hat > 1.01 for some parameters — check traceplots"))
  
  out_path <- file.path(OUTPUT_DIR, paste0("fit_", label, ".rds"))
  fit$save_object(out_path)
  message(sprintf("  Saved → %s", out_path))
  fit
}


# =============================================================================
# 5. M1 — MuLan track-level
# =============================================================================
message("\n══ M1: MuLan Track ══")

df_m1 <- master %>%
  filter(!is.na(popularity), !is.na(lmc_mulan), !is.na(song_age),
         !is.na(mood_happy)) %>%
  drop_na(artist_code, genre_cluster)

b1 <- build_stan_data_base(df_m1)

m1_standata <- c(
  list(
    N        = nrow(b1$df),
    N_artist = max(b1$df$artist_idx),
    N_genre  = max(b1$df$genre_idx),
    y        = b1$df$pop_scaled,
    lmc_z    = zsc(b1$df$lmc_mulan),
    artist_id   = b1$df$artist_idx,
    genre_id    = b1$df$genre_idx,
    orientation = b1$artist_tbl$orientation_num
  ),
  mood_standata(b1$df)
)

message(sprintf("  Sample: %d songs, %d artists, %d genres",
                m1_standata$N, m1_standata$N_artist, m1_standata$N_genre))

fit_m1 <- run_and_save(stan_track, m1_standata, "M1_mulan_track")


# =============================================================================
# 6. M2 — CLAP track-level
# =============================================================================
message("\n══ M2: CLAP Track ══")

df_m2 <- master %>%
  filter(!is.na(popularity), !is.na(lmc_clap), !is.na(song_age),
         !is.na(mood_happy)) %>%
  drop_na(artist_code, genre_cluster)

b2 <- build_stan_data_base(df_m2)

m2_standata <- c(
  list(
    N        = nrow(b2$df),
    N_artist = max(b2$df$artist_idx),
    N_genre  = max(b2$df$genre_idx),
    y        = b2$df$pop_scaled,
    lmc_z    = zsc(b2$df$lmc_clap),
    artist_id   = b2$df$artist_idx,
    genre_id    = b2$df$genre_idx,
    orientation = b2$artist_tbl$orientation_num
  ),
  mood_standata(b2$df)
)

message(sprintf("  Sample: %d songs, %d artists, %d genres",
                m2_standata$N, m2_standata$N_artist, m2_standata$N_genre))

fit_m2 <- run_and_save(stan_track, m2_standata, "M2_clap_track")


# =============================================================================
# 7. M3 — Segment-level
# =============================================================================

if (has_seg) {
  message("\n══ M3: Segment ══")
  
  df_m3 <- master %>%
    inner_join(seg, by = "song_id") %>%
    filter(!is.na(popularity), !is.na(song_age), !is.na(mood_happy)) %>%
    mutate(
      seg_chorus = if_else(is.na(mean_lmc_chorus), mean_lmc_all, mean_lmc_chorus),
      seg_verse  = if_else(is.na(mean_lmc_verse),  mean_lmc_all, mean_lmc_verse),
      seg_sd     = if_else(is.na(sd_lmc),          0,            sd_lmc)
    ) %>%
    drop_na(artist_code, genre_cluster, seg_chorus, seg_verse, seg_sd)
  
  b3 <- build_stan_data_base(df_m3)
  
  m3_standata <- c(
    list(
      N        = nrow(b3$df),
      N_artist = max(b3$df$artist_idx),
      N_genre  = max(b3$df$genre_idx),
      y            = b3$df$pop_scaled,
      chorus_lmc_z = zsc(b3$df$seg_chorus),
      verse_lmc_z  = zsc(b3$df$seg_verse),
      lmc_sd_z     = zsc(b3$df$seg_sd),
      artist_id    = b3$df$artist_idx,
      genre_id     = b3$df$genre_idx,
      orientation  = b3$artist_tbl$orientation_num
    ),
    mood_standata(b3$df)
  )
  
  message(sprintf("  Sample: %d songs, %d artists, %d genres",
                  m3_standata$N, m3_standata$N_artist, m3_standata$N_genre))
  
  fit_m3 <- run_and_save(stan_segment, m3_standata, "M3_segment")
}


# =============================================================================
# 8. M4 — Timeline-level
# =============================================================================

if (has_tl) {
  message("\n══ M4: Timeline ══")
  
  df_m4 <- master %>%
    inner_join(traj, by = "song_id") %>%
    filter(!is.na(popularity), !is.na(song_age), !is.na(mood_happy)) %>%
    drop_na(artist_code, genre_cluster,
            mean_lmc_tl, lmc_slope, lmc_curve, lmc_change, sd_lmc_tl)
  
  b4 <- build_stan_data_base(df_m4)
  
  m4_standata <- c(
    list(
      N        = nrow(b4$df),
      N_artist = max(b4$df$artist_idx),
      N_genre  = max(b4$df$genre_idx),
      y            = b4$df$pop_scaled,
      mean_lmc_z   = zsc(b4$df$mean_lmc_tl),
      lmc_slope_z  = zsc(b4$df$lmc_slope),
      lmc_curve_z  = zsc(b4$df$lmc_curve),
      lmc_change_z = zsc(b4$df$lmc_change),
      lmc_sd_z     = zsc(b4$df$sd_lmc_tl),
      artist_id    = b4$df$artist_idx,
      genre_id     = b4$df$genre_idx,
      orientation  = b4$artist_tbl$orientation_num,
      artist_genre = b4$df %>%
        distinct(artist_idx, genre_idx) %>%
        arrange(artist_idx) %>%
        pull(genre_idx)
    ),
    mood_standata(b4$df)
  )
  
  message(sprintf("  Sample: %d songs, %d artists, %d genres",
                  m4_standata$N, m4_standata$N_artist, m4_standata$N_genre))
  
  fit_m4 <- run_and_save(stan_timeline, m4_standata, "M4_timeline")
}


# =============================================================================
# 9. Quick LOO comparison
# =============================================================================
message("\n══ LOO-CV comparison ══")

fits <- list(
  "M1: MuLan Track" = fit_m1,
  "M2: CLAP Track"  = fit_m2
)
if (has_seg && exists("fit_m3")) fits[["M3: Segment"]]  <- fit_m3
if (has_tl  && exists("fit_m4")) fits[["M4: Timeline"]] <- fit_m4

Ns <- c(m1_standata$N, m2_standata$N,
        if (has_seg) m3_standata$N,
        if (has_tl)  m4_standata$N)

loo_res <- map(fits, function(fit) {
  ll <- fit$draws("log_lik", format = "matrix")
  loo(ll, cores = 4)
})

loo_table <- map2_dfr(loo_res, names(loo_res), function(res, label) {
  tibble(
    Model         = label,
    N             = Ns[which(names(fits) == label)],
    ELPD          = round(res$estimates["elpd_loo", "Estimate"], 2),
    SE            = round(res$estimates["elpd_loo", "SE"],       2),
    ELPD_per_song = round(res$estimates["elpd_loo", "Estimate"] /
                            Ns[which(names(fits) == label)], 4),
    p_eff         = round(res$estimates["p_loo",    "Estimate"], 1),
    k_bad         = sum(res$pointwise[, "influence_pareto_k"] > 0.7)
  )
}) %>% arrange(desc(ELPD_per_song))

message("\nLOO summary (higher ELPD/song = better):")
print(loo_table)

write_csv(loo_table, file.path(OUTPUT_DIR, "loo_comparison.csv"))

message("\n", strrep("═", 60))
message(sprintf("All fits saved to: %s", OUTPUT_DIR))
message("Next step: quarto render bayes/report.qmd")
message(strrep("═", 60))

fit_m1$summary()



genre_int_params <- paste0("[", 1:n_genre, "]")
genre_int_post <- fit_m3$summary(variables = contains()) %>%
  mutate(genre = genre_levels[as.integer(str_extract(variable, "\\d+"))]) %>%
  filter(!is.na(genre))

p_genre_int <- ggplot(genre_int_post, aes(x = median, y = reorder(genre, median),
                                          color = genre)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = q5, xmax = q95),
                 height = 0.3, linewidth = 1.2) +
  geom_point(size = 4) +
  scale_color_manual(values = GENRE_COLS, guide = "none") +
  labs(title = "Genre Intercepts (Baseline Popularity M3)",
       subtitle = "Deviation from grand mean. 90% CI.",
       x = "α_genre (logit scale)", y = NULL)

p_genre_int

ggplot(mod_draws, aes(x = gamma, fill = model)) +
  geom_density(alpha = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_brewer(palette = "Set2", name = "Model") +
  labs(title = "Orientation × LMC Moderation Across Models",
       subtitle = "Positive γ = narrative artists show stronger LMC-popularity link",
       x = "γ (logit scale)", y = "Density")

mcmc_trace(fit_m4$draws(format = "array"),
           pars = c("beta_lmc", "beta_slope", "beta_curve",
                    "beta_change", "beta_sd", "gamma_lmc"),
           facet_args = list(ncol = 2)) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "M4: Timeline — Key Parameters")

mcmc_trace(fit_m3$draws(format = "array"),
           pars = c("beta_chorus", "beta_verse", "beta_sd",
                    "gamma_chorus", "sigma_artist", "sigma_genre"),
           facet_args = list(ncol = 2)) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "M3: Segment — Key Parameters")

# M1 traceplots
mcmc_trace(fit_m1$draws(format = "array"),
           pars = c("beta_lmc", "gamma_lmc", "sigma_artist", "sigma_genre"),
           facet_args = list(ncol = 2)) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "M1: MuLan Track — Key Parameters")

# M2 traceplots
mcmc_trace(fit_m2$draws(format = "array"),
           pars = c("beta_lmc", "gamma_lmc", "sigma_artist", "sigma_genre"),
           facet_args = list(ncol = 2)) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "M2: MuLan Track — Key Parameters")

mood_params <- c("beta_happy", "beta_sad", "beta_relaxed",
                 "beta_aggressive", "beta_party", "beta_dance", "beta_voice")

mood_labels <- c(
  "beta_happy"      = "Happy",
  "beta_sad"        = "Sad",
  "beta_relaxed"    = "Relaxed",
  "beta_aggressive" = "Aggressive",
  "beta_party"      = "Party",
  "beta_dance"      = "Danceability",
  "beta_voice"      = "Voice / Instrumental"
)

mood_post <- fit_m3$summary(variables = mood_params) %>%
  mutate(label = recode(variable, !!!mood_labels))

p_mood <- ggplot(mood_post, aes(x = median, y = reorder(label, median),
                                color = median > 0)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = q5, xmax = q95),
                 height = 0.3, linewidth = 1.2) +
  geom_point(size = 4) +
  scale_color_manual(values = c("TRUE" = "#2A9D8F", "FALSE" = "#E76F51"),
                     guide = "none") +
  labs(title = "Mood Tag Coefficients (M3)",
       subtitle = "Effect on popularity. 90% CI. Teal = positive, orange = negative.",
       x = "β (logit scale)", y = NULL)

master |> 
  select(genre, popularity) |> 
  mutate(genre = case_when(
    genre %in% c("hip-hop")                              ~ "Hip-Hop",
    genre %in% c("folk", "folk-rock", "country")        ~ "Folk/Country",
    genre %in% c("pop")                                  ~ "Pop",
    genre %in% c("electronic", "psychedelic-electronic") ~ "Electronic",
    TRUE                                                 ~ "Other"
  )) |>
  ggplot(mapping = aes(x = popularity, fill = genre)) +
  geom_density(alpha = .5) +
  labs(title = "Outcome Distribution by Genre",
       subtitle = "Spotify popularity, scaled 0-100",
       x = "Spotify Popularity", y = "Density")

