# =============================================================================
# analysis.R — Musical Congruence: Full Statistical Analysis
# =============================================================================
#
# Corresponds to Sections 4–6 of the paper.
#
# Structure
# ---------
#   0.  Setup & data loading
#   1.  Descriptive statistics & data quality
#   2.  LMC correlations across models (convergent validity)
#   3.  Main effect: LMC → popularity (H1)
#   4.  Mixed-effects models (artist + genre random effects)
#   5.  Moderation by orientation: narrative vs. production-forward (H2)
#   6.  Nonlinearity test (H3 — inverted-U, per Askin & Mauskapf 2017)
#   7.  Model comparison across embedding models
#   8.  Segment-level analysis (if data available)
#   9.  Robustness checks
#  10.  Visualizations
#
# Output files (all in analysis/output/)
# ----------------------------------------
#   tables/
#     01_descriptives.txt
#     02_lmc_correlations.txt
#     03_main_effects.txt
#     04_mixed_effects.txt
#     05_moderation.txt
#     06_nonlinearity.txt
#     07_model_comparison.txt
#   figures/
#     01_lmc_distributions.pdf
#     02_popularity_distribution.pdf
#     03_lmc_vs_popularity_mulan.pdf
#     04_lmc_vs_popularity_all_models.pdf
#     05_genre_boxplot.pdf
#     06_orientation_moderation.pdf
#     07_nonlinearity.pdf
#     08_model_comparison_coefs.pdf
#     09_correlation_matrix.pdf
#     10_segment_heatmap.pdf
#     11_pca_biplot.pdf
# =============================================================================

# ── 0. Setup ──────────────────────────────────────────────────────────────────

required_packages <- c(
  "tidyverse", "lme4", "lmerTest", "broom.mixed", "ggrepel",
  "patchwork",  "modelsummary", "performance", "ggcorrplot",
  "sjPlot", "scales", "corrplot", "RColorBrewer", "viridis",
  "effectsize", "marginaleffects"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# Output directories
out_dir    <- here::here("analysis", "output")
tab_dir    <- file.path(out_dir, "tables")
fig_dir    <- file.path(out_dir, "figures")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ggplot theme
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

# Colour palette: genre clusters
GENRE_COLORS <- c(
  "hip-hop"               = "#E63946",
  "folk-rock"             = "#2A9D8F",
  "folk"                  = "#52B788",
  "country"               = "#F4A261",
  "pop"                   = "#457B9D",
  "electronic"            = "#9B2335",
  "psychedelic-electronic" = "#C77DFF"
)

ORIENTATION_COLORS <- c(
  "narrative"  = "#2A9D8F",
  "production" = "#E76F51"
)

save_fig <- function(name, w = 8, h = 6) {
  ggsave(file.path(fig_dir, name), width = w, height = h,
         device = "pdf", dpi = 300)
  message("  Figure saved: ", name)
}

save_table <- function(x, name) {
  sink(file.path(tab_dir, name))
  print(x)
  sink()
  message("  Table saved: ", name)
}

# ── 0.1  Load data ────────────────────────────────────────────────────────────

data_path <- here::here("results", "master_results.csv")
if (!file.exists(data_path)) {
  stop("master_results.csv not found. Run 09_combine_results.py first.")
}

raw <- read_csv(data_path, show_col_types = FALSE)
message(sprintf("Loaded %d songs × %d columns", nrow(raw), ncol(raw)))

# ── 0.2  Factor encoding ──────────────────────────────────────────────────────

df <- raw %>%
  mutate(
    artist_code  = factor(artist_code),
    genre        = factor(genre),
    orientation  = factor(orientation, levels = c("production", "narrative")),
    genre_cluster = case_when(
      genre %in% c("hip-hop")                       ~ "Hip-Hop",
      genre %in% c("folk", "folk-rock", "country")  ~ "Folk/Country",
      genre %in% c("pop")                            ~ "Pop",
      genre %in% c("electronic", "psychedelic-electronic") ~ "Electronic",
      TRUE                                           ~ "Other"
    ) %>% factor(levels = c("Hip-Hop", "Folk/Country", "Pop", "Electronic")),

    # Centred LMC (for quadratic terms — avoids collinearity)
    lmc_mulan_c      = scale(lmc_mulan,     center = TRUE, scale = TRUE)[,1],
    lmc_clap_c       = scale(lmc_clap,      center = TRUE, scale = TRUE)[,1],
    lmc_mert_sbert_c = scale(lmc_mert_sbert,center = TRUE, scale = TRUE)[,1],

    # Squared terms for nonlinearity tests
    lmc_mulan_c2      = lmc_mulan_c^2,
    lmc_clap_c2       = lmc_clap_c^2,
    lmc_mert_sbert_c2 = lmc_mert_sbert_c^2,

    # Track duration in minutes
    duration_min = duration_ms / 60000,

    # Logarithm of popularity for robustness check
    log_popularity = log1p(popularity),

    # Song age (rough proxy: release year; use as control)
    release_year = as.numeric(substr(release_date, 1, 4)),
    song_age     = 2025 - release_year,
  )

# Analysis sample: songs with popularity AND at least one LMC score
df_full <- df %>%
  filter(!is.na(popularity), !is.na(lmc_mulan))

message(sprintf("Analysis sample: %d songs, %d artists, %d genre clusters",
                nrow(df_full), n_distinct(df_full$artist_code),
                n_distinct(df_full$genre_cluster)))


# =============================================================================
# 1.  DESCRIPTIVE STATISTICS
# =============================================================================
message("\n── 1. Descriptive Statistics ──")

desc_vars <- c("popularity", "lmc_mulan", "lmc_clap", "lmc_mert_sbert",
               "tempo", "danceability", "energy", "valence",
               "acousticness", "speechiness", "duration_min")

desc_table <- df_full %>%
  select(all_of(desc_vars)) %>%
  summarise(across(everything(), list(
    n    = ~sum(!is.na(.)),
    mean = ~mean(., na.rm = TRUE),
    sd   = ~sd(., na.rm = TRUE),
    min  = ~min(., na.rm = TRUE),
    p25  = ~quantile(., 0.25, na.rm = TRUE),
    med  = ~median(., na.rm = TRUE),
    p75  = ~quantile(., 0.75, na.rm = TRUE),
    max  = ~max(., na.rm = TRUE)
  ), .names = "{.col}__{.fn}")) %>%
  pivot_longer(everything(),
               names_to   = c("variable", "stat"),
               names_sep  = "__") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(across(where(is.numeric), ~round(., 3)))

print(desc_table, n = nrow(desc_table))
save_table(desc_table, "01_descriptives.txt")

# Per-genre descriptives
genre_desc <- df_full %>%
  group_by(genre_cluster) %>%
  summarise(
    n            = n(),
    pop_mean     = round(mean(popularity, na.rm = TRUE), 1),
    pop_sd       = round(sd(popularity, na.rm = TRUE), 1),
    lmc_mulan_m  = round(mean(lmc_mulan, na.rm = TRUE), 4),
    lmc_mulan_sd = round(sd(lmc_mulan, na.rm = TRUE), 4),
    .groups = "drop"
  )
print(genre_desc)


# =============================================================================
# 2.  LMC CROSS-MODEL CORRELATIONS  (convergent validity)
# =============================================================================
message("\n── 2. LMC Cross-Model Correlations ──")

lmc_corr_df <- df_full %>%
  select(lmc_mulan, lmc_clap, lmc_mert_sbert) %>%
  drop_na()

if (nrow(lmc_corr_df) >= 5) {
  corr_mat <- cor(lmc_corr_df, use = "pairwise.complete.obs", method = "pearson")
  corr_test <- cor.test(lmc_corr_df$lmc_mulan, lmc_corr_df$lmc_clap)

  message(sprintf("MuLan vs. CLAP:       r = %.3f (p = %.4f)",
                  corr_mat["lmc_mulan","lmc_clap"], corr_test$p.value))
  save_table(round(corr_mat, 3), "02_lmc_correlations.txt")
}


# =============================================================================
# 3.  MAIN EFFECT: LMC → POPULARITY  (OLS, naive, no controls)
# =============================================================================
message("\n── 3. Main Effects (OLS) ──")

# Model 3.1: MuLan LMC only
m3_mulan <- lm(popularity ~ lmc_mulan_c, data = df_full)

# Model 3.2: CLAP LMC only
m3_clap  <- lm(popularity ~ lmc_clap_c, data = df_full)

# Model 3.3: MERT+SBERT LMC only
m3_sbert <- lm(popularity ~ lmc_mert_sbert_c, data = df_full)

# Model 3.4: All three LMC measures simultaneously
m3_all   <- lm(popularity ~ lmc_mulan_c + lmc_clap_c + lmc_mert_sbert_c, data = df_full)

main_effect_table <- modelsummary(
  list(
    "MuLan"      = m3_mulan,
    "CLAP"       = m3_clap,
    "MERT+SBERT" = m3_sbert,
    "Combined"   = m3_all
  ),
  stars     = TRUE,
  gof_map   = c("nobs", "r.squared", "adj.r.squared", "AIC"),
  output    = "data.frame"
)
print(main_effect_table)
save_table(main_effect_table, "03_main_effects.txt")


# =============================================================================
# 4.  MIXED-EFFECTS MODELS  (artist + genre random effects)
# =============================================================================
message("\n── 4. Mixed-Effects Models ──")

# Model 4.1: Random intercepts for artist (songs nested in artists)
m4_ri_artist <- lmer(
  popularity ~ lmc_mulan_c + (1 | artist_code),
  data   = df_full,
  REML   = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

# Model 4.2: Random intercepts for artist + genre_cluster
m4_ri_both <- lmer(
  popularity ~ lmc_mulan_c + (1 | artist_code) + (1 | genre_cluster),
  data   = df_full,
  REML   = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

# Model 4.3: Audio feature controls + random artist intercepts
m4_controls <- lmer(
  popularity ~ lmc_mulan_c +
    energy + danceability + valence + acousticness +
    speechiness + loudness + tempo + duration_min +
    (1 | artist_code),
  data   = df_full,
  REML   = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

# Model 4.4: Full controls + genre fixed effects
m4_full <- lmer(
  popularity ~ lmc_mulan_c + genre_cluster +
    energy + danceability + valence + acousticness +
    speechiness + loudness + tempo + duration_min +
    (1 | artist_code),
  data   = df_full,
  REML   = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

# Likelihood ratio test: does LMC improve fit?
m4_null <- lmer(
  popularity ~ energy + danceability + valence + acousticness +
    speechiness + loudness + tempo + duration_min +
    (1 | artist_code),
  data   = df_full,
  REML   = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)
lrt_4 <- anova(m4_null, m4_controls)
message("LRT — LMC added to controls:")
print(lrt_4)

me_table <- modelsummary(
  list(
    "Artist RE"        = m4_ri_artist,
    "Artist+Genre RE"  = m4_ri_both,
    "+ Controls"       = m4_controls,
    "Full"             = m4_full
  ),
  stars   = TRUE,
  gof_map = c("nobs", "AIC", "BIC"),
  output  = "data.frame"
)
save_table(me_table, "04_mixed_effects.txt")
print(summary(m4_full))


# =============================================================================
# 5.  MODERATION BY ORIENTATION  (narrative vs. production-forward)
# =============================================================================
message("\n── 5. Moderation: Narrative vs. Production-Forward ──")

# Model 5.1: Orientation main effect
m5_main <- lmer(
  popularity ~ lmc_mulan_c + orientation + (1 | artist_code),
  data = df_full, REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

# Model 5.2: Interaction (key moderation model)
m5_interact <- lmer(
  popularity ~ lmc_mulan_c * orientation + (1 | artist_code),
  data = df_full, REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

# Model 5.3: Full controls + interaction
m5_full <- lmer(
  popularity ~ lmc_mulan_c * orientation +
    energy + danceability + valence + acousticness +
    speechiness + loudness + tempo + duration_min +
    (1 | artist_code),
  data = df_full, REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

lrt_5 <- anova(m5_main, m5_interact)
message("LRT — orientation interaction:")
print(lrt_5)

# Simple slopes: LMC effect within each orientation
df_narr <- df_full %>% filter(orientation == "narrative")
df_prod <- df_full %>% filter(orientation == "production")

m5_narr <- lmer(popularity ~ lmc_mulan_c + (1 | artist_code),
                data = df_narr, REML = FALSE,
                control = lmerControl(optimizer = "bobyqa"))
m5_prod <- lmer(popularity ~ lmc_mulan_c + (1 | artist_code),
                data = df_prod, REML = FALSE,
                control = lmerControl(optimizer = "bobyqa"))

message("Simple slope — narrative artists:")
print(summary(m5_narr)$coefficients["lmc_mulan_c",])
message("Simple slope — production artists:")
print(summary(m5_prod)$coefficients["lmc_mulan_c",])

mod_table <- modelsummary(
  list(
    "Main effect"   = m5_main,
    "Interaction"   = m5_interact,
    "Full + Inter." = m5_full
  ),
  stars   = TRUE,
  gof_map = c("nobs", "AIC"),
  output  = "data.frame"
)
save_table(mod_table, "05_moderation.txt")


# =============================================================================
# 6.  NONLINEARITY TEST  (inverted-U per Askin & Mauskapf 2017)
# =============================================================================
message("\n── 6. Nonlinearity (Quadratic LMC) ──")

# Model 6.1: Linear + quadratic MuLan LMC
m6_quad_mulan <- lmer(
  popularity ~ lmc_mulan_c + lmc_mulan_c2 + (1 | artist_code),
  data = df_full, REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

# Model 6.2: Quadratic + orientation moderation
m6_quad_mod <- lmer(
  popularity ~ lmc_mulan_c + lmc_mulan_c2 + orientation +
    lmc_mulan_c:orientation + lmc_mulan_c2:orientation +
    (1 | artist_code),
  data = df_full, REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

lrt_6 <- anova(m4_ri_artist, m6_quad_mulan)
message("LRT — quadratic term (MuLan):")
print(lrt_6)

# Extract peak of the inverted-U (if quadratic is negative)
coefs <- fixef(m6_quad_mulan)
b1 <- coefs["lmc_mulan_c"]
b2 <- coefs["lmc_mulan_c2"]
if (!is.na(b2) && b2 < 0) {
  peak_z   <- -b1 / (2 * b2)
  lmc_mu   <- mean(df_full$lmc_mulan, na.rm = TRUE)
  lmc_sd   <- sd(df_full$lmc_mulan,   na.rm = TRUE)
  peak_lmc <- peak_z * lmc_sd + lmc_mu
  message(sprintf("Inverted-U peak: LMC_mulan = %.4f (z = %.2f)", peak_lmc, peak_z))
}

nl_table <- modelsummary(
  list(
    "Linear"    = m4_ri_artist,
    "Quadratic" = m6_quad_mulan,
    "Quad+Mod"  = m6_quad_mod
  ),
  stars   = TRUE,
  gof_map = c("nobs", "AIC", "BIC"),
  output  = "data.frame"
)
save_table(nl_table, "06_nonlinearity.txt")


# =============================================================================
# 7.  MODEL COMPARISON ACROSS EMBEDDING MODELS
# =============================================================================
message("\n── 7. Cross-Model Comparison ──")

# Filter to songs with all three LMC scores
df_trio <- df_full %>%
  drop_na(lmc_mulan, lmc_clap, lmc_mert_sbert)

message(sprintf("Complete cases (all 3 models): %d songs", nrow(df_trio)))

if (nrow(df_trio) >= 20) {
  mc_mulan <- lmer(popularity ~ lmc_mulan_c     + (1 | artist_code),
                   data = df_trio, REML = FALSE,
                   control = lmerControl(optimizer = "bobyqa"))
  mc_clap  <- lmer(popularity ~ lmc_clap_c      + (1 | artist_code),
                   data = df_trio, REML = FALSE,
                   control = lmerControl(optimizer = "bobyqa"))
  mc_sbert <- lmer(popularity ~ lmc_mert_sbert_c + (1 | artist_code),
                   data = df_trio, REML = FALSE,
                   control = lmerControl(optimizer = "bobyqa"))

  # Cross-validated R² (leave-one-artist-out)
  loao_r2 <- function(model_formula, data) {
    artists <- unique(data$artist_code)
    preds   <- numeric(nrow(data))
    for (art in artists) {
      train <- data %>% filter(artist_code != art)
      test  <- data %>% filter(artist_code == art)
      if (nrow(train) < 5) next
      fit   <- lmer(model_formula, data = train, REML = FALSE,
                    control = lmerControl(optimizer = "bobyqa"))
      preds[data$artist_code == art] <- predict(fit, newdata = test, allow.new.levels = TRUE)
    }
    ss_res <- sum((data$popularity - preds)^2, na.rm = TRUE)
    ss_tot <- sum((data$popularity - mean(data$popularity, na.rm = TRUE))^2, na.rm = TRUE)
    1 - ss_res / ss_tot
  }

  message("Computing leave-one-artist-out CV R²…")
  cv_r2_mulan <- tryCatch(
    loao_r2(popularity ~ lmc_mulan_c + (1 | artist_code), df_trio), error = function(e) NA)
  cv_r2_clap  <- tryCatch(
    loao_r2(popularity ~ lmc_clap_c  + (1 | artist_code), df_trio), error = function(e) NA)
  cv_r2_sbert <- tryCatch(
    loao_r2(popularity ~ lmc_mert_sbert_c + (1 | artist_code), df_trio), error = function(e) NA)

  cv_table <- tibble(
    Model       = c("MuQ-MuLan", "LAION-CLAP-Music", "MERT+SBERT"),
    AIC         = c(AIC(mc_mulan), AIC(mc_clap), AIC(mc_sbert)),
    Beta        = c(fixef(mc_mulan)["lmc_mulan_c"],
                    fixef(mc_clap)["lmc_clap_c"],
                    fixef(mc_sbert)["lmc_mert_sbert_c"]),
    LOAO_CV_R2  = c(cv_r2_mulan, cv_r2_clap, cv_r2_sbert),
    Joint_Training = c("Yes", "Yes", "No (Baseline)")
  ) %>%
    mutate(across(where(is.numeric), ~round(., 4)))

  print(cv_table)
  save_table(cv_table, "07_model_comparison.txt")
}


# =============================================================================
# 8.  SEGMENT-LEVEL ANALYSIS  (if data available)
# =============================================================================
message("\n── 8. Segment-Level Analysis ──")

seg_path <- here::here("results", "segment_analysis", "segment_summary.csv")

if (file.exists(seg_path)) {
  seg <- read_csv(seg_path, show_col_types = FALSE)

  df_seg <- df_full %>%
    left_join(seg, by = "song_id") %>%
    filter(!is.na(seg_mean_lmc))

  message(sprintf("Segment data available for %d songs", nrow(df_seg)))

  # Does chorus LMC predict popularity better than mean track LMC?
  if (nrow(df_seg) >= 20 && !all(is.na(df_seg$seg_chorus_lmc))) {
    m8_track  <- lmer(popularity ~ scale(lmc_mulan)    + (1 | artist_code),
                      data = df_seg, REML = FALSE,
                      control = lmerControl(optimizer = "bobyqa"))
    m8_seg    <- lmer(popularity ~ scale(seg_mean_lmc)  + (1 | artist_code),
                      data = df_seg, REML = FALSE,
                      control = lmerControl(optimizer = "bobyqa"))
    m8_chorus <- lmer(popularity ~ scale(seg_chorus_lmc) + (1 | artist_code),
                      data = df_seg %>% filter(!is.na(seg_chorus_lmc)),
                      REML = FALSE,
                      control = lmerControl(optimizer = "bobyqa"))
    m8_both   <- lmer(popularity ~ scale(lmc_mulan) + scale(seg_chorus_lmc) +
                        (1 | artist_code),
                      data = df_seg %>% filter(!is.na(seg_chorus_lmc)),
                      REML = FALSE,
                      control = lmerControl(optimizer = "bobyqa"))

    seg_table <- modelsummary(
      list(
        "Track LMC"   = m8_track,
        "Seg Mean"    = m8_seg,
        "Chorus LMC"  = m8_chorus,
        "Both"        = m8_both
      ),
      stars = TRUE, gof_map = c("nobs", "AIC"), output = "data.frame"
    )
    save_table(seg_table, "08_segment_analysis.txt")
  }

  # Congruence consistency: does high SD predict lower popularity?
  if (nrow(df_seg) >= 20 && !all(is.na(df_seg$seg_sd_lmc))) {
    m8_sd <- lmer(popularity ~ scale(lmc_mulan) + scale(seg_sd_lmc) +
                    (1 | artist_code),
                  data = df_seg, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa"))
    message("Segment SD (congruence consistency) model:")
    print(summary(m8_sd)$coefficients)
  }

} else {
  message("No segment data found. Run 07_segment_analysis.py first.")
}


# =============================================================================
# 9.  ROBUSTNESS CHECKS
# =============================================================================
message("\n── 9. Robustness Checks ──")

# 9.1 Log-transformed popularity
m9_log <- lmer(log_popularity ~ lmc_mulan_c + (1 | artist_code),
               data = df_full, REML = FALSE,
               control = lmerControl(optimizer = "bobyqa"))
message("Log(popularity) model:")
print(summary(m9_log)$coefficients["lmc_mulan_c",])

# 9.2 Exclude explicit songs
df_clean <- df_full %>% filter(!explicit | is.na(explicit))
m9_noexpl <- lmer(popularity ~ lmc_mulan_c + (1 | artist_code),
                  data = df_clean, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa"))
message(sprintf("Excl. explicit (%d songs):", nrow(df_clean)))
print(summary(m9_noexpl)$coefficients["lmc_mulan_c",])

# 9.3 Song-age control
if (!all(is.na(df_full$song_age))) {
  m9_age <- lmer(popularity ~ lmc_mulan_c + song_age + (1 | artist_code),
                 data = df_full %>% filter(!is.na(song_age)),
                 REML = FALSE,
                 control = lmerControl(optimizer = "bobyqa"))
  message("Age-controlled model:")
  print(summary(m9_age)$coefficients["lmc_mulan_c",])
}

# 9.4 Correlation-based robustness: Spearman
spear <- cor(df_full$lmc_mulan, df_full$popularity,
             method = "spearman", use = "pairwise.complete.obs")
message(sprintf("Spearman rho (LMC_mulan ~ popularity): %.4f", spear))

rob_table <- modelsummary(
  list(
    "Main"         = m4_ri_artist,
    "Log(Y)"       = m9_log,
    "No Explicit"  = m9_noexpl
  ),
  stars = TRUE, gof_map = c("nobs", "AIC"), output = "data.frame"
)
save_table(rob_table, "09_robustness.txt")


# =============================================================================
# 10.  VISUALIZATIONS
# =============================================================================
message("\n── 10. Visualizations ──")

# ── Figure 1: LMC distributions by model ──────────────────────────────────────
p_lmc_dist <- df_full %>%
  select(song_id, genre_cluster,
         `MuQ-MuLan`       = lmc_mulan,
         `LAION-CLAP-Music` = lmc_clap,
         `MERT+SBERT`       = lmc_mert_sbert) %>%
  pivot_longer(cols = c(`MuQ-MuLan`, `LAION-CLAP-Music`, `MERT+SBERT`),
               names_to = "model", values_to = "lmc") %>%
  drop_na(lmc) %>%
  ggplot(aes(x = lmc, fill = genre_cluster)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~model, scales = "free_x") +
  scale_fill_brewer(palette = "Set2", name = "Genre Cluster") +
  labs(
    title    = "Distribution of Lyric-Music Congruence (LMC) by Embedding Model",
    subtitle = "Cosine similarity between audio and text embeddings in shared/independent spaces",
    x        = "LMC Score",
    y        = "Density"
  )
print(p_lmc_dist)
save_fig("01_lmc_distributions.pdf", w = 12, h = 5)

# ── Figure 2: Popularity distribution ────────────────────────────────────────
p_pop <- df_full %>%
  ggplot(aes(x = popularity, fill = genre_cluster)) +
  geom_histogram(bins = 20, color = "white", alpha = 0.8) +
  scale_fill_brewer(palette = "Set2", name = "Genre Cluster") +
  labs(
    title    = "Distribution of Spotify Popularity Scores",
    subtitle = "Recency-weighted composite of cumulative streaming activity (0–100)",
    x        = "Popularity Score",
    y        = "Count"
  )
print(p_pop)
save_fig("02_popularity_distribution.pdf", w = 8, h = 5)

# ── Figure 3: LMC vs. Popularity (MuLan) — main scatter ──────────────────────
smooth_data <- df_full %>% drop_na(lmc_mulan, popularity)

p_scatter_mulan <- smooth_data %>%
  ggplot(aes(x = lmc_mulan, y = popularity)) +
  geom_point(aes(color = genre_cluster, shape = orientation),
             size = 3, alpha = 0.75) +
  geom_smooth(method = "lm", se = TRUE, color = "black",
              linetype = "solid", linewidth = 0.8) +
  geom_smooth(method = "loess", se = FALSE, color = "grey50",
              linetype = "dashed", linewidth = 0.7) +
  geom_text_repel(aes(label = title), size = 2.5, max.overlaps = 15,
                  color = "grey30") +
  scale_color_brewer(palette = "Set2", name = "Genre Cluster") +
  scale_shape_manual(values = c("narrative" = 16, "production" = 17),
                     name = "Orientation") +
  labs(
    title    = "Lyric-Music Congruence (MuQ-MuLan) vs. Spotify Popularity",
    subtitle = "Solid: linear fit; Dashed: LOESS. Points labelled by song title.",
    x        = "LMC Score (MuQ-MuLan)",
    y        = "Spotify Popularity"
  )
print(p_scatter_mulan)
save_fig("03_lmc_vs_popularity_mulan.pdf", w = 11, h = 8)

# ── Figure 4: LMC vs. Popularity — all 3 models panel ────────────────────────
df_three_models <- df_full %>%
  select(song_id, genre_cluster, orientation, popularity,
         `MuQ-MuLan`        = lmc_mulan,
         `LAION-CLAP-Music`  = lmc_clap,
         `MERT+SBERT`        = lmc_mert_sbert) %>%
  pivot_longer(cols = c(`MuQ-MuLan`, `LAION-CLAP-Music`, `MERT+SBERT`),
               names_to = "model", values_to = "lmc") %>%
  drop_na(lmc, popularity)

p_three <- df_three_models %>%
  ggplot(aes(x = lmc, y = popularity, color = genre_cluster)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "black",
              linewidth = 0.8) +
  facet_wrap(~model, scales = "free_x") +
  scale_color_brewer(palette = "Set2", name = "Genre Cluster") +
  labs(
    title    = "LMC vs. Popularity: Comparison Across Embedding Models",
    subtitle = "Joint embedding (MuLan, CLAP) vs. late-fusion baseline (MERT+SBERT)",
    x = "LMC Score", y = "Spotify Popularity"
  )
print(p_three)
save_fig("04_lmc_vs_popularity_all_models.pdf", w = 13, h = 5)

# ── Figure 5: Genre box plot ──────────────────────────────────────────────────
p_genre_box <- df_full %>%
  drop_na(lmc_mulan) %>%
  ggplot(aes(x = reorder(genre_cluster, lmc_mulan, median),
             y = lmc_mulan, fill = genre_cluster)) +
  geom_boxplot(alpha = 0.75, outlier.shape = 21) +
  geom_jitter(width = 0.15, alpha = 0.4, size = 1.5) +
  scale_fill_brewer(palette = "Set2", guide = "none") +
  coord_flip() +
  labs(
    title    = "LMC Distribution by Genre Cluster (MuQ-MuLan)",
    subtitle = "Is lyric-music congruence systematically different across genres?",
    x = "Genre Cluster", y = "LMC Score (MuQ-MuLan)"
  )
print(p_genre_box)
save_fig("05_genre_boxplot.pdf", w = 8, h = 5)

# ── Figure 6: Orientation moderation ─────────────────────────────────────────
p_orient <- df_full %>%
  drop_na(lmc_mulan, popularity) %>%
  ggplot(aes(x = lmc_mulan, y = popularity,
             color = orientation, fill = orientation)) +
  geom_point(aes(shape = genre_cluster), alpha = 0.65, size = 2.5) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.15, linewidth = 1) +
  scale_color_manual(values = ORIENTATION_COLORS, name = "Orientation") +
  scale_fill_manual(values  = ORIENTATION_COLORS, name = "Orientation") +
  scale_shape_manual(values = c(16, 17, 15, 18), name = "Genre Cluster") +
  labs(
    title    = "LMC × Orientation Interaction (H2)",
    subtitle = paste0("Narrative-forward artists: fluency + transportation pathway\n",
                      "Production-forward artists: fluency pathway only"),
    x = "LMC Score (MuQ-MuLan)", y = "Spotify Popularity"
  ) +
  theme(legend.position = "right")
print(p_orient)
save_fig("06_orientation_moderation.pdf", w = 10, h = 7)

# ── Figure 7: Nonlinearity (quadratic fit) ────────────────────────────────────
p_nonlin <- df_full %>%
  drop_na(lmc_mulan, popularity) %>%
  ggplot(aes(x = lmc_mulan, y = popularity)) +
  geom_point(aes(color = genre_cluster), size = 2.5, alpha = 0.65) +
  stat_smooth(method   = "lm",
              formula   = y ~ poly(x, 2),
              se        = TRUE,
              color     = "black",
              linewidth = 1,
              linetype  = "solid") +
  stat_smooth(method = "lm",
              formula = y ~ x,
              se      = FALSE,
              color   = "grey60",
              linewidth = 0.7,
              linetype  = "dashed") +
  scale_color_brewer(palette = "Set2", name = "Genre Cluster") +
  labs(
    title    = "Nonlinearity Test: Quadratic LMC Effect (H3)",
    subtitle = "Solid: quadratic fit; Dashed: linear baseline",
    x = "LMC Score (MuQ-MuLan)", y = "Spotify Popularity"
  )
print(p_nonlin)
save_fig("07_nonlinearity.pdf", w = 9, h = 6)

# ── Figure 8: Coefficient comparison across models ────────────────────────────
if (exists("cv_table") && nrow(df_trio) >= 20) {
  coef_df <- tibble(
    model = c("MuQ-MuLan\n(joint)", "LAION-CLAP-Music\n(joint)", "MERT+SBERT\n(late fusion)"),
    beta  = c(fixef(mc_mulan)["lmc_mulan_c"],
              fixef(mc_clap)["lmc_clap_c"],
              fixef(mc_sbert)["lmc_mert_sbert_c"]),
    se    = c(sqrt(diag(vcov(mc_mulan)))["lmc_mulan_c"],
              sqrt(diag(vcov(mc_clap)))["lmc_clap_c"],
              sqrt(diag(vcov(mc_sbert)))["lmc_mert_sbert_c"]),
    joint = c(TRUE, TRUE, FALSE)
  ) %>%
    mutate(
      lo = beta - 1.96 * se,
      hi = beta + 1.96 * se
    )

  p_coefs <- coef_df %>%
    ggplot(aes(x = beta, y = reorder(model, beta), color = joint)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2, linewidth = 1) +
    geom_point(size = 4) +
    scale_color_manual(values = c("FALSE" = "#E76F51", "TRUE" = "#2A9D8F"),
                       labels = c("FALSE" = "Late Fusion", "TRUE" = "Joint Embedding"),
                       name   = "Training Strategy") +
    labs(
      title    = "LMC Effect on Popularity: Cross-Model Coefficient Comparison",
      subtitle = "Mixed-effects estimates (artist random intercepts). Bars = 95% CI.",
      x = "Standardised Coefficient (β)", y = NULL
    )
  print(p_coefs)
  save_fig("08_model_comparison_coefs.pdf", w = 9, h = 5)
}

# ── Figure 9: Correlation matrix ─────────────────────────────────────────────
corr_vars <- c("popularity", "lmc_mulan", "lmc_clap", "lmc_mert_sbert",
               "energy", "danceability", "valence", "acousticness",
               "speechiness", "loudness")

corr_labels <- c("Popularity", "LMC MuLan", "LMC CLAP", "LMC MERT+SBERT",
                 "Energy", "Danceability", "Valence", "Acousticness",
                 "Speechiness", "Loudness")

corr_data <- df_full %>%
  select(all_of(corr_vars)) %>%
  drop_na()

if (nrow(corr_data) >= 5) {
  corr_matrix <- cor(corr_data, use = "pairwise.complete.obs")
  rownames(corr_matrix) <- corr_labels
  colnames(corr_matrix) <- corr_labels

  pdf(file.path(fig_dir, "09_correlation_matrix.pdf"), width = 9, height = 8)
  corrplot(corr_matrix,
           method    = "color",
           type      = "upper",
           addCoef.col = "black",
           number.cex  = 0.65,
           tl.cex      = 0.8,
           tl.col      = "black",
           col         = COL2("RdBu", 200),
           cl.ratio    = 0.2,
           title       = "Correlation Matrix: LMC, Popularity, and Audio Features",
           mar         = c(0, 0, 2, 0))
  dev.off()
  message("  Figure saved: 09_correlation_matrix.pdf")
}

# ── Figure 10: Segment heatmap (if segment data available) ───────────────────
if (file.exists(seg_path) && exists("df_seg") && nrow(df_seg) >= 10) {
  seg_detail_path <- here::here("results", "segment_analysis", "segment_details.json")

  if (file.exists(seg_detail_path)) {
    seg_detail_raw <- jsonlite::fromJSON(seg_detail_path)

    # Flatten to data frame
    seg_long <- map_dfr(names(seg_detail_raw), function(sid) {
      d <- seg_detail_raw[[sid]]
      if (length(d) == 0) return(NULL)
      as_tibble(d) %>% mutate(song_id = sid)
    })

    if (nrow(seg_long) > 0) {
      seg_heat_df <- seg_long %>%
        left_join(df_full %>% select(song_id, artist_name, genre_cluster,
                                     orientation, popularity),
                  by = "song_id") %>%
        mutate(
          song_label    = paste0(str_sub(title %||% song_id, 1, 20)),
          section_label = paste0(section_idx + 1, ". ", str_sub(header, 1, 12))
        )

      p_heatmap <- seg_heat_df %>%
        filter(song_id %in% (df_seg %>% arrange(desc(popularity)) %>%
                               slice_head(n = 25) %>% pull(song_id))) %>%
        ggplot(aes(x = section_label, y = reorder(song_id, -lmc),
                   fill = lmc)) +
        geom_tile(color = "white") +
        scale_fill_gradient2(low = "#E63946", mid = "#F4F1DE", high = "#2A9D8F",
                             midpoint = median(seg_heat_df$lmc, na.rm = TRUE),
                             name = "LMC") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
              axis.text.y = element_text(size = 7)) +
        labs(
          title    = "Segment-Level LMC Heatmap (MuQ-MuLan, top 25 by popularity)",
          subtitle = "Each column = one lyrical section. Colour = congruence score.",
          x = "Section", y = "Song ID"
        )
      print(p_heatmap)
      save_fig("10_segment_heatmap.pdf", w = 14, h = 8)
    }
  }
}

# ── Figure 11: PCA biplot (audio embedding space) ────────────────────────────
pca_vars <- c("lmc_mulan", "lmc_clap", "lmc_mert_sbert",
              "energy", "danceability", "valence", "acousticness", "speechiness")

pca_data <- df_full %>%
  select(song_id, genre_cluster, orientation, popularity, all_of(pca_vars)) %>%
  drop_na()

if (nrow(pca_data) >= 10) {
  pca_res  <- prcomp(pca_data %>% select(all_of(pca_vars)), scale. = TRUE)
  pca_df   <- as_tibble(pca_res$x[, 1:2]) %>%
    bind_cols(pca_data %>% select(song_id, genre_cluster, orientation, popularity))
  pca_load <- as_tibble(pca_res$rotation[, 1:2], rownames = "variable")
  scale_f  <- max(abs(pca_df[, c("PC1","PC2")])) /
               max(abs(pca_load[, c("PC1","PC2")])) * 0.7

  p_pca <- ggplot() +
    geom_point(data = pca_df,
               aes(x = PC1, y = PC2, color = genre_cluster,
                   shape = orientation, size = popularity),
               alpha = 0.7) +
    geom_segment(data = pca_load,
                 aes(x = 0, y = 0, xend = PC1 * scale_f, yend = PC2 * scale_f),
                 arrow = arrow(length = unit(0.2, "cm")), color = "black",
                 linewidth = 0.6) +
    geom_text_repel(data = pca_load,
                    aes(x = PC1 * scale_f * 1.1, y = PC2 * scale_f * 1.1,
                        label = variable),
                    size = 3.2, color = "black") +
    scale_color_brewer(palette = "Set2", name = "Genre Cluster") +
    scale_shape_manual(values = c("narrative" = 16, "production" = 17),
                       name = "Orientation") +
    scale_size_continuous(name = "Popularity", range = c(1.5, 5)) +
    labs(
      title    = "PCA Biplot: LMC and Audio Feature Space",
      subtitle = paste0("PC1: ",
                        round(pca_res$sdev[1]^2 / sum(pca_res$sdev^2) * 100, 1),
                        "% var | PC2: ",
                        round(pca_res$sdev[2]^2 / sum(pca_res$sdev^2) * 100, 1), "% var"),
      x = "PC1", y = "PC2"
    )
  print(p_pca)
  save_fig("11_pca_biplot.pdf", w = 11, h = 8)
}

# =============================================================================
# Done
# =============================================================================
message("\n", strrep("═", 60))
message("Analysis complete.")
message(sprintf("  Tables  → %s", tab_dir))
message(sprintf("  Figures → %s", fig_dir))
message(strrep("═", 60))

