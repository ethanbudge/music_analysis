# =============================================================================
# summary_stats.R — Quick descriptive look at the gathered corpus.
#
# Reads results/master_results.csv (+ lmc_lines.csv) and prints/saves:
#   • corpus completeness counts,
#   • distribution of popularity, genre, orientation,
#   • summary statistics for every LMC measure (model × method),
#   • correlations among LMC measures and with popularity,
#   • a couple of overview figures.
#
# Run:  Rscript analysis/summary_stats.R
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse); library(here)
})

RESULTS <- here::here("results")
FIG_DIR <- here::here("analysis", "output", "figures")
TAB_DIR <- here::here("analysis", "output", "tables")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

master <- read_csv(file.path(RESULTS, "master_results.csv"), show_col_types = FALSE)

cat("\n================ CORPUS OVERVIEW ================\n")
cat(sprintf("Songs in master table : %d\n", nrow(master)))
cat(sprintf("With Spotify popularity: %d\n", sum(!is.na(master$spotify_popularity))))
cat(sprintf("With mood features     : %d\n", sum(!is.na(master$mood_happy))))
cat(sprintf("Distinct artists       : %d\n", dplyr::n_distinct(master$artist)))

cat("\n-- Genre (recovered) --\n");        print(sort(table(master$genre), decreasing = TRUE))
cat("\n-- Orientation (recovered) --\n");  print(table(master$orientation, useNA = "ifany"))

cat("\n-- Popularity (Spotify 0-100) --\n")
print(summary(master$spotify_popularity))

# LMC measures: every "<model>_<method>" column.
lmc_cols <- grep("^(mulan|clap)_", names(master), value = TRUE)
if (length(lmc_cols)) {
  cat("\n================ LMC MEASURE SUMMARIES ================\n")
  lmc_summary <- master %>%
    summarise(across(all_of(lmc_cols),
                     list(n = ~sum(!is.na(.)), mean = ~mean(., na.rm = TRUE),
                          sd = ~sd(., na.rm = TRUE), min = ~min(., na.rm = TRUE),
                          max = ~max(., na.rm = TRUE)))) %>%
    pivot_longer(everything(),
                 names_to = c("measure", ".value"), names_pattern = "(.*)_(n|mean|sd|min|max)")
  print(as.data.frame(lmc_summary), digits = 3)
  write_csv(lmc_summary, file.path(TAB_DIR, "lmc_summary.csv"))

  # Correlation of each LMC measure with popularity.
  if (sum(!is.na(master$spotify_popularity)) > 3) {
    cors <- sapply(lmc_cols, function(c)
      suppressWarnings(cor(master[[c]], master$spotify_popularity, use = "pairwise")))
    cat("\n-- Correlation of LMC measures with popularity --\n")
    print(round(sort(cors, decreasing = TRUE), 3))
  }

  # Overview figure: LMC measure distributions.
  p <- master %>% select(track_id, all_of(lmc_cols)) %>%
    pivot_longer(-track_id, names_to = "measure", values_to = "lmc") %>%
    ggplot(aes(lmc)) + geom_histogram(bins = 30) +
    facet_wrap(~measure, scales = "free_y") +
    labs(title = "Distribution of LMC measures", x = "cosine similarity", y = "count") +
    theme_minimal()
  ggsave(file.path(FIG_DIR, "lmc_distributions.pdf"), p, width = 11, height = 7)
}

# Line-level timeline overview, if present.
lines_path <- file.path(RESULTS, "lmc_lines.csv")
if (file.exists(lines_path)) {
  lines <- read_csv(lines_path, show_col_types = FALSE)
  cat(sprintf("\nLine-level rows: %d across %d songs\n",
              nrow(lines), dplyr::n_distinct(lines$track_id)))
  p2 <- lines %>% filter(window == "buf5") %>%
    ggplot(aes(position_pct, lmc, colour = factor(is_chorus))) +
    geom_smooth(se = FALSE, method = "loess", formula = y ~ x) +
    scale_colour_manual(values = c("0" = "grey60", "1" = "firebrick"),
                        labels = c("non-chorus", "chorus"), name = NULL) +
    labs(title = "Line-level LMC over song position (±5 s window)",
         x = "song position (%)", y = "LMC") + theme_minimal()
  ggsave(file.path(FIG_DIR, "lmc_timeline.pdf"), p2, width = 9, height = 5)
}

cat("\nFigures → ", FIG_DIR, "\nTables  → ", TAB_DIR, "\n")
