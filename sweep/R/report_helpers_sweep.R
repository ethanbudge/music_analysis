# =============================================================================
# report_helpers_sweep.R — helpers for sweep/lmc_report_sweep.qmd (60-fit grid).
#
# Reuses analysis/report_helpers.R (extract_fit, plot_effects, plot_genre_slope,
# PPC machinery) and sweep/R/run_models_sweep.R (build_corpus_sweep, the sweep
# data builders), and only OVERRIDES the tag parsers so tags of the form
#   track_<model>_<prompt>_<measure>   and   segment_<model>_<prompt>
# resolve to embedding = "<model>_<prompt>" and measure = "<measure>".
# =============================================================================

suppressPackageStartupMessages({ library(tidyverse); library(here); library(loo) })

source(here::here("analysis", "report_helpers.R"))        # extract_fit + plots + run_models.R
source(here::here("sweep", "R", "run_models_sweep.R"))    # build_corpus_sweep, load_master_sweep

# extract_fit() reads fits from the global OUTPUT — point it at the sweep folder.
OUTPUT <- here::here("sweep", "output")

# ─── Override the tag parsers for {model}_{prompt} embeddings ─────────────────
tag_embedding <- function(tag) {
  m <- stringr::str_match(tag, "^(?:track|segment)_(mulan|clap|msclap|clamp3)_(raw|contains|idea)")
  ifelse(is.na(m[, 2]), NA_character_, paste0(m[, 2], "_", m[, 3]))
}
tag_measure <- function(tag) {
  if (!startsWith(tag, "track_")) return(NA_character_)
  sub("^track_(?:mulan|clap|msclap|clamp3)_(?:raw|contains|idea)_", "", tag, perl = TRUE)
}

# ─── Loading the LOO table + fit list ─────────────────────────────────────────
load_loo_sweep <- function() {
  p <- file.path(OUTPUT, "loo_all_sweep.csv")
  if (!file.exists(p)) stop("Not found: ", p, " — run run_models_sweep.R first.")
  readr::read_csv(p, show_col_types = FALSE)
}

STRUCTURE_LEVELS <- c("track_song", "track_line_buf1", "track_line_buf5",
                      "track_line_buf10", "segment")
STRUCTURE_LABEL  <- c(track_song = "Song-wide", track_line_buf1 = "Line ±1 s",
                      track_line_buf5 = "Line ±5 s", track_line_buf10 = "Line ±10 s",
                      segment = "Segment (chorus/verse)")
MODEL_LABEL  <- c(mulan = "MuQ-MuLan", clap = "LAION-CLAP", msclap = "MS-CLAP", clamp3 = "CLaMP 3")
PROMPT_LABEL <- c(raw = "raw", contains = "contains", idea = "idea")

theme_sweep <- function() theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), plot.title.position = "plot")

# ─── Headline comparison views ────────────────────────────────────────────────
# Δelpd relative to the single best fit (0 = best; more negative = worse).
loo_delta <- function(tab = load_loo_sweep()) {
  best <- max(tab$elpd_loo)
  tab %>% mutate(delta = elpd_loo - best,
                 structure = factor(structure, levels = STRUCTURE_LEVELS),
                 model = factor(model, levels = names(MODEL_LABEL)),
                 prompt = factor(prompt, levels = names(PROMPT_LABEL)))
}

# Heatmap of Δelpd over the full 12 × 5 grid (embedding rows × structure cols).
plot_loo_heatmap <- function(tab = load_loo_sweep()) {
  d <- loo_delta(tab) %>%
    mutate(emb_label = paste0(MODEL_LABEL[as.character(model)], " · ", prompt))
  ggplot(d, aes(structure, reorder(emb_label, elpd_loo), fill = delta)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = sprintf("%.0f", delta)), size = 2.6) +
    scale_x_discrete(labels = STRUCTURE_LABEL) +
    scale_fill_viridis_c(option = "magma", name = "Δelpd\n(vs best)") +
    labs(x = NULL, y = NULL, title = "LOO Δelpd across the 60 fits (0 = best)") +
    theme_sweep() + theme(axis.text.x = element_text(angle = 25, hjust = 1))
}

# Marginal "which is best" summaries: mean elpd by model, by prompt, by structure.
loo_marginals <- function(tab = load_loo_sweep()) list(
  by_model     = tab %>% group_by(model)     %>% summarise(mean_elpd = mean(elpd_loo), .groups = "drop") %>% arrange(desc(mean_elpd)),
  by_prompt    = tab %>% group_by(prompt)    %>% summarise(mean_elpd = mean(elpd_loo), .groups = "drop") %>% arrange(desc(mean_elpd)),
  by_structure = tab %>% group_by(structure) %>% summarise(mean_elpd = mean(elpd_loo), .groups = "drop") %>% arrange(desc(mean_elpd)))

# Diagnostics roll-up across all fits (divergences / R-hat / ESS / Pareto-k).
sweep_diagnostics <- function(tab = load_loo_sweep(), corpus_df = NULL) {
  purrr::map_dfr(tab$tag, function(tg) {
    m <- tryCatch(extract_fit(tg, corpus_df = NULL), error = function(e) NULL)
    if (is.null(m)) return(NULL)
    m$diag
  })
}
