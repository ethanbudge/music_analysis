# =============================================================================
# run_models_sweep.R — Fit the expanded LMC → popularity battery (60 fits).
#
# Grid: 4 models × 3 prompts = 12 "embeddings" (e.g. mulan_raw, clamp3_idea),
# each fit at 5 structures:
#     track model : song, line_buf1, line_buf5, line_buf10   (4 fits)
#     segment model : chorus vs non-chorus                    (1 fit)
#   → 60 Stan fits. NO curvature models this time.
#
# We SOURCE the observational stan/run_models.R and reuse its data builders
# (track_data, segment_data, controls_matrix, base_stan_data) and fit_one() —
# the Stan models (model_track_v4, model_segment_v4) are UNCHANGED. Only the
# battery, the master CSV, and the output directory differ.
#
# Sampling: 4 chains × (1000 warmup + 1000 sampling), MERT PCA controls, one
# GLOBAL complete-case corpus (a song must have all 60 measures) so every LOO
# object is mutually comparable.
#
# Usage:
#   Rscript sweep/R/run_models_sweep.R                 # all, N=all, controls=mert
#   Rscript sweep/R/run_models_sweep.R 500 1 mert      # N, seed, controls
# =============================================================================

suppressPackageStartupMessages({ library(cmdstanr); library(tidyverse); library(loo); library(here) })

# Reuse every helper from the observational runner (does NOT auto-run when sourced).
source(here::here("stan", "run_models.R"))

# Redirect fit outputs to the sweep folder (fit_one() writes to the global OUTPUT_DIR).
OUTPUT_DIR <- here::here("sweep", "output")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

SWEEP_MASTER   <- here::here("results", "master_results_sweep.csv")
SWEEP_MODELS   <- c("mulan", "clap", "msclap", "clamp3")
SWEEP_PROMPTS  <- c("raw", "contains", "idea")
SWEEP_MEASURES <- c("song", "line_buf1", "line_buf5", "line_buf10")   # track-model measures

# 12 embedding tokens, model-major: mulan_raw, mulan_contains, …, clamp3_idea.
sweep_embeddings <- function()
  as.vector(t(outer(SWEEP_MODELS, SWEEP_PROMPTS, paste, sep = "_")))

# The 6 LMC columns an embedding needs (4 track measures + 2 segment measures).
embedding_cols_sweep <- function(emb)
  c(paste0(emb, "_", SWEEP_MEASURES), paste0(emb, "_seg_chorus"), paste0(emb, "_seg_nonchorus"))

load_master_sweep <- function() {
  if (!file.exists(SWEEP_MASTER))
    stop("Not found: ", SWEEP_MASTER, " — run the embeddings_sweep.ipynb notebook first ",
         "(build_master.build()).")
  read_csv(SWEEP_MASTER, show_col_types = FALSE)
}

# Which embeddings actually have all their columns in the master (skip un-computed ones).
available_embeddings_sweep <- function(master) {
  embs <- sweep_embeddings()
  keep <- embs[vapply(embs, function(e) all(embedding_cols_sweep(e) %in% names(master)), logical(1))]
  missing <- setdiff(embs, keep)
  if (length(missing))
    warning(sprintf("Skipping embedding(s) with missing columns: %s. ",
                    paste(missing, collapse = ", ")),
            "Compute them in the notebook + rebuild the sweep master.")
  keep
}

# GLOBAL complete-case corpus across every available embedding × structure.
build_corpus_sweep <- function(N = NULL, seed = 42, controls = "mert",
                               master = load_master_sweep()) {
  embs <- available_embeddings_sweep(master)
  if (!length(embs)) stop("No sweep embeddings are fully present in ", SWEEP_MASTER)
  required <- c("spotify_popularity", "genre", control_cols(master, controls),
                unlist(lapply(embs, embedding_cols_sweep)))
  list(df = sample_corpus(master, N = N, seed = seed, required = required), embeddings = embs)
}

# ─── Battery ─────────────────────────────────────────────────────────────────
run_all_sweep <- function(N = NULL, seed = 42, controls = "mert",
                          iter = 1000, chains = 4) {
  corpus <- build_corpus_sweep(N = N, seed = seed, controls = controls)
  df <- corpus$df; embs <- corpus$embeddings
  message(sprintf("Shared complete-case corpus: %d songs | embeddings: %d | controls: %s | %d fits",
                  nrow(df), length(embs), controls, length(embs) * 5))

  loos <- list()
  for (emb in embs) {
    for (meas in SWEEP_MEASURES) {                       # 4 track fits per embedding
      col <- paste0(emb, "_", meas)
      if (!col %in% names(df)) next
      fit <- fit_one("model_track_v4", track_data(df, col, controls),
                     paste0("track_", emb, "_", meas), chains = chains, iter = iter)
      loos[[paste0(emb, ":track_", meas)]] <- fit$loo()
    }
    fit <- fit_one("model_segment_v4", segment_data(df, emb, controls),   # 1 segment fit
                   paste0("segment_", emb), chains = chains, iter = iter)
    loos[[paste0(emb, ":segment")]] <- fit$loo()
  }

  # ── LOO comparison across all 60 (valid: one shared corpus) ─────────────────
  if (length(loos) > 1) {
    cmp <- loo::loo_compare(loos)
    saveRDS(cmp, file.path(OUTPUT_DIR, "loo_compare_all_sweep.rds"))
    tidy <- imap_dfr(loos, function(l, nm) {
      parts <- strsplit(nm, ":", fixed = TRUE)[[1]]
      mp <- strsplit(parts[1], "_", fixed = TRUE)[[1]]      # embedding = model_prompt
      struct <- parts[2]
      tag <- if (struct == "segment") paste0("segment_", parts[1])
             else paste0("track_", parts[1], "_", sub("^track_", "", struct))
      tibble(tag = tag, embedding = parts[1], model = mp[1], prompt = mp[2], structure = struct,
             elpd_loo = l$estimates["elpd_loo", "Estimate"],
             se       = l$estimates["elpd_loo", "SE"],
             p_loo    = l$estimates["p_loo", "Estimate"],
             k_gt_0.7 = sum(l$diagnostics$pareto_k > 0.7))
    }) %>% arrange(desc(elpd_loo))
    write_csv(tidy, file.path(OUTPUT_DIR, "loo_all_sweep.csv"))
    message("Best 6 by elpd_loo:"); print(utils::head(tidy, 6))
  }
  message("Done. ", length(loos), " fits saved to ", OUTPUT_DIR)
}

# Run when invoked as a script (not when sourced).
if (sys.nframe() == 0) {
  a <- commandArgs(trailingOnly = TRUE)
  run_all_sweep(N        = if (length(a) >= 1) as.integer(a[1]) else NULL,
                seed     = if (length(a) >= 2) as.integer(a[2]) else 42,
                controls = if (length(a) >= 3) a[3] else "mert",
                iter = 1000, chains = 4)
}
