# =============================================================================
# evaluate_models.R — Family-agnostic sampler/diagnostic sweep over every fit in
# stan/output/. Complements analysis/lmc_report.qmd (which does the substantive
# posterior interpretation); this script is the quick "are the chains healthy and
# is LOO trustworthy?" pass for the WHOLE v4 battery at once.
#
# Per fit: divergences, max-treedepth hits, per-chain E-BFMI, max R-hat, min ESS,
# LOO (elpd / p_loo / Pareto-k), a trace + energy + Pareto-k PDF, and a PPC overlay
# (observed y reconstructed via the data builders, using the controls toggle saved
# in <tag>.labels.rds). Works for any family — track / segment / curve / segcurve /
# linecurve — by operating on whatever SCALAR parameters each fit exposes.
#
# Usage:  Rscript stan/evaluate_models.R [N] [seed]   (N/seed only for PPC corpus)
# =============================================================================

suppressPackageStartupMessages({
  library(cmdstanr); library(posterior); library(bayesplot)
  library(loo); library(tidyverse); library(here)
})
bayesplot::color_scheme_set("brightblue"); theme_set(theme_minimal(base_size = 11))

STAN_DIR <- here::here("stan"); OUTPUT <- file.path(STAN_DIR, "output")
FIG_DIR  <- here::here("analysis", "output", "figures", "diagnostics")
TAB_DIR  <- here::here("analysis", "output", "tables")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)
source(file.path(STAN_DIR, "run_models.R"))   # data builders (for PPC), build_corpus

args <- commandArgs(trailingOnly = TRUE)
N_arg    <- if (length(args) >= 1) as.integer(args[1]) else NULL
seed_arg <- if (length(args) >= 2) as.integer(args[2]) else 42

# Curated focal scalars to trace if present (intersected with each fit's params).
FOCAL_ANY <- c("beta_lmc", "gamma_lmc", "beta_chorus", "beta_nonchorus",
               "chorus_vs_nonchorus", "gamma_chorus", "gamma_intercept", "theta",
               "mu_global", "sigma_b", "sigma_bc", "sigma_bnc", "sigma_f",
               "sigma_line", "tau_u", "sigma_genre", "sigma_lmc_genre",
               "sigma_chorus_genre", "phi_intercept", "sigma_phi_genre")
FUNNEL <- c("phi_intercept", "sigma_phi_genre", "sigma_genre", "sigma_lmc_genre",
            "sigma_chorus_genre", "sigma_b", "tau_u", "sigma_line")

fit_family <- function(tag) sub("_.*$", "", tag)   # track/segment/curve/segcurve/linecurve
tag_embedding <- function(tag) str_match(tag, "^[a-z]+_([a-z]+)")[, 2]

# Reconstruct a fit's Stan data (for observed y) using the saved controls toggle.
rebuild_data <- function(tag, df, controls) {
  fam <- fit_family(tag); emb <- tag_embedding(tag)
  meas <- if (fam == "track") sub("^track_[a-z]+_", "", tag) else NA
  switch(fam,
    track    = track_data(df, paste0(emb, "_", meas), controls),
    segment  = segment_data(df, emb, controls),
    curve    = curve_data(df, emb, controls = controls),
    segcurve = segment_curve_data(df, emb, controls = controls),
    linecurve = line_curve_data(df, emb, controls = controls), NULL)
}

fit_files <- list.files(OUTPUT, pattern = "\\.rds$", full.names = TRUE)
fit_files <- fit_files[!grepl("loo_compare|loo_trajectory|\\.labels", basename(fit_files))]
if (!length(fit_files)) stop("No fits in ", OUTPUT, " — run run_models.R first.")
tags <- sub("\\.rds$", "", basename(fit_files))
message(sprintf("Found %d fits: %s", length(tags), paste(tags, collapse = ", ")))

# Best-effort shared corpus for PPC overlays (controls read per-fit below).
embs <- unique(na.omit(tag_embedding(tags)))

diag_rows <- list(); loo_list <- list()
for (i in seq_along(fit_files)) {
  tag <- tags[i]; message("\n──────── ", tag, " ────────")
  fit <- readRDS(fit_files[i])
  mp <- fit$metadata()$model_params; scalars <- mp[!grepl("\\[", mp)]
  focal <- intersect(FOCAL_ANY, scalars)

  ds <- fit$diagnostic_summary(quiet = TRUE)
  sm <- fit$summary(focal)
  diag_rows[[tag]] <- tibble(
    fit = tag, family = fit_family(tag), n_div = sum(ds$num_divergent),
    n_treedepth = sum(ds$num_max_treedepth), ebfmi_min = min(ds$ebfmi),
    ebfmi_lt_0.3 = sum(ds$ebfmi < 0.3), rhat_max = max(sm$rhat, na.rm = TRUE),
    ess_bulk_min = min(sm$ess_bulk, na.rm = TRUE))
  print(diag_rows[[tag]]); cat("  E-BFMI/chain:", sprintf("%.2f", ds$ebfmi), "\n")

  loo_obj <- tryCatch(fit$loo(), error = function(e) NULL)
  if (!is.null(loo_obj)) {
    loo_list[[tag]] <- loo_obj; pk <- loo_obj$diagnostics$pareto_k
    cat(sprintf("  elpd_loo=%.1f (se %.1f) | p_loo=%.1f | k>0.7: %d/%d\n",
                loo_obj$estimates["elpd_loo", "Estimate"], loo_obj$estimates["elpd_loo", "SE"],
                loo_obj$estimates["p_loo", "Estimate"], sum(pk > 0.7), length(pk)))
  }

  np <- nuts_params(fit)
  pdf(file.path(FIG_DIR, paste0(tag, ".pdf")), width = 10, height = 7)
  if (length(focal)) print(mcmc_trace(fit$draws(focal), np = np) + ggtitle(paste0(tag, " — trace")))
  print(mcmc_nuts_energy(np) + ggtitle(paste0(tag, " — energy (low E-BFMI ⇒ poor mixing)")))
  pv <- intersect(FUNNEL, scalars)
  if (length(pv) >= 2) print(mcmc_pairs(fit$draws(pv), np = np,
                                        off_diag_args = list(size = 0.6, alpha = 0.4)))
  if (!is.null(loo_obj)) print(plot(loo_obj, main = paste0(tag, " — Pareto-k")))
  lab <- tryCatch(readRDS(file.path(OUTPUT, paste0(tag, ".labels.rds"))), error = function(e) NULL)
  if (!is.null(lab$controls)) {
    cd <- tryCatch(build_corpus(tag_embedding(tag), N = N_arg, seed = seed_arg,
                                controls = lab$controls)$df, error = function(e) NULL)
    sd_data <- if (!is.null(cd)) tryCatch(rebuild_data(tag, cd, lab$controls), error = function(e) NULL)
    yrep <- tryCatch(fit$draws("y_rep", format = "draws_matrix"), error = function(e) NULL)
    if (!is.null(sd_data) && !is.null(yrep) && length(sd_data$y) == ncol(yrep)) {
      idx <- sample(nrow(yrep), min(100, nrow(yrep)))
      print(ppc_dens_overlay(sd_data$y, as.matrix(yrep)[idx, ]) + ggtitle(paste0(tag, " — PPC")))
    }
  }
  dev.off()
  rm(fit); gc(verbose = FALSE)
}

diag_tbl <- bind_rows(diag_rows) %>% arrange(family, fit)
cat("\n=========== SAMPLER HEALTH ===========\n"); print(as.data.frame(diag_tbl), row.names = FALSE)
write_csv(diag_tbl, file.path(TAB_DIR, "sampler_diagnostics.csv"))
bad <- diag_tbl %>% filter(ebfmi_min < 0.3 | rhat_max > 1.01 | n_div > 0)
if (nrow(bad)) cat("\n⚠  Check:", paste(bad$fit, collapse = ", "), "\n")

if (length(loo_list) > 1) {
  cmp <- loo::loo_compare(loo_list)
  cat("\n=========== LOO COMPARISON ===========\n"); print(cmp)
  write_csv(as.data.frame(cmp) %>% rownames_to_column("fit"), file.path(TAB_DIR, "loo_comparison.csv"))
  pk_tbl <- imap_dfr(loo_list, ~ tibble(fit = .y,
    elpd_loo = .x$estimates["elpd_loo", "Estimate"], p_loo = .x$estimates["p_loo", "Estimate"],
    k_gt_0.7 = sum(.x$diagnostics$pareto_k > 0.7), k_max = max(.x$diagnostics$pareto_k))) %>%
    arrange(desc(elpd_loo))
  cat("\n--- LOO reliability (k_gt_0.7 should be 0) ---\n"); print(as.data.frame(pk_tbl), row.names = FALSE)
  write_csv(pk_tbl, file.path(TAB_DIR, "loo_reliability.csv"))
}
message("\nDone. Tables → analysis/output/tables/, figures → analysis/output/figures/diagnostics/")
