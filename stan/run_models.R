# =============================================================================
# run_models.R — Fit the LMC → popularity Bayesian models (cmdstanr), v4 family.
#
# v4 = the reparameterized v3 structure (no artist effect, non-centred genre,
# genre-varying precision submodel, recalibrated priors) PLUS:
#   • a GENERIC control matrix X[N,K] so the control set is a runtime toggle —
#       controls = "mood" | "mert" | "both" | "none"   (default "mert")
#   • functional (scalar-on-function) trajectory models that replace the old
#     two-stage timeline summaries:
#       model_curve_v4          ∫ β(t)·LMC(t) dt                 ("curvature")
#       model_segment_curve_v4  ∫ β_c(t)·LMC_chorus + β_nc(t)·…  ("segment+curve")
#       model_line_curve_v4     one-stage line-level joint model (experimental)
#
# The whole battery fits on ONE shared complete-case corpus so every LOO object
# is mutually comparable. Genre, control names, and the functional plotting grid
# are persisted next to each fit (<tag>.labels.rds) so the report never has to
# guess a mapping.
#
# Usage:
#   Rscript stan/run_models.R                      # model=both, N=all, controls=mert
#   Rscript stan/run_models.R both 500 1 mert      # embeddings, N, seed, controls
#   Rscript stan/run_models.R mulan 0 1 both       # N=0 → all; controls=mood+MERT
# =============================================================================

suppressPackageStartupMessages({
  library(cmdstanr); library(tidyverse); library(loo); library(here); library(splines)
})

STAN_DIR   <- here::here("stan")
OUTPUT_DIR <- file.path(STAN_DIR, "output")
RESULTS    <- here::here("results")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

zsc <- function(x) as.numeric(scale(x))
boundary_adjust <- function(y, n) (y * (n - 1) + 0.5) / n   # Smithson & Verkuilen

CONTEXT_WINDOWS <- c("exact", "buf1", "buf5", "buf10")
TRACK_MEASURES  <- c("song", paste0("line_", CONTEXT_WINDOWS))
FUNC_WINDOW     <- "buf5"     # context window used by the functional trajectory models
KB_DEFAULT      <- 8         # functional B-spline basis dimension
POLY_DEGREE     <- 2         # orthogonal-polynomial degree (quadratic) for polycurve
GRID_DEFAULT    <- 100       # β(t) plotting-grid resolution

MOOD_COLUMNS <- c("mood_happy", "mood_sad", "mood_relaxed", "mood_aggressive",
                  "mood_party", "danceability", "voice_instrumental")

# ─── Data loading ────────────────────────────────────────────────────────────
load_master <- function() read_csv(file.path(RESULTS, "master_results.csv"), show_col_types = FALSE)

sample_corpus <- function(df, N = NULL, seed = 42, required = NULL) {
  if (!is.null(required)) df <- df %>% drop_na(any_of(required))
  set.seed(seed)
  if (is.null(N) || N == 0 || N >= nrow(df)) return(df)
  df[sample(nrow(df), N), , drop = FALSE]
}

orient_num <- function(o) dplyr::recode(o, narrative = 1, production = 0, .default = 0.5)

# ─── Controls toggle ─────────────────────────────────────────────────────────
# Which columns form the control block for a given toggle. song_age is ALWAYS a
# control (a real confounder, on neither the mood nor MERT path) and is added
# separately in controls_matrix().
control_cols <- function(df, controls) {
  mood <- intersect(MOOD_COLUMNS, names(df))
  mert <- sort(grep("^mert_pc[0-9]+$", names(df), value = TRUE))
  cols <- switch(controls,
                 mood = mood, mert = mert, both = c(mood, mert), none = character(0),
                 stop("controls must be one of: mood, mert, both, none"))
  if (controls %in% c("mert", "both") && !length(mert))
    stop("controls='", controls, "' but no mert_pc* columns in master_results.csv. ",
         "Run mert.extract_pending() + combine.build_master() first.")
  cols
}

# Build the standardised control design matrix X[N,K] (+ song_age) and its names.
controls_matrix <- function(df, controls) {
  cols <- control_cols(df, controls)
  age  <- zsc(replace_na(df$song_age_years, median(df$song_age_years, na.rm = TRUE)))
  X <- if (length(cols)) do.call(cbind, lapply(cols, function(cc) zsc(df[[cc]])))
       else matrix(nrow = nrow(df), ncol = 0)
  if (length(cols)) colnames(X) <- cols
  X <- cbind(X, song_age = age)
  list(K = ncol(X), X = X, names = colnames(X))
}

# Shared (control-free) Stan data: outcome, genre grouping, orientation moderator.
base_stan_data <- function(df) {
  df <- df %>% mutate(genre_id = as.integer(factor(genre)),
                      y = boundary_adjust(spotify_popularity / 100, nrow(df)))
  list(N = nrow(df), N_genre = max(df$genre_id), y = df$y,
       genre_id = df$genre_id,
       orientation = orient_num(df$orientation),
       orientation_known = as.numeric(df$orientation %in% c("narrative", "production")))
}

# Persist the AUTHORITATIVE label map (genre levels, control names, functional
# grid) as an attribute; fit_one() saves it next to the fit.
finalize <- function(stan_list, df, controls, control_names, meta = list()) {
  attr(stan_list, "labels") <- c(list(
    genre_levels  = levels(factor(df$genre)),
    controls      = controls,
    control_names = control_names,
    N             = nrow(df),
    built_at      = as.character(Sys.time())), meta)
  stan_list
}

# ─── Scalar / segment data builders ──────────────────────────────────────────
track_data <- function(df, lmc_col, controls = "mert") {
  stopifnot(lmc_col %in% names(df))
  df <- df %>% drop_na(all_of(c("spotify_popularity", lmc_col, control_cols(df, controls))))
  cm <- controls_matrix(df, controls)
  finalize(c(base_stan_data(df), list(lmc_z = zsc(df[[lmc_col]]), K = cm$K, X = cm$X)),
           df, controls, cm$names)
}

segment_data <- function(df, model, controls = "mert") {
  cc <- paste0(model, "_seg_chorus"); nc <- paste0(model, "_seg_nonchorus")
  df <- df %>% drop_na(all_of(c("spotify_popularity", cc, nc, control_cols(df, controls))))
  cm <- controls_matrix(df, controls)
  finalize(c(base_stan_data(df),
             list(chorus_lmc_z = zsc(df[[cc]]), nonchorus_lmc_z = zsc(df[[nc]]),
                  K = cm$K, X = cm$X)),
           df, controls, cm$names)
}

# ─── Functional (scalar-on-function) machinery ───────────────────────────────
# Two bases over normalised song position t∈[0,1] are supported: a B-spline
# (penalized in Stan → the "curve"/"hier" models) and an ORTHOGONAL POLYNOMIAL
# (degree POLY_DEGREE → the parsimonious "polycurve" model). Each returns an
# evaluator ev(t) → [length(t) × dim] reusing fixed knots/coefs, plus the basis
# on a plotting grid so the report can reconstruct β(t).
build_basis <- function(type = "spline", k = KB_DEFAULT, G = GRID_DEFAULT) {
  proto <- seq(0, 1, length.out = 200)
  if (type == "spline") {
    sp <- splines::bs(proto, df = k, intercept = TRUE)
    ev <- function(t) predict(sp, pmin(pmax(t, 0), 1))
    dim <- k
  } else {                                   # orthogonal polynomial (+ intercept col)
    pb <- poly(proto, degree = k)
    ev <- function(t) cbind(1, predict(pb, pmin(pmax(t, 0), 1)))
    dim <- k + 1
  }
  grid <- seq(0, 1, length.out = G)
  list(ev = ev, dim = dim, G = G, grid = grid, Bgrid = ev(grid))
}

# Line-level series for the functional models (one model × window), restricted to
# the corpus and globally z-scored so β(t) is on a comparable scale.
func_lines <- function(track_ids, model, window = FUNC_WINDOW) {
  read_csv(file.path(RESULTS, "lmc_lines.csv"), show_col_types = FALSE) %>%
    filter(model == !!model, window == !!window, track_id %in% track_ids) %>%
    arrange(track_id, line_idx) %>%
    mutate(pos = position_pct / 100, lmc_z = zsc(lmc))
}

# Per-song functional predictor P[s,·] ≈ ∫ x_s(t) φ(t) dt (mean over the song's
# lines, optionally restricted to e.g. chorus-only lines), for any basis evaluator.
func_predictor <- function(lines, ev, ids, subset = NULL) {
  d0 <- ncol(ev(0.5))
  P <- matrix(0.0, nrow = length(ids), ncol = d0)
  L <- if (is.null(subset)) lines else dplyr::filter(lines, !!subset)
  by_track <- split(L, L$track_id)
  for (i in seq_along(ids)) {
    d <- by_track[[as.character(ids[i])]]
    if (is.null(d) || !nrow(d)) next
    P[i, ] <- colMeans(ev(d$pos) * d$lmc_z)
  }
  P
}

# Common functional pieces for a corpus df (all curve models share these).
func_common <- function(df, model, window, controls, basis = "spline",
                        k = KB_DEFAULT, G = GRID_DEFAULT) {
  lines <- func_lines(df$track_id, model, window)
  df <- df %>% filter(track_id %in% unique(lines$track_id)) %>%
    drop_na(all_of(c("spotify_popularity", control_cols(df, controls))))
  lines <- lines %>% filter(track_id %in% df$track_id)
  b <- build_basis(basis, k, G)
  c(list(df = df, lines = lines, ids = df$track_id, cm = controls_matrix(df, controls)), b)
}

func_meta <- function(fc, window, basis) list(func_window = window, func_grid = fc$grid,
                                              func_dim = fc$dim, func_basis = basis)

# Spline models: curve (population) and hier (genre-varying) share this builder.
curve_data <- function(df, model, window = FUNC_WINDOW, controls = "mert") {
  fc <- func_common(df, model, window, controls, basis = "spline")
  Z <- func_predictor(fc$lines, fc$ev, fc$ids)
  finalize(c(base_stan_data(fc$df),
             list(Kb = fc$dim, Z = Z, G = fc$G, Bgrid = fc$Bgrid, K = fc$cm$K, X = fc$cm$X)),
           fc$df, controls, fc$cm$names, func_meta(fc, window, "spline"))
}

# Orthogonal-polynomial (genre-varying) model.
poly_curve_data <- function(df, model, window = FUNC_WINDOW, controls = "mert",
                            degree = POLY_DEGREE) {
  fc <- func_common(df, model, window, controls, basis = "poly", k = degree)
  P <- func_predictor(fc$lines, fc$ev, fc$ids)
  finalize(c(base_stan_data(fc$df),
             list(D = fc$dim, P = P, G = fc$G, Bgrid = fc$Bgrid, K = fc$cm$K, X = fc$cm$X)),
           fc$df, controls, fc$cm$names, func_meta(fc, window, "poly"))
}

segment_curve_data <- function(df, model, window = FUNC_WINDOW, controls = "mert") {
  fc <- func_common(df, model, window, controls, basis = "spline")
  Zc  <- func_predictor(fc$lines, fc$ev, fc$ids, subset = quote(is_chorus == 1))
  Znc <- func_predictor(fc$lines, fc$ev, fc$ids, subset = quote(is_chorus == 0))
  finalize(c(base_stan_data(fc$df),
             list(Kb = fc$dim, Zc = Zc, Znc = Znc, G = fc$G, Bgrid = fc$Bgrid,
                  K = fc$cm$K, X = fc$cm$X)),
           fc$df, controls, fc$cm$names, func_meta(fc, window, "spline"))
}

# Experimental one-stage line-level model: long line arrays + the song map.
line_curve_data <- function(df, model, window = FUNC_WINDOW, controls = "mert") {
  fc <- func_common(df, model, window, controls, basis = "spline")
  L <- fc$lines %>% mutate(song = match(track_id, fc$ids)) %>% drop_na(song)
  finalize(c(base_stan_data(fc$df),
             list(K = fc$cm$K, X = fc$cm$X,
                  M = nrow(L), song = L$song, Kb = fc$dim,
                  LB = fc$ev(L$pos), lmc = L$lmc_z,
                  G = fc$G, Bgrid = fc$Bgrid)),
           fc$df, controls, fc$cm$names, func_meta(fc, window, "spline"))
}

# ─── Fitting ─────────────────────────────────────────────────────────────────
.compiled <- new.env()
get_model <- function(name) {
  if (is.null(.compiled[[name]]))
    .compiled[[name]] <- cmdstan_model(file.path(STAN_DIR, paste0(name, ".stan")))
  .compiled[[name]]
}

fit_one <- function(stan_name, data, tag, chains = 4, iter = 1500, ...) {
  message(sprintf("── fitting %s  [%s]  (N=%d, genres=%d, K=%d)", stan_name, tag,
                  data$N, data$N_genre, data$K))
  fit <- get_model(stan_name)$sample(
    data = data, chains = chains, parallel_chains = chains,
    iter_warmup = iter, iter_sampling = iter, refresh = 0,
    adapt_delta = 0.95, max_treedepth = 12, seed = 42, ...)
  fit$save_object(file.path(OUTPUT_DIR, paste0(tag, ".rds")))
  lab <- attr(data, "labels")
  if (!is.null(lab)) saveRDS(lab, file.path(OUTPUT_DIR, paste0(tag, ".labels.rds")))
  fit
}

# ─── Corpus assembly ─────────────────────────────────────────────────────────
available_embeddings <- function(df, want) {
  keep <- want[vapply(want, function(m) any(grepl(paste0("^", m, "_"), names(df))), logical(1))]
  missing <- setdiff(want, keep)
  if (length(missing))
    warning(sprintf("No LMC columns for embedding(s): %s — skipping. ", paste(missing, collapse = ", ")),
            "Run embeddings.embed_pending() + combine.build_master() first.")
  keep
}

embedding_cols <- function(model)
  c(paste0(model, "_", TRACK_MEASURES), paste0(model, "_seg_chorus"), paste0(model, "_seg_nonchorus"))

build_corpus <- function(models = c("mulan", "clap"), N = NULL, seed = 42,
                         controls = "mert", master = load_master()) {
  if (identical(models, "both")) models <- c("mulan", "clap")
  models <- available_embeddings(master, models)
  if (!length(models)) stop("None of the requested embeddings are present in master_results.csv.")
  required <- c("spotify_popularity", "genre", control_cols(master, controls),
                unlist(lapply(models, embedding_cols)))
  list(df = sample_corpus(master, N = N, seed = seed, required = required), models = models)
}

# ─── Default battery ─────────────────────────────────────────────────────────
# Fits, per embedding, the track measures + the three trajectory-comparison
# models (segment / curvature / segment+curvature), all on one shared corpus.
run_all <- function(model = c("mulan", "clap"), N = NULL, seed = 42,
                    controls = "mert", include_line = FALSE) {
  corpus <- build_corpus(model, N = N, seed = seed, controls = controls)
  df <- corpus$df; models <- corpus$models
  message(sprintf("Shared corpus: %d songs | embeddings: %s | controls: %s",
                  nrow(df), paste(models, collapse = ", "), controls))

  loos <- list()
  for (model in models) {
    for (m in TRACK_MEASURES) {
      col <- paste0(model, "_", m)
      if (!col %in% names(df)) next
      fit <- fit_one("model_track_v4", track_data(df, col, controls),
                     paste0("track_", model, "_", m))
      loos[[paste0(model, ":track_", m)]] <- fit$loo()
    }
    # Trajectory-comparison trio (segment vs curvature vs segment+curvature).
    fit <- fit_one("model_segment_v4", segment_data(df, model, controls),
                   paste0("segment_", model))
    loos[[paste0(model, ":segment")]] <- fit$loo()
    fit <- fit_one("model_curve_v4", curve_data(df, model, controls = controls),
                   paste0("curve_", model))
    loos[[paste0(model, ":curve")]] <- fit$loo()
    fit <- fit_one("model_segment_curve_v4", segment_curve_data(df, model, controls = controls),
                   paste0("segcurve_", model))
    loos[[paste0(model, ":segcurve")]] <- fit$loo()
    # Genre-varying trajectory: orthogonal-polynomial (robust) + spline (flexible).
    fit <- fit_one("model_curve_poly_v4", poly_curve_data(df, model, controls = controls),
                   paste0("polycurve_", model))
    loos[[paste0(model, ":polycurve")]] <- fit$loo()
    fit <- fit_one("model_curve_hier_v4", curve_data(df, model, controls = controls),
                   paste0("hiercurve_", model))
    loos[[paste0(model, ":hiercurve")]] <- fit$loo()
    if (include_line) {
      fit <- fit_one("model_line_curve_v4", line_curve_data(df, model, controls = controls),
                     paste0("linecurve_", model))
      loos[[paste0(model, ":linecurve")]] <- fit$loo()
    }
  }

  if (length(loos) > 1) {
    cmp <- loo::loo_compare(loos); print(cmp)
    saveRDS(cmp, file.path(OUTPUT_DIR, "loo_compare_all.rds"))
    # Tidy model × embedding LOO table (direct MuLan-vs-CLAP comparison artifact).
    tidy <- imap_dfr(loos, function(l, nm) {
      parts <- strsplit(nm, ":", fixed = TRUE)[[1]]
      tibble(embedding = parts[1], model = parts[2],
             elpd_loo = l$estimates["elpd_loo", "Estimate"], se = l$estimates["elpd_loo", "SE"],
             p_loo = l$estimates["p_loo", "Estimate"], k_gt_0.7 = sum(l$diagnostics$pareto_k > 0.7))
    }) %>% arrange(desc(elpd_loo))
    write_csv(tidy, file.path(OUTPUT_DIR, "loo_all.csv"))
    # Focused trajectory comparison (segment / curve / segment+curve) per embedding.
    for (model in models) {
      sub <- loos[grepl(paste0("^", model, ":(segment|curve|segcurve|polycurve|hiercurve|linecurve)$"), names(loos))]
      if (length(sub) > 1)
        saveRDS(loo::loo_compare(sub),
                file.path(OUTPUT_DIR, paste0("loo_trajectory_", model, ".rds")))
    }
  }
  message("Done. Fits saved to ", OUTPUT_DIR)
}

# Run the battery when invoked as a script (not when sourced).
#   Rscript stan/run_models.R [model] [N] [seed] [controls]
if (sys.nframe() == 0) {
  a <- commandArgs(trailingOnly = TRUE)
  run_all(model    = if (length(a) >= 1) a[1] else c("mulan", "clap"),
          N        = if (length(a) >= 2) as.integer(a[2]) else NULL,
          seed     = if (length(a) >= 3) as.integer(a[3]) else 42,
          controls = if (length(a) >= 4) a[4] else "mert")
}

