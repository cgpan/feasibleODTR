# ============================================================================
# File:     generate_synthetic_data.R
# Purpose:  Simulate the ODTR tutorial's synthetic dataset and compute the
#           corresponding ground-truth optimal rule (unconstrained and a
#           priori constrained).
# Inputs:   script/dgp_functions.R (sourced).
# Outputs:  data/synthetic_odtr_n5000.csv   (observed, TMLE-ready)
#           data/synthetic_odtr_truth.rds   (ground truth; verification only)
# Author:   ODTR tutorial (synthetic data project)
# Date:     2026-04-17
# ============================================================================


# -----------------------------------------------------------------
# 0. Config block (the ONLY place to touch for routine overrides)
# -----------------------------------------------------------------
CONFIG <- list(
  n                 = 10000L,
  seed              = 20260421L,
  out_csv           = "data/synthetic_odtr_n10000.csv",
  out_rds           = "data/synthetic_odtr_truth_n10000.rds",
  dgp_functions_src = "script/dgp_functions.R",
  # Use CONFIG$coef_override to adjust specific DGP coefficients without
  # editing dgp_functions.R. The override list is merged onto default_coefs()
  # by simple recursive replacement.
  coef_override     = NULL
)


# -----------------------------------------------------------------
# 1. Resolve project root, source helpers, check packages
# -----------------------------------------------------------------

# Allow script to be run from either project root or script/.
.project_root <- if (file.exists(CONFIG$dgp_functions_src)) {
  getwd()
} else if (file.exists(file.path("..", CONFIG$dgp_functions_src))) {
  dirname(getwd())
} else {
  stop("Could not locate ", CONFIG$dgp_functions_src,
       "; run from project root or from script/.")
}
setwd(.project_root)
source(CONFIG$dgp_functions_src)

check_packages(c("stats", "utils"))   # base-R only is enough for this script

# Recursive list-merge for coef overrides: right overrides left at each key.
.merge_coefs <- function(base, over) {
  if (is.null(over)) return(base)
  for (nm in names(over)) {
    if (is.list(over[[nm]]) && is.list(base[[nm]])) {
      base[[nm]] <- .merge_coefs(base[[nm]], over[[nm]])
    } else {
      base[[nm]] <- over[[nm]]
    }
  }
  base
}

coefs <- .merge_coefs(default_coefs(), CONFIG$coef_override)


# -----------------------------------------------------------------
# 2. Baseline + stage-by-stage simulation
# -----------------------------------------------------------------

# Fixed master seed; each sampling step uses a distinct sub-seed so that a
# user can tweak one stage without reshuffling the others.
set.seed(CONFIG$seed)
sub_seeds <- list(
  baseline   = CONFIG$seed + 1L,
  prop_g9    = CONFIG$seed + 2L,
  noise_g9   = CONFIG$seed + 3L,
  ident_g10  = CONFIG$seed + 4L,
  prop_g10   = CONFIG$seed + 5L,
  noise_g10  = CONFIG$seed + 6L,
  ident_g11  = CONFIG$seed + 7L,
  prop_g11   = CONFIG$seed + 8L,
  noise_g11  = CONFIG$seed + 9L,
  ident_g12  = CONFIG$seed + 10L,
  prop_g12   = CONFIG$seed + 11L,
  noise_g12  = CONFIG$seed + 12L
)

df <- make_baseline(CONFIG$n, coefs, seed = sub_seeds$baseline)

# -- Stage 1: Grade 9 -------------------------------------------------
U1 <- utilities_g9(df, coefs)
P1 <- row_softmax(U1)
df$course_g9 <- sample_treatment(P1, seed = sub_seeds$prop_g9)
gpa_g9_raw   <- simulate_raw_gpa(df, df$course_g9, "g9", coefs,
                                 ident_name = "math_identity_g9",
                                 seed = sub_seeds$noise_g9)
df$math_identity_g10 <- identity_update(
  prev_ident = df$math_identity_g9,
  gpa_raw    = gpa_g9_raw,
  course     = df$course_g9,
  stage      = "g9",
  coefs      = coefs,
  seed       = sub_seeds$ident_g10
)

# -- Stage 2: Grade 10 ------------------------------------------------
U2 <- utilities_later(df, prev_course = df$course_g9, p = coefs$prop$g10,
                     prev_key_name = "shift_by_prev",
                     ident_name = "math_identity_g10")
P2 <- row_softmax(U2)
df$course_g10 <- sample_treatment(P2, seed = sub_seeds$prop_g10)
gpa_g10_raw   <- simulate_raw_gpa(df, df$course_g10, "g10", coefs,
                                  ident_name = "math_identity_g10",
                                  seed = sub_seeds$noise_g10)
df$math_identity_g11 <- identity_update(
  prev_ident = df$math_identity_g10,
  gpa_raw    = gpa_g10_raw,
  course     = df$course_g10,
  stage      = "g10",
  coefs      = coefs,
  seed       = sub_seeds$ident_g11
)

# -- Stage 3: Grade 11 ------------------------------------------------
U3 <- utilities_later(df, prev_course = df$course_g10, p = coefs$prop$g11,
                     prev_key_name = "shift_by_prev_g10",
                     ident_name = "math_identity_g11")
P3 <- row_softmax(U3)
df$course_g11 <- sample_treatment(P3, seed = sub_seeds$prop_g11)
gpa_g11_raw   <- simulate_raw_gpa(df, df$course_g11, "g11", coefs,
                                  ident_name = "math_identity_g11",
                                  seed = sub_seeds$noise_g11)
df$math_identity_g12 <- identity_update(
  prev_ident = df$math_identity_g11,
  gpa_raw    = gpa_g11_raw,
  course     = df$course_g11,
  stage      = "g11",
  coefs      = coefs,
  seed       = sub_seeds$ident_g12
)

# -- Stage 4: Grade 12 ------------------------------------------------
U4 <- utilities_later(df, prev_course = df$course_g11, p = coefs$prop$g12,
                     prev_key_name = "shift_by_prev_g11",
                     ident_name = "math_identity_g12")
P4 <- row_softmax(U4)
df$course_g12 <- sample_treatment(P4, seed = sub_seeds$prop_g12)
gpa_g12_raw   <- simulate_raw_gpa(df, df$course_g12, "g12", coefs,
                                  ident_name = "math_identity_g12",
                                  seed = sub_seeds$noise_g12)

# -- Cumulative outcomes ---------------------------------------------
raw_mat    <- cbind(gpa_g9_raw, gpa_g10_raw, gpa_g11_raw, gpa_g12_raw)
levels_mat <- cbind(df$course_g9, df$course_g10, df$course_g11, df$course_g12)

df$math_gpa_g9        <- pmin(pmax(gpa_g9_raw, 0), 4)
df$math_gpa_g10_cum   <- cumulative_gpa(raw_mat[, 1:2, drop = FALSE],
                                        levels_mat[, 1:2, drop = FALSE], coefs)
df$math_gpa_g11_cum   <- cumulative_gpa(raw_mat[, 1:3, drop = FALSE],
                                        levels_mat[, 1:3, drop = FALSE], coefs)
df$math_gpa_g12_cum   <- cumulative_gpa(raw_mat[, 1:4, drop = FALSE],
                                        levels_mat[, 1:4, drop = FALSE], coefs)


# -----------------------------------------------------------------
# 3. Observed dataset column ordering (TMLE-ready)
# -----------------------------------------------------------------
# Treatments written as integers 1..J (cheap; demo_tmle_pipeline.R converts
# to factors at load time, matching the reference pipeline's conventions).

observed_cols <- c(
  "pre_hs_math", "student_ses", "student_male",
  "math_identity_g9",  "course_g9",  "math_gpa_g9",
  "math_identity_g10", "course_g10", "math_gpa_g10_cum",
  "math_identity_g11", "course_g11", "math_gpa_g11_cum",
  "math_identity_g12", "course_g12", "math_gpa_g12_cum"
)
observed_df <- df[, observed_cols]


# -----------------------------------------------------------------
# 4. Ground-truth optimal rule (vectorized DP over 4x5x6x7 tuples)
# -----------------------------------------------------------------
#
# For each observation i with baseline (pre_hs_math, ses, male, identity_g9)
# we enumerate every candidate sequence (a1, a2, a3, a4) and compute the
# DETERMINISTIC expected Y4 (replace every noise draw by its mean, use
# expected identity transitions). The best sequence per obs gives:
#   - the stage-1 optimal action (constrained / unconstrained),
#   - the deterministic value V* under that policy.
#
# We do both an unconstrained pass (all levels) and a constrained pass
# (tuples that respect the a_priori_feasible prerequisite chain).

raw_gpa_mean_vec <- function(df_i, course_scalar, stage, ident_vec) {
  # Convenience wrapper for the DP: a single integer treatment broadcast to
  # every row; ident_vec passed directly instead of a column name. Must
  # stay in sync with raw_gpa_mean() in dgp_functions.R, including the
  # Option-B prehs_top bonus on top-level courses.
  o <- coefs$outcome
  sk <- o[[stage]]
  readiness <- pnorm(o$readiness_prehs * df_i$pre_hs_math +
                     o$readiness_ident * ident_vec)
  a <- course_scalar
  mismatch <- o$mismatch_scalar * pmax(0, sk$advance[a] - readiness)^2
  prehs_top_vec <- if (is.null(sk$prehs_top)) rep(0, length(sk$main)) else sk$prehs_top
  prehs_top_bonus <- prehs_top_vec[a] * df_i$pre_hs_math
  o$intercept +
    o$beta_prehs * df_i$pre_hs_math +
    o$beta_ses   * df_i$student_ses +
    o$beta_male  * df_i$student_male +
    o$beta_ident * ident_vec +
    sk$main[a] +
    sk$interact[a] * ident_vec +
    mismatch +
    prehs_top_bonus
}

deterministic_identity <- function(prev_ident, gpa_mean, course_scalar, stage) {
  iu <- coefs$identity_update
  boost <- iu[[paste0("boost_", stage)]][course_scalar]
  out <- iu$persist * prev_ident +
         iu$gpa_boost * (gpa_mean - iu$gpa_center) +
         iu$course_boost * boost
  pmin(pmax(out, iu$clip_lo), iu$clip_hi)
}

tuple_feasible <- function(a1, a2, a3, a4) {
  all(c(
    a2 %in% a_priori_feasible(list(course_g9 = a1), "g10"),
    a3 %in% a_priori_feasible(list(course_g9 = a1, course_g10 = a2), "g11"),
    a4 %in% a_priori_feasible(list(course_g9 = a1, course_g10 = a2,
                                   course_g11 = a3), "g12")
  ))
}

# V2.4 PS-aware oracle helper. Same shape as utilities_later() but takes a
# free-form identity vector (instead of the column-name lookup) and a single
# scalar previous-course candidate broadcast to all rows -- so we can
# evaluate the TRUE propensity along the deterministic DP transition path
# inside the 840-tuple enumeration.
utilities_later_vec <- function(df, prev_course_scalar, p, prev_key_name,
                                ident_vec) {
  n <- nrow(df)
  J <- length(p$alpha)
  U <- matrix(0, n, J)
  for (j in seq_len(J)) {
    U[, j] <- p$alpha[j] +
              p$beta_prehs[j] * df$pre_hs_math +
              p$beta_ses[j]   * df$student_ses +
              p$beta_ident[j] * ident_vec
  }
  shift_vec <- p[[prev_key_name]][[as.character(prev_course_scalar)]]
  U + matrix(shift_vec, n, J, byrow = TRUE)
}

n <- CONFIG$n
slope <- coefs$cumulative$slope

# V2.4 PS-trim cutoff: uniform 0.05 at every stage. Deliberately stricter
# than the TMLE pipeline's per-stage cutoffs (0.05 / 0.04 / 0.03 / 0.03)
# so the PS-aware oracle is an independent benchmark, not a number tuned
# to match the estimator (which would invite a data-hacking critique).
PS_TRIM_CUT <- 0.05

# containers
Y_unc   <- rep(-Inf, n); best_tuple_unc   <- matrix(NA_integer_, n, 4)
Y_con   <- rep(-Inf, n); best_tuple_con   <- matrix(NA_integer_, n, 4)
# V2.4 containers: PS-trimmed unconstrained / constrained variants.
Y_unc_pstrim <- rep(-Inf, n); best_tuple_unc_pstrim <- matrix(NA_integer_, n, 4)
Y_con_pstrim <- rep(-Inf, n); best_tuple_con_pstrim <- matrix(NA_integer_, n, 4)
colnames(best_tuple_unc) <- colnames(best_tuple_con) <-
  colnames(best_tuple_unc_pstrim) <- colnames(best_tuple_con_pstrim) <-
  c("a1","a2","a3","a4")

# Stage-1 propensity matrix (does not depend on any DP candidate; uses the
# observed math_identity_g9). Rows of the corresponding column give the
# true P(A1 = a1 | H1) for each student.
P1_dp <- row_softmax(utilities_g9(df, coefs))

# immediate-reward blips at stage 1: n x 4 matrix of E[gpa_g9_raw | H, a].
blip_g9_myopic <- matrix(NA_real_, n, coefs$levels$g9)

for (a1 in seq_len(coefs$levels$g9)) {
  mu1_vec <- raw_gpa_mean_vec(df, a1, "g9", df$math_identity_g9)
  blip_g9_myopic[, a1] <- mu1_vec

  w1     <- 1 + slope * (a1 - 1)
  ident2 <- deterministic_identity(df$math_identity_g9, mu1_vec, a1, "g9")

  # PS-aware feasibility at stage 1 for this candidate a1.
  feas1_ps <- P1_dp[, a1] >= PS_TRIM_CUT

  # Stage-2 propensity given (a1) -- depends only on a1 + ident2 (both fixed
  # in this loop body), so compute once, not per-(a3,a4).
  P2_dp_a1 <- row_softmax(
    utilities_later_vec(df, a1, coefs$prop$g10, "shift_by_prev", ident2)
  )

  for (a2 in seq_len(coefs$levels$g10)) {
    mu2_vec <- raw_gpa_mean_vec(df, a2, "g10", ident2)
    w2      <- 1 + slope * (a2 - 1)
    ident3  <- deterministic_identity(ident2, mu2_vec, a2, "g10")

    feas12_ps <- feas1_ps & (P2_dp_a1[, a2] >= PS_TRIM_CUT)

    P3_dp_a2 <- row_softmax(
      utilities_later_vec(df, a2, coefs$prop$g11, "shift_by_prev_g10", ident3)
    )

    for (a3 in seq_len(coefs$levels$g11)) {
      mu3_vec <- raw_gpa_mean_vec(df, a3, "g11", ident3)
      w3      <- 1 + slope * (a3 - 1)
      ident4  <- deterministic_identity(ident3, mu3_vec, a3, "g11")

      feas123_ps <- feas12_ps & (P3_dp_a2[, a3] >= PS_TRIM_CUT)

      P4_dp_a3 <- row_softmax(
        utilities_later_vec(df, a3, coefs$prop$g12, "shift_by_prev_g11", ident4)
      )

      for (a4 in seq_len(coefs$levels$g12)) {
        mu4_vec <- raw_gpa_mean_vec(df, a4, "g12", ident4)
        w4      <- 1 + slope * (a4 - 1)

        num <- w1 * mu1_vec + w2 * mu2_vec + w3 * mu3_vec + w4 * mu4_vec
        den <- w1 + w2 + w3 + w4
        Y_final <- pmin(pmax(num / den, 0), 4)

        feas1234_ps <- feas123_ps & (P4_dp_a3[, a4] >= PS_TRIM_CUT)

        # unconstrained update (structural: ignore both PS and a-priori)
        mask <- Y_final > Y_unc
        if (any(mask)) {
          Y_unc[mask] <- Y_final[mask]
          best_tuple_unc[mask, ] <-
            matrix(c(a1, a2, a3, a4), nrow = sum(mask), ncol = 4, byrow = TRUE)
        }
        # constrained update (a-priori only -- the V2.0 oracle definition)
        if (tuple_feasible(a1, a2, a3, a4)) {
          mask <- Y_final > Y_con
          if (any(mask)) {
            Y_con[mask] <- Y_final[mask]
            best_tuple_con[mask, ] <-
              matrix(c(a1, a2, a3, a4), nrow = sum(mask), ncol = 4, byrow = TRUE)
          }
        }

        # V2.4: PS-trimmed unconstrained (PS positivity only, no a-priori chain)
        if (any(feas1234_ps)) {
          mask <- feas1234_ps & (Y_final > Y_unc_pstrim)
          if (any(mask)) {
            Y_unc_pstrim[mask] <- Y_final[mask]
            best_tuple_unc_pstrim[mask, ] <-
              matrix(c(a1, a2, a3, a4), nrow = sum(mask), ncol = 4, byrow = TRUE)
          }
        }
        # V2.4: PS-trimmed constrained (PS positivity AND a-priori chain --
        # this is the oracle the constrained TMLE estimator is actually
        # targeting once both safeguards are imposed).
        if (tuple_feasible(a1, a2, a3, a4) && any(feas1234_ps)) {
          mask <- feas1234_ps & (Y_final > Y_con_pstrim)
          if (any(mask)) {
            Y_con_pstrim[mask] <- Y_final[mask]
            best_tuple_con_pstrim[mask, ] <-
              matrix(c(a1, a2, a3, a4), nrow = sum(mask), ncol = 4, byrow = TRUE)
          }
        }
      }
    }
  }
}

# Deterministic Y under OBSERVED policy: use observed courses and observed
# math_identity trajectory, plug noise-free mean raw GPA; this is the
# "structural expectation of Y4 if A = A_observed" on the deterministic
# transition system -- NOT the realized Y4 (that is simply math_gpa_g12_cum).
# The verification step uses this alongside realized Y4 for context.
det_Y_obs <- local({
  mu1 <- raw_gpa_mean(df, df$course_g9,  "g9",  coefs, "math_identity_g9")
  mu2 <- raw_gpa_mean(df, df$course_g10, "g10", coefs, "math_identity_g10")
  mu3 <- raw_gpa_mean(df, df$course_g11, "g11", coefs, "math_identity_g11")
  mu4 <- raw_gpa_mean(df, df$course_g12, "g12", coefs, "math_identity_g12")
  w   <- 1 + slope * (levels_mat - 1)
  num <- w[, 1] * mu1 + w[, 2] * mu2 + w[, 3] * mu3 + w[, 4] * mu4
  den <- rowSums(w)
  pmin(pmax(num / den, 0), 4)
})

# Per-stage myopic blips: E[gpa_gk_raw | observed history at k, A_k = a]
# for each stage k. Used by demo_tmle_pipeline.R as the ground-truth
# benchmark for the stage-k optimal action (which is what per-stage TMLE
# learners -- not doing full backward induction -- recover in expectation).
ident_cols <- c(g9  = "math_identity_g9",  g10 = "math_identity_g10",
                g11 = "math_identity_g11", g12 = "math_identity_g12")
compute_myopic_blip <- function(stage) {
  J <- coefs$levels[[stage]]
  ident <- ident_cols[[stage]]
  out <- matrix(NA_real_, n, J)
  for (a in seq_len(J))
    out[, a] <- raw_gpa_mean(df, rep(a, n), stage, coefs, ident)
  out
}
blip_g9_myopic_full  <- compute_myopic_blip("g9")
blip_g10_myopic_full <- compute_myopic_blip("g10")
blip_g11_myopic_full <- compute_myopic_blip("g11")
blip_g12_myopic_full <- compute_myopic_blip("g12")

otr_myopic_per_stage <- list(
  g9  = apply(blip_g9_myopic_full,  1, which.max),
  g10 = apply(blip_g10_myopic_full, 1, which.max),
  g11 = apply(blip_g11_myopic_full, 1, which.max),
  g12 = apply(blip_g12_myopic_full, 1, which.max)
)

# V2.4: students with no PS-feasible tuple (under the 0.05 cutoff) leave
# Y_*_pstrim at -Inf and best_tuple_*_pstrim rows all NA. Coerce -Inf to
# NA so downstream code can detect missing PS-aware oracle values.
ps_unc_missing <- !is.finite(Y_unc_pstrim)
ps_con_missing <- !is.finite(Y_con_pstrim)
Y_unc_pstrim[ps_unc_missing] <- NA_real_
Y_con_pstrim[ps_con_missing] <- NA_real_

truth <- list(
  # Stage-1 optimal actions (the visible G9 flip we teach against)
  otr_g9_unconstr = best_tuple_unc[, "a1"],
  otr_g9_constr   = best_tuple_con[, "a1"],
  # Full optimal sequences (context; less central to the tutorial)
  otr_sequence_unconstr = best_tuple_unc,
  otr_sequence_constr   = best_tuple_con,
  # Deterministic values (used for the V_unconstr > V_constr > V_obs check)
  value_unconstr = Y_unc,
  value_constr   = Y_con,
  value_obs_det  = det_Y_obs,
  value_obs_real = df$math_gpa_g12_cum,
  # V2.4 PS-aware oracle: strongest-Y sequence among tuples that pass a
  # uniform 0.05 propensity cutoff at every stage along the deterministic
  # DP transition path. Two variants:
  #   _unconstr_pstrim: PS positivity only.
  #   _constr_pstrim:   PS positivity + a-priori prereq chain.
  # NA on the sparse subset of students who have no PS-feasible tuple.
  otr_g9_unconstr_pstrim       = best_tuple_unc_pstrim[, "a1"],
  otr_g9_constr_pstrim         = best_tuple_con_pstrim[, "a1"],
  otr_sequence_unconstr_pstrim = best_tuple_unc_pstrim,
  otr_sequence_constr_pstrim   = best_tuple_con_pstrim,
  value_unconstr_pstrim        = Y_unc_pstrim,
  value_constr_pstrim          = Y_con_pstrim,
  ps_trim_cutoff               = PS_TRIM_CUT,
  ps_trim_unc_missing_n        = sum(ps_unc_missing),
  ps_trim_con_missing_n        = sum(ps_con_missing),
  # Stage-1 myopic blips -- visible in the blip plot (shape: n x 4)
  blip_g9_myopic = blip_g9_myopic,
  # Per-stage myopic blips (E[gpa_gk_raw | observed H_k, A_k = a]). Used
  # by demo_tmle_pipeline.R as the ground-truth benchmark when TMLE fits
  # each stage independently on the stage's realized cumulative outcome.
  blip_per_stage_myopic = list(
    g9  = blip_g9_myopic_full,
    g10 = blip_g10_myopic_full,
    g11 = blip_g11_myopic_full,
    g12 = blip_g12_myopic_full
  ),
  otr_myopic_per_stage = otr_myopic_per_stage,
  # Also keep raw stage GPAs for any downstream per-stage truth analysis
  raw_stage_gpas = raw_mat,
  # Feasibility info
  g9_levels  = coefs$levels$g9,
  g10_levels = coefs$levels$g10,
  g11_levels = coefs$levels$g11,
  g12_levels = coefs$levels$g12,
  # Seed & config used
  seed   = CONFIG$seed,
  coefs  = coefs
)


# -----------------------------------------------------------------
# 5. Write outputs
# -----------------------------------------------------------------
if (!dir.exists("data")) dir.create("data")

write.csv(observed_df, file = CONFIG$out_csv, row.names = FALSE)
saveRDS(truth, file = CONFIG$out_rds)

cat("Wrote observed data to: ", CONFIG$out_csv, "\n",
    "Wrote ground truth to:  ", CONFIG$out_rds, "\n",
    "n = ", nrow(observed_df), ", cols = ", ncol(observed_df), "\n", sep = "")

# Quick one-shot summary (no plotting; that belongs in verification)
cat("\n-- quick summary --\n")
cat("course_g9 marginal:\n"); print(round(prop.table(table(observed_df$course_g9)), 3))
cat("mean math_gpa_g12_cum =", round(mean(observed_df$math_gpa_g12_cum), 3), "\n")
cat("V_unconstr        (mean) =", round(mean(Y_unc), 3), "\n")
cat("V_constr          (mean) =", round(mean(Y_con), 3), "\n")
cat("V_unconstr_pstrim (mean) =",
    round(mean(Y_unc_pstrim, na.rm = TRUE), 3),
    " [missing n =", sum(ps_unc_missing), "]\n")
cat("V_constr_pstrim   (mean) =",
    round(mean(Y_con_pstrim, na.rm = TRUE), 3),
    " [missing n =", sum(ps_con_missing), "]\n")
cat("V_obs_real        (mean) =", round(mean(df$math_gpa_g12_cum), 3), "\n")
