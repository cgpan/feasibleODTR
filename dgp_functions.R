# ============================================================================
# File:     dgp_functions.R
# Purpose:  Pure helper functions for the ODTR tutorial's synthetic data
#           generating process (DGP). No side effects, no file I/O.
# Inputs:   None (sourced by generate_synthetic_data.R and
#           verify_synthetic_data.R).
# Outputs:  A collection of named functions plus default_coefs().
# Author:   ODTR tutorial (synthetic data project)
# Date:     2026-04-17
#
# Design notes
# ------------
#   * K = 4 decision stages (Grades 9-12). Treatment menu GROWS with grade:
#       G9  : 1..4   (BasicMath, Algebra1, Geometry, Algebra2)
#       G10 : 1..5   (... + PreCalc)
#       G11 : 1..6   (... + Calculus)
#       G12 : 1..7   (... + APCalc)
#   * Propensities use a softmax over linear-in-history utilities. Near-zero
#     cells are deliberate: they motivate the trimming step in the tutorial.
#   * Outcome (stage-k raw GPA) mixes (i) main effect, (ii) identity x course
#     interaction (the key heterogeneity), (iii) a single scalar mismatch
#     penalty that is stronger for more advanced courses via the shared
#     `advance_k` vector (no per-level mismatch vector -> avoids double
#     counting with `main_k`).
#   * Cumulative GPA can be unweighted or weighted. Weighted uses integer
#     course level via w = 1 + 0.15 * (level - 1): BasicMath 1.00, APCalc 1.90.
#   * Feasibility (`a_priori_feasible`) is NOT enforced in the simulator:
#     rare violations must occur so that TMLE has something to trim.
# ============================================================================


# ----------------------------------------------------------------------------
# Package availability helper
# ----------------------------------------------------------------------------
# Rationale: don't install packages silently; print actionable instructions.

#' Check that required packages are installed; stop with an install hint if not.
#' @param pkgs character vector of package names.
#' @return invisible(TRUE) if all available; otherwise stop().
check_packages <- function(pkgs) {
  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    msg <- paste0(
      "The following required package(s) are not installed: ",
      paste(missing_pkgs, collapse = ", "), "\n",
      "Install with: install.packages(c(",
      paste(sprintf('"%s"', missing_pkgs), collapse = ", "), "))"
    )
    stop(msg, call. = FALSE)
  }
  invisible(TRUE)
}


# ----------------------------------------------------------------------------
# Default coefficients: one place to adjust the DGP
# ----------------------------------------------------------------------------
# Every coefficient used by the DGP lives here so callers can inspect or
# override without hunting through function bodies.

#' Return the default DGP coefficients as a nested list.
#' @return list with elements: baseline, prop (per stage), outcome (per stage),
#'         identity_update, cumulative, levels.
default_coefs <- function() {
  list(

    # -- baseline structural noise ----------------------------------------
    # ident_on_math = 0.10 (weaker than the ~0.30 one might expect). The
    # weak linkage between pre_hs_math and math_identity_g9 widens the
    # "well-prepared but low-identity" cohort. That's the cohort for whom
    # APCalc is structurally optimal (via prehs_top boost) yet Alg1 is the
    # myopic G9 choice, so the feasibility chain blocks them under
    # constraints. This is what produces the V_unc - V_con gap.
    baseline = list(
      ses_on_math   = 0.40,           # cor(pre_hs_math, student_ses) target
      male_prob     = 0.50,
      ident_on_math = 0.10,
      ident_on_ses  = 0.20,
      ident_on_male = 0.10,
      ident_noise_sd = 0.965          # adjusted so marginal Var(identity) ≈ 1
    ),

    # -- number of feasible levels at each stage --------------------------
    levels = list(g9 = 4L, g10 = 5L, g11 = 6L, g12 = 7L),

    # -- propensity (softmax) utilities -----------------------------------
    # V2.1_A_Redesign level taxonomy:
    #   1: BasicMath, 2: Algebra 1, 3: Geometry, 4: Algebra 2,
    #   5: AdvancedMath (terminal advanced -- e.g. trig/stats),
    #   6: PreCalculus, 7: Calculus.
    # PreCalc was level 5 in V2.0; Calc was level 6; APCalc (level 7)
    # is consolidated into Calc. AdvancedMath is a NEW middle-level
    # course filling a real role in HSLS taxonomy.
    #
    # Course availability per stage:
    #   G9 : 1-4 (no advanced)
    #   G10: 1-5 (+ AdvancedMath; needs Alg2 at G9)
    #   G11: 1-6 (+ PreCalc; PreCalc/AdvMath need Alg2 or AdvMath at G10)
    #   G12: 1-7 (+ Calc; Calc needs PreCalc at G11)
    #
    # G9 intercepts unchanged (HSLS Fig-1 target). G10/G11/G12 use
    # advance-biased shifts plus HARD prereq enforcement: shifts of
    # -2.5 push infeasible-prereq cells well below cut-off, leaving
    # ~0.05-0.3% violations -- low enough that the constrained
    # estimator clearly wins, high enough to keep TMLE's positivity
    # well-defined.
    #
    # Senior-slump shift retained at G12 for students who fulfilled
    # the Alg-II state requirement by Grade 11.
    prop = list(

      g9 = list(
        alpha = c(-2.296, 1.173, 0.413, -1.667),
        beta_prehs   = c(-2.0, -0.3,  0.8,  2.0),
        beta_ses     = c(-0.5, -0.2,  0.3,  0.5),
        beta_ident   = c(-0.3,  0.1,  0.4,  0.6)
      ),

      g10 = list(
        # 5 levels: BasicMath, Alg1, Geom, Alg2, AdvancedMath.
        # AdvMath alpha very negative; only g9=Alg2 students access it.
        alpha        = c( 0.4, 0.8, 0.3, -0.5, -3.0),
        beta_prehs   = c(-0.3,-0.3, 0.6,  1.2,  1.5),
        beta_ses     = c( 0.0, 0.0, 0.3,  0.4,  0.0),
        beta_ident   = c( 0.0, 0.1, 0.3,  0.4,  0.5),
        shift_by_prev = list(
          "1" = c( 1.5, 2.5, 0.0, 0.0, -2.5),  # BasicMath -> Alg1; AdvMath blocked
          "2" = c(-1.5, 0.5, 2.5, 0.5, -2.5),  # Alg1 -> Geom (modal); AdvMath blocked
          "3" = c(-2.5,-1.5, 0.5, 3.0, -2.5),  # Geom -> Alg2; AdvMath still blocked
          "4" = c(-3.0,-2.0,-1.0, 0.5,  4.5)   # Alg2 -> AdvMath (modal advance)
        )
      ),

      g11 = list(
        # 6 levels: BasicMath, Alg1, Geom, Alg2, AdvMath, PreCalc.
        # AdvMath/PreCalc alpha very negative; only g10 in {Alg2, AdvMath}
        # students access them.
        alpha        = c( 0.4, 0.6, 0.3, -0.3, -3.0, -3.0),
        beta_prehs   = c(-0.3,-0.3, 0.5,  1.0,  1.3,  2.0),
        beta_ses     = c( 0.0, 0.0, 0.3,  0.4,  0.4,  0.0),
        beta_ident   = c( 0.0, 0.1, 0.3,  0.4,  0.4,  0.5),
        shift_by_prev_g10 = list(
          "1" = c( 1.5, 2.5, 0.0, 0.0, -2.5, -2.5),
          "2" = c(-1.5, 0.5, 2.5, 0.5, -2.5, -2.5),
          "3" = c(-2.5,-1.5, 0.5, 3.0, -2.5, -2.5),
          # Alg2 at G10: modal PreCalc, some AdvMath.
          "4" = c(-3.0,-2.0,-1.0, 0.5,  2.5,  4.0),
          # AdvMath at G10: modal PreCalc, some retake AdvMath.
          "5" = c(-3.0,-2.0,-1.5, 0.0,  1.5,  4.0)
        )
      ),

      g12 = list(
        # 7 levels: BasicMath, Alg1, Geom, Alg2, AdvMath, PreCalc, Calc.
        # Calc alpha very negative; only g11=PreCalc students access it.
        alpha        = c( 0.4, 0.6, 0.3, -0.3, -3.0, -2.5, -3.5),
        beta_prehs   = c(-0.3,-0.3, 0.4,  0.9,  1.2,  1.5,  2.0),
        beta_ses     = c( 0.0, 0.0, 0.3,  0.4,  0.4,  0.4,  0.0),
        beta_ident   = c( 0.0, 0.1, 0.3,  0.4,  0.4,  0.4,  0.5),
        shift_by_prev_g11 = list(
          "1" = c( 1.5, 2.5, 0.0, 0.0, -2.0, -2.5, -2.5),
          "2" = c(-1.5, 0.5, 2.5, 0.5, -2.0, -2.5, -2.5),
          "3" = c(-2.5,-1.5, 0.5, 3.0, -2.0, -2.5, -2.5),
          # Alg2 at G11: STRONGLY modal PreCalc, some AdvMath, modest slump.
          # Bumped PreCalc shift +4.0 -> +4.5 and slump +0.3 -> 0.0 so
          # PreCalc is the clear G12 modal per the V2.1_A_Redesign spec.
          "4" = c( 0.0,-2.0,-1.0, 0.5,  3.0,  4.5, -2.5),
          # AdvMath at G11: modal PreCalc, some retake; ~25% senior slump.
          "5" = c(-0.5,-2.0,-1.5,-0.5,  1.5,  3.0, -2.5),
          # PreCalc at G11: modal Calc; small senior slump.
          "6" = c(-2.0,-2.0,-2.0,-1.5, -1.0,  0.5,  4.5)
        )
      )
    ),

    # -- outcome (raw stage-k GPA) ----------------------------------------
    # V2.1_A_Redesign outcome model:
    #   - main_g12[6] = 0.35 (PreCalc): largest "structural reward" for
    #     the modal college-prep terminal course.
    #   - main_g12[7] = 0.10 (Calc): modest main, compensated by
    #     prehs_top[7] = 0.40 -- Calc is the STEM apex, structurally
    #     optimal only for well-prepared students.
    #   - main_g12[5] = 0.25 (AdvMath): respectable terminal but below
    #     PreCalc; an attractive "off-STEM" option.
    #   - prehs_top[6] = 0.10 (PreCalc): small reward for prepared
    #     students, since PreCalc is genuinely useful.
    #   - prehs_top[7] = 0.40 (Calc): the V_unc - V_con gap engine,
    #     unchanged from V2.0's 0.40 calibration.
    # No prehs_top at G10 / G11 -- preserves the "skip-cohort" design
    # where the unconstrained learner wants Calc-via-PreCalc-elsewhere
    # for some students whose constrained path is blocked.
    outcome = list(

      intercept       = 2.00,
      beta_prehs      = 0.35,
      beta_ses        = 0.15,
      beta_male       = 0.05,
      beta_ident      = 0.20,
      mismatch_scalar = -0.50,          # shared across stages
      readiness_prehs = 0.60,           # weights inside readiness = pnorm(...)
      readiness_ident = 0.40,
      noise_sd        = 0.35,

      g9 = list(
        main     = c(-0.15, 0.10, 0.25, 0.15),
        interact = c(-0.20, 0.00, 0.15, 0.35),
        advance  = c( 0.00, 0.33, 0.67, 1.00)
      ),
      g10 = list(
        # 5 levels; AdvMath similar payoff to Alg2.
        main      = c(-0.20, 0.00, 0.15, 0.25, 0.20),
        interact  = c(-0.25,-0.05, 0.10, 0.30, 0.40),
        advance   = c( 0.00, 0.25, 0.50, 0.75, 1.00),
        prehs_top = c( 0.00, 0.00, 0.00, 0.00, 0.00)
      ),
      g11 = list(
        # 6 levels; PreCalc highest main, AdvMath modest above Alg2.
        main      = c(-0.20,-0.05, 0.10, 0.20, 0.25, 0.30),
        interact  = c(-0.25,-0.10, 0.05, 0.20, 0.35, 0.45),
        advance   = c( 0.00, 0.20, 0.40, 0.60, 0.80, 1.00),
        prehs_top = c( 0.00, 0.00, 0.00, 0.00, 0.00, 0.00)
      ),
      g12 = list(
        # 7 levels; PreCalc=highest main, Calc=lower main + biggest
        # prehs_top. AdvMath sits between Alg2 and PreCalc.
        #
        # CALIBRATION NOTES:
        # - prehs_top boost ONLY on Calc (level 7). Two empirical tries
        #   showed an earlier 0.10 boost on PreCalc cannibalised the
        #   V_unc - V_con gap; remove restored V2.0-style behaviour.
        # - prehs_top[7] = 0.55 (vs V2.0's 0.40). Bumped because the
        #   V2.1 constrained alternative is PreCalc (main=0.35) rather
        #   than V2.0's Calc-as-fallback (main=0.30); the +0.05 in
        #   constrained reward had to be offset by giving Calc a
        #   stronger prehs lever, otherwise the V_unc - V_con gap
        #   shrank to ~0.001.
        main      = c(-0.25,-0.10, 0.05, 0.15, 0.25, 0.35, 0.10),
        interact  = c(-0.30,-0.15, 0.00, 0.15, 0.30, 0.45, 0.55),
        advance   = c( 0.00, 0.17, 0.33, 0.50, 0.67, 0.83, 1.00),
        prehs_top = c( 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.55)
      )
    ),

    # -- identity update --------------------------------------------------
    identity_update = list(
      persist        = 0.70,
      gpa_boost      = 0.20,
      gpa_center     = 2.50,
      course_boost   = 0.10,
      noise_sd       = 0.30,
      clip_lo        = -3,
      clip_hi        =  3,
      # per-stage boost vectors. Fraction of advance so higher courses
      # build more identity.
      # V2.1_A_Redesign: AdvMath at G10/G11/G12 boosts identity similarly
      # to PreCalc (it's a real advanced course); Calc at G12 is the apex.
      boost_g9  = c(-0.30, 0.00, 0.20, 0.40),
      boost_g10 = c(-0.30,-0.10, 0.10, 0.30, 0.45),
      boost_g11 = c(-0.30,-0.10, 0.10, 0.25, 0.40, 0.55),
      boost_g12 = c(-0.30,-0.10, 0.10, 0.20, 0.35, 0.50, 0.65)
    ),

    # -- cumulative GPA weighting -----------------------------------------
    cumulative = list(
      weighted   = TRUE,
      slope      = 0.15,   # weight = 1 + slope * (integer_level - 1)
      clip_lo    = 0,
      clip_hi    = 4
    )
  )
}


# ----------------------------------------------------------------------------
# Baseline generator
# ----------------------------------------------------------------------------

#' Generate baseline covariates and Grade-9 math identity.
#'
#' @param n integer sample size.
#' @param coefs list from default_coefs(); uses coefs$baseline.
#' @param seed optional integer seed; if non-NULL, set.seed(seed) locally.
#' @return data.frame with columns
#'         pre_hs_math, student_ses, student_male, math_identity_g9.
make_baseline <- function(n, coefs, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  p <- coefs$baseline

  z1 <- rnorm(n)
  z2 <- rnorm(n)
  z3 <- rnorm(n)

  pre_hs_math  <- z1
  # cor(pre_hs_math, student_ses) -> p$ses_on_math
  student_ses  <- p$ses_on_math * z1 + sqrt(1 - p$ses_on_math^2) * z2
  student_male <- as.integer(runif(n) < p$male_prob)

  math_identity_g9 <- p$ident_on_math * pre_hs_math +
                      p$ident_on_ses  * student_ses +
                      p$ident_on_male * student_male +
                      p$ident_noise_sd * z3

  data.frame(
    pre_hs_math      = pre_hs_math,
    student_ses      = student_ses,
    student_male     = student_male,
    math_identity_g9 = math_identity_g9
  )
}


# ----------------------------------------------------------------------------
# Utility and propensity construction (stage-wise)
# ----------------------------------------------------------------------------

#' Softmax over rows of a matrix (numerically stable).
#' @param U n x J matrix of utilities.
#' @return n x J matrix of probabilities, each row sums to 1.
row_softmax <- function(U) {
  m <- apply(U, 1, max)
  E <- exp(U - m)
  E / rowSums(E)
}

#' Build the utility matrix at Grade 9.
#' @param df data.frame containing pre_hs_math, student_ses, math_identity_g9.
#' @param coefs default_coefs() list.
#' @return n x 4 utility matrix.
utilities_g9 <- function(df, coefs) {
  p <- coefs$prop$g9
  # vectorized outer construction
  n <- nrow(df)
  J <- length(p$alpha)
  U <- matrix(0, n, J)
  for (j in seq_len(J)) {
    U[, j] <- p$alpha[j] +
              p$beta_prehs[j] * df$pre_hs_math +
              p$beta_ses[j]   * df$student_ses +
              p$beta_ident[j] * df$math_identity_g9
  }
  U
}

#' Generic utility constructor for stages >= 10.
#' @param df data.frame with current math_identity and pre_hs_math / ses.
#' @param prev_course integer vector, previous stage's treatment (1-based).
#' @param p list: one of coefs$prop$g10, g11, g12.
#' @param prev_key_name name of the shift list in `p` (e.g. "shift_by_prev",
#'        "shift_by_prev_g10", "shift_by_prev_g11").
#' @param ident_name column name of the moderator in df.
#' @return n x J utility matrix.
utilities_later <- function(df, prev_course, p, prev_key_name, ident_name) {
  n <- nrow(df)
  J <- length(p$alpha)
  U <- matrix(0, n, J)
  ident_vec <- df[[ident_name]]
  for (j in seq_len(J)) {
    U[, j] <- p$alpha[j] +
              p$beta_prehs[j] * df$pre_hs_math +
              p$beta_ses[j]   * df$student_ses +
              p$beta_ident[j] * ident_vec
  }
  # add previous-course-dependent shifts
  shifts <- p[[prev_key_name]]
  prev_chr <- as.character(prev_course)
  unique_keys <- names(shifts)
  for (k in unique_keys) {
    mask <- prev_chr == k
    if (any(mask)) {
      U[mask, ] <- U[mask, , drop = FALSE] +
                   matrix(shifts[[k]], sum(mask), J, byrow = TRUE)
    }
  }
  U
}


# ----------------------------------------------------------------------------
# Treatment sampler (softmax -> integer 1..J)
# ----------------------------------------------------------------------------

#' Sample one treatment per row from a probability matrix.
#' @param P n x J probability matrix (rows sum to 1).
#' @param seed optional integer seed.
#' @return integer vector length n, values in 1..J.
sample_treatment <- function(P, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(P); J <- ncol(P)
  u <- runif(n)
  cumP <- t(apply(P, 1, cumsum))
  # which cumulative bucket does u fall into?
  # vectorized: rowSums(cumP < u) + 1 gives 1..J
  as.integer(rowSums(cumP < u) + 1L)
}


# ----------------------------------------------------------------------------
# Stage-k raw GPA (structural expectation + noise)
# ----------------------------------------------------------------------------

#' Compute the MEAN (expectation) of raw stage-k GPA given a treatment vector.
#' Used both for outcome simulation and for closed-form ground-truth blips.
#' @param df data.frame with pre_hs_math, student_ses, student_male, and the
#'        stage's math_identity column.
#' @param course integer vector of treatments (1..J).
#' @param stage one of "g9","g10","g11","g12".
#' @param coefs default_coefs() list.
#' @param ident_name name of the identity column (e.g. "math_identity_g9").
#' @return numeric vector length n, unclipped expectation.
raw_gpa_mean <- function(df, course, stage, coefs, ident_name) {
  o    <- coefs$outcome
  sk   <- o[[stage]]
  ident_vec <- df[[ident_name]]

  readiness <- pnorm(o$readiness_prehs * df$pre_hs_math +
                     o$readiness_ident * ident_vec)

  main     <- sk$main[course]
  interact <- sk$interact[course]
  advance  <- sk$advance[course]
  mismatch <- o$mismatch_scalar * pmax(0, advance - readiness)^2

  # Option-B coefficient: prehs_top_boost is a pre_hs_math-scaled bonus
  # applied only on top-level courses at each later stage. Decouples
  # "top course is structurally optimal" from "identity is high" — so a
  # well-prepared but low-identity student benefits structurally from
  # APCalc, yet can only reach it by skipping the prereq chain.
  # G9 has no prehs_top entry; coalesce to zero.
  prehs_top_vec <- if (is.null(sk$prehs_top)) rep(0, length(sk$main)) else sk$prehs_top
  prehs_top_bonus <- prehs_top_vec[course] * df$pre_hs_math

  o$intercept +
    o$beta_prehs * df$pre_hs_math +
    o$beta_ses   * df$student_ses +
    o$beta_male  * df$student_male +
    o$beta_ident * ident_vec +
    main +
    interact * ident_vec +
    mismatch +
    prehs_top_bonus
}

#' Simulate raw stage-k GPA (mean + noise, clipped to [0,4]).
#' @inheritParams raw_gpa_mean
#' @param seed optional integer seed.
#' @return numeric vector length n, in [0,4].
simulate_raw_gpa <- function(df, course, stage, coefs, ident_name, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  mu <- raw_gpa_mean(df, course, stage, coefs, ident_name)
  eps <- rnorm(length(mu), 0, coefs$outcome$noise_sd)
  pmin(pmax(mu + eps, 0), 4)
}


# ----------------------------------------------------------------------------
# Identity transition
# ----------------------------------------------------------------------------

#' One-step identity update.
#' @param prev_ident numeric vector, identity at stage k.
#' @param gpa_raw numeric vector, raw stage-k GPA (NOT cumulative).
#' @param course integer vector, stage-k treatment.
#' @param stage source stage (the one we are leaving): "g9","g10","g11".
#'        Determines which boost vector is used.
#' @param coefs default_coefs() list.
#' @param seed optional integer seed.
#' @return numeric vector, identity at stage k+1 (clipped to [-3, 3]).
identity_update <- function(prev_ident, gpa_raw, course, stage, coefs, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  iu <- coefs$identity_update
  boost_name <- paste0("boost_", stage)
  boost_vec  <- iu[[boost_name]]
  boost      <- boost_vec[course]
  eps <- rnorm(length(prev_ident), 0, iu$noise_sd)
  out <- iu$persist  * prev_ident +
         iu$gpa_boost * (gpa_raw - iu$gpa_center) +
         iu$course_boost * boost +
         eps
  pmin(pmax(out, iu$clip_lo), iu$clip_hi)
}


# ----------------------------------------------------------------------------
# Cumulative GPA (clipped weighted average of raw stage GPAs)
# ----------------------------------------------------------------------------

#' Cumulative GPA across stages 1..k.
#' @param raw_stages n x k numeric matrix of raw stage GPAs.
#' @param levels_stages n x k integer matrix of course levels taken.
#' @param coefs default_coefs() list. Uses coefs$cumulative.
#' @param weighted logical override; defaults to coefs$cumulative$weighted.
#' @return numeric vector length n, in [0, 4].
cumulative_gpa <- function(raw_stages, levels_stages, coefs, weighted = NULL) {
  stopifnot(is.matrix(raw_stages), is.matrix(levels_stages),
            all(dim(raw_stages) == dim(levels_stages)))
  cc <- coefs$cumulative
  if (is.null(weighted)) weighted <- cc$weighted
  if (weighted) {
    W <- 1 + cc$slope * (levels_stages - 1)       # BasicMath=1 ... APCalc=1.9
  } else {
    W <- matrix(1, nrow(raw_stages), ncol(raw_stages))
  }
  num <- rowSums(raw_stages * W)
  den <- rowSums(W)
  pmin(pmax(num / den, cc$clip_lo), cc$clip_hi)
}


# ----------------------------------------------------------------------------
# A priori feasibility
# ----------------------------------------------------------------------------
# NOTE: this is NOT called by the simulator. The simulator allows rare
# violations (tiny but nonzero propensity on infeasible cells) so TMLE has
# something worth trimming. `a_priori_feasible` is used by:
#   (a) the constrained optimal-rule search in the ground-truth block, and
#   (b) verification checks.

#' Return the a priori feasible treatment levels for one student at one stage.
#'
#' V2.1_A_Redesign rules:
#'   G9 : 1-4 unrestricted.
#'   G10: levels 1-4 unrestricted; level 5 (AdvancedMath) requires
#'        course_g9 == 4 (Algebra 2).
#'   G11: levels 1-4 unrestricted; levels 5 (AdvMath) and 6 (PreCalc)
#'        require course_g10 in {4, 5} (Alg 2 or AdvMath at G10).
#'   G12: levels 1-6 unrestricted; level 7 (Calculus) requires
#'        course_g11 == 6 (PreCalc was taken at G11). PreCalc is not
#'        available at G10, so the only way to satisfy the prereq is via
#'        G11 PreCalc.
#'
#' @param history named list / named numeric vector with elements
#'        course_g9, course_g10, course_g11 (as far as they are known).
#' @param stage one of "g9","g10","g11","g12".
#' @return integer vector of feasible levels.
a_priori_feasible <- function(history, stage) {
  switch(stage,
    g9 = 1:4,
    g10 = {
      cg9 <- history[["course_g9"]]
      base <- 1:4
      if (!is.null(cg9) && cg9 == 4L) c(base, 5L) else base
    },
    g11 = {
      cg10 <- history[["course_g10"]]
      base <- 1:4
      adv_open <- !is.null(cg10) && cg10 %in% c(4L, 5L)
      if (adv_open) c(base, 5L, 6L) else base
    },
    g12 = {
      cg10 <- history[["course_g10"]]
      cg11 <- history[["course_g11"]]
      base <- 1:4
      # AdvMath / PreCalc at G12 still need Alg-2 in history (either G10
      # or G11). The propensity model already softly enforces this, but
      # making it a HARD rule keeps the constrained DP from rescuing
      # students whose chain skipped Alg-2 -- which is essential to the
      # V_unc - V_con gap mechanism.
      alg2_history <- (!is.null(cg10) && cg10 >= 4L) ||
                      (!is.null(cg11) && cg11 >= 4L)
      if (alg2_history) base <- c(base, 5L, 6L)
      precalc_taken <- (!is.null(cg11) && cg11 == 6L)
      if (precalc_taken) base <- c(base, 7L)
      base
    },
    stop("unknown stage: ", stage)
  )
}

#' Vectorized feasibility: n x J_max logical matrix (TRUE if feasible).
#'
#' V2.1_A_Redesign rules; see `a_priori_feasible()` for details.
#'
#' @param course_g9,course_g10,course_g11 integer vectors (pass NULL when
#'        not yet observed).
#' @param stage one of "g9","g10","g11","g12".
#' @param J_max integer, total number of levels at that stage.
#' @return n x J_max logical matrix.
a_priori_feasible_mat <- function(course_g9 = NULL, course_g10 = NULL,
                                  course_g11 = NULL, stage, J_max) {
  n <- length(c(course_g9, course_g10, course_g11)) /
       sum(c(!is.null(course_g9), !is.null(course_g10), !is.null(course_g11)))
  n <- as.integer(n)
  M <- matrix(TRUE, n, J_max)
  if (stage == "g9") return(M)
  if (stage == "g10") {
    M[, 5] <- (course_g9 == 4L)             # AdvMath needs Alg2 at G9
    return(M)
  }
  if (stage == "g11") {
    open_adv <- course_g10 %in% c(4L, 5L)
    M[, 5] <- open_adv                      # AdvMath needs Alg2/AdvMath at G10
    M[, 6] <- open_adv                      # PreCalc needs Alg2/AdvMath at G10
    return(M)
  }
  if (stage == "g12") {
    alg2_hist <- (course_g10 >= 4L) | (course_g11 >= 4L)
    M[, 5] <- alg2_hist                      # AdvMath needs Alg-2 in history
    M[, 6] <- alg2_hist                      # PreCalc needs Alg-2 in history
    M[, 7] <- (course_g11 == 6L)             # Calc needs PreCalc at G11
    return(M)
  }
  stop("unknown stage: ", stage)
}
