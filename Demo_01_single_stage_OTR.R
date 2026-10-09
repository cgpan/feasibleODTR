# date: 2026.04.19
# purpose:
#   Single-stage (G9-only) ODTR baseline against the multi-stage learner.
#   Fits ONE TMLE blip-revere model on the G9 outcome (math_gpa_g9) and
#   then derives TWO learned rules from that fit:
#
#     UNCONSTRAINED  : argmax of the blip with no trimming.
#     CONSTRAINED    : PS-trim at PS_CUTOFF (0.05) + preHS-based a-priori
#                      constraint mimicking real practice -- students with
#                      strong preHS scores (> +1.5 sd) cannot pick the two
#                      lowest-level math (1,2); students with weak preHS
#                      scores (< -1.5 sd) cannot pick the two highest-level
#                      math (3,4).
#
#   Both rules' values are TMLE-updated via Optimizer$new (single-stage
#   pooled update from Updating_Fun_Single_Stage.R). Final workspace image
#   is consumed by 05_single_stage_evaluation.R.
#
# Inputs:
#   ../data/synthetic_odtr_n5000.csv
#   253_D_TMLE_functions.R, 286_TMLE_flexible_using_GLM.R,
#   ODTR_Functions_V1.0.0.R, Updating_Fun_Single_Stage.R
#
# Outputs:
#   ../data/tmle_results/03_single_stage_g9_20260418.rdata


# -----------------------------------------------------------------
# 0. Packages, paths, parallel plan
# -----------------------------------------------------------------
library(fastDummies)
library(data.table)
library(dplyr)
library(tmle3)
library(sl3)
library(tlverse)
library(tmle3mopttx)
library(devtools)
library(caret)
library(purrr)
library(future)
library(future.apply)
library(origami)


getwd()

# Use parallel computing to speed up.
# The fitting process may take 4-5 minutes since it includes CV and multiple ML model fittings.
options(future.globals.maxSize = 8 * 1024^3)
plan(multisession, workers = max(1, parallel::detectCores() - 2))


source("ODTR_Functions_V1.0.0.R")
source("Updating_Fun_Single_Stage.R")

time_start <- Sys.time()
# -----------------------------------------------------------------
# 1. Config (the only knobs)
# -----------------------------------------------------------------
PS_CUTOFF        <- 0.03            # single-stage feasibility cutoff
#PS_CUTOFF         <- 0
PREHS_CUTOFFS    <- c(-1.5, 1.5)    # weak/strong preHS sd thresholds
#PREHS_CUTOFFS  <- c(-Inf, Inf)
RUN_DATE_TAG     <- "20260424"
SAMPLE_SIZE      <- 10000
OUT_RDATA        <- file.path("data", "tmle_results",
                              paste0("Demo_01_single_stage_g9_", RUN_DATE_TAG, "_", PS_CUTOFF,"_N_", SAMPLE_SIZE,
                                     ".rdata"))


# -----------------------------------------------------------------
# 2. Load + prepare data
# -----------------------------------------------------------------
df_sim <- read.csv("data/synthetic_odtr_n10000.csv")
df_sim$course_g9 <- factor(df_sim$course_g9, levels = as.character(1:4))

# Stable row id (so re-orderings are recoverable downstream).
df_sim$.row_id <- seq_len(nrow(df_sim))

# Global CV folds, fixed for reproducibility.
set.seed(20260105)
V <- 10
folds_global   <- make_folds(n = nrow(df_sim), fold_fun = folds_vfold, V = V)
foldvec_global <- folds2foldvec(folds_global)

make_stage_folds <- function(stage_df, foldvec_global, V) {
  fv <- foldvec_global[stage_df$.row_id]
  lapply(seq_len(V), function(v) fold_from_foldvec(v, fv))
}
folds_fixed <- make_stage_folds(df_sim, foldvec_global, V)


# -----------------------------------------------------------------
# 3. Super Learner libraries (match 01_multi_stage_MA.R)
# -----------------------------------------------------------------
lrn_xgboost_50  <- Lrnr_xgboost$new(nrounds = 50)
lrn_xgboost_100 <- Lrnr_xgboost$new(nrounds = 100)
lrn_xgboost_500 <- Lrnr_xgboost$new(nrounds = 500)
lrn_mean        <- Lrnr_mean$new()
lrn_glm         <- Lrnr_glm_fast$new()
lrn_lasso       <- Lrnr_glmnet$new(alpha = 1)

Q_learner <- Lrnr_sl$new(
  learners    = list(lrn_xgboost_50, lrn_xgboost_100, lrn_xgboost_500,
                     lrn_mean, lrn_glm, lrn_lasso),
  metalearner = Lrnr_nnls$new()
)

mn_metalearner <- make_learner(
  Lrnr_solnp,
  eval_function    = loss_loglik_multinomial,
  learner_function = metalearner_linear_multinomial
)

g_learner <- make_learner(
  Lrnr_sl,
  list(lrn_xgboost_50, lrn_xgboost_100, lrn_xgboost_500,
       Lrnr_glmnet$new(),
       Lrnr_ranger$new(mtry = 2, min.node.size = 5,
                       num.trees = 100, classification = TRUE)),
  mn_metalearner
)

learners_blip <- list(lrn_xgboost_50, lrn_xgboost_100, lrn_xgboost_500, lrn_glm)
b_learner     <- create_mv_learners(learners = learners_blip)

learner_list <- list(Y = Q_learner, A = g_learner, B = b_learner)


# -----------------------------------------------------------------
# 4. TMLE blip-revere fit (single G9 fit -- both rules derive from it)
# -----------------------------------------------------------------
node_list_09 <- list(
  W = c("pre_hs_math", "student_ses", "student_male", "math_identity_g9"),
  A = "course_g9",
  Y = "math_gpa_g9"
)

tmle_spec_g9 <- tmle3_mopttx_blip_revere(
  V        = c("pre_hs_math", "student_ses", "student_male", "math_identity_g9"),
  type     = "blip1",
  learners = learner_list,
  maximize = TRUE,
  complex  = TRUE,
  realistic = FALSE,
  resource = 1,
  interpret = FALSE
)

tmle_task_g9 <- tmle_spec_g9$make_tmle_task(
  data      = df_sim,
  node_list = node_list_09,
  folds     = folds_fixed
)

# Same retry-with-incrementing-seed pattern as 01_multi_stage_MA.R --
# the SL fits sometimes hit numerical issues that go away with a different
# seed. Cap at 50 attempts.
fit_success  <- FALSE
attempts     <- 0
max_attempts <- 50
seed         <- 15L

time_fit_start <- Sys.time()
while (!fit_success && attempts < max_attempts) {
  attempts <- attempts + 1L
  seed     <- seed + 1L
  set.seed(seed)
  tryCatch({
    initial_likelihood <- tmle_spec_g9$make_initial_likelihood(
      tmle_task_g9, learner_list = learner_list
    )
    updater <- tmle_spec_g9$make_updater()
    targeted_likelihood <- tmle_spec_g9$make_targeted_likelihood(
      initial_likelihood, updater
    )
    tmle_params <- tmle_spec_g9$make_params(
      tmle_task_g9, likelihood = targeted_likelihood
    )
    updater$tmle_params <- tmle_params
    fit_g9 <- fit_tmle3(tmle_task_g9, targeted_likelihood,
                        tmle_params, updater)
    fit_success <- TRUE
  }, error = function(e) {
    message("attempt ", attempts, " failed: ", conditionMessage(e))
  })
}
if (!fit_success) stop("TMLE blip-revere fit failed after ", max_attempts, " attempts.")
time_fit_end <- Sys.time()
cat("\nTMLE fit succeeded on attempt", attempts,
    "(seed =", seed, "); fit time =",
    round(as.numeric(difftime(time_fit_end, time_fit_start, units = "mins")), 2),
    "min\n")


# -----------------------------------------------------------------
# 5. Extract TMLE components (CV blips, CV PS, no-CV PS, CFs)
# -----------------------------------------------------------------
results_g9 <- extract_tmle_components(
  node_list     = node_list_09,
  tmle_fit      = fit_g9,
  tmle_spec     = tmle_spec_g9,
  tmle_task     = tmle_task_g9,
  original_data = df_sim
)

blip_df1     <- results_g9$blip_df       # n x 4 CV blips
ps_df1       <- results_g9$ps_df         # n x 4 CV PS
ps_noCV_df1  <- results_g9$ps_noCV_df    # n x 4 no-CV PS (used for trimming)
cf_df1       <- results_g9$cf_df         # n x 4 CV counterfactual outcomes
QAW          <- results_g9$QAW           # n CV outcome at observed A
Y_min        <- results_g9$Y_min
Y_max        <- results_g9$Y_max


# -----------------------------------------------------------------
# 6. Derive learned rules
# -----------------------------------------------------------------

# 6a. UNCONSTRAINED: raw blip argmax.
OTR_uncon_g9 <- apply(blip_df1, 1, which.max)
cat("\nUnconstrained learned G9 rule -- marginal:\n")
print(table(OTR_uncon_g9))

# 6b. CONSTRAINED: copy of blip with PS<cutoff cells set to -Inf, then the
# preHS-based a-priori knockouts. NOTE: ps_noCV_df1 (uncrossfitted PS) is
# the trimming target -- consistent with the multi-stage scripts which
# use ps_noCV_df{k} for the feasibility step and ps_df{k} (CV) only for
# value updating.
blip_df1_feas <- blip_df1
blip_df1_feas[ps_noCV_df1 < PS_CUTOFF] <- -Inf

# preHS-based a-priori:
#   weak  preHS (<= -1.5): cannot pick high-level math (3,4)
#   strong preHS (>= +1.5): cannot pick low-level math (1,2)
weak_mask   <- df_sim$pre_hs_math < PREHS_CUTOFFS[1]
strong_mask <- df_sim$pre_hs_math > PREHS_CUTOFFS[2]
blip_df1_feas[weak_mask,   c(3L, 4L)] <- -Inf
blip_df1_feas[strong_mask, c(1L, 2L)] <- -Inf

n_weak   <- sum(weak_mask)
n_strong <- sum(strong_mask)
cat(sprintf("\npreHS knockouts: %d weak students lose levels {3,4}; %d strong students lose levels {1,2}\n",
            n_weak, n_strong))

# Sanity: a tiny number of students may have ALL four cells = -Inf if PS
# trimming wipes their {1,2} (or {3,4}) tail and the preHS knockout hits
# the other tail. Fall back to the unconstrained argmax for them.
all_inf_idx <- apply(blip_df1_feas, 1, function(x) all(is.infinite(x)))
n_all_inf <- sum(all_inf_idx)
if (n_all_inf > 0) {
  cat(sprintf("Note: %d students had all 4 cells trimmed; falling back to unconstrained argmax for them.\n",
              n_all_inf))
  blip_df1_feas[all_inf_idx, ] <- blip_df1[all_inf_idx, ]
}
OTR_con_g9 <- apply(blip_df1_feas, 1, which.max)
cat("\nConstrained learned G9 rule -- marginal:\n")
print(table(OTR_con_g9))

# Cross-tab unc vs con (for diagnostics).
cat("\nUnconstrained vs constrained (rows = unc, cols = con):\n")
print(addmargins(table(OTR_uncon_g9, OTR_con_g9)))


# -----------------------------------------------------------------
# 7. TMLE value updating (single-stage) for both rules
# -----------------------------------------------------------------
A1_int      <- as.integer(as.character(df_sim$course_g9))
y_obs_g9    <- df_sim$math_gpa_g9
y_min_used  <- min(y_obs_g9)
y_max_used  <- max(y_obs_g9)

run_updater <- function(otr_rule) {
  Qd <- cf_df1[cbind(seq_len(nrow(cf_df1)), otr_rule)]
  opt <- Optimizer$new(
    treat_obs         = A1_int,
    y_obs             = y_obs_g9,
    y_pre             = QAW,
    y_global_min      = y_min_used,
    y_global_max      = y_max_used,
    propensity_matrix = as.matrix(ps_df1),
    OTR_pre           = otr_rule,
    values_est_init   = Qd
  )
  opt$update()
}

fit_unc <- run_updater(OTR_uncon_g9)
fit_con <- run_updater(OTR_con_g9)

psi_unc_g9 <- fit_unc$psi_hat;   se_unc_g9 <- fit_unc$se
psi_con_g9 <- fit_con$psi_hat;   se_con_g9 <- fit_con$se

cat("\n--- TMLE value estimates (single-stage G9) ---\n")
cat(sprintf("  Unconstrained: psi_hat = %.4f, SE = %.4f, 95%% CI = [%.4f, %.4f]\n",
            psi_unc_g9, se_unc_g9,
            psi_unc_g9 - 1.96 * se_unc_g9, psi_unc_g9 + 1.96 * se_unc_g9))
cat(sprintf("  Constrained  : psi_hat = %.4f, SE = %.4f, 95%% CI = [%.4f, %.4f]\n",
            psi_con_g9, se_con_g9,
            psi_con_g9 - 1.96 * se_con_g9, psi_con_g9 + 1.96 * se_con_g9))
cat(sprintf("  Observed mean math_gpa_g9 = %.4f\n", mean(y_obs_g9)))


# -----------------------------------------------------------------
# 8. Save workspace image for 05_single_stage_evaluation.R
# -----------------------------------------------------------------
dir.create(dirname(OUT_RDATA), showWarnings = FALSE, recursive = TRUE)
#save.image(file = OUT_RDATA)
cat("\nSaved single-stage workspace image to:\n  ", OUT_RDATA, "\n", sep = "")

time_end <- Sys.time()

time_end - time_start


# extract the saved info
table(OTR_con_g9)
table(df_sim$course_g9)
hist(df_sim$pre_hs_math)

df_uncon <- data.frame(
  A = as.integer(df_sim$course_g9),
  pre_hs_math = df_sim$pre_hs_math,
  OTR_uncon = OTR_uncon_g9
)
# save the df_uncon as rdata

save(df_uncon, file = "data/03_single_OTR_uncon.rdata")

