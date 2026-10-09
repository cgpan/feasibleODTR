# date: 2026.04.22  (V2a)
# purpose:
#   1. Multi-stage ODTR model on the synthetic dataset WITH three feasibility
#      constraints applied during backward induction:
#        (a) PS-based trimming at each stage (using the non-CV PS matrix).
#        (b) A priori prerequisite chain (V2.1_A_Redesign taxonomy):
#            G10: AdvMath (5) needs Alg-2 (4) at G9
#            G11: AdvMath (5) or PreCalc (6) need Alg-2 OR AdvMath at G10
#            G12: AdvMath (5) or PreCalc (6) need Alg-2 in history;
#                 Calc (7) needs PreCalc (6) at G11.
#        (c) No-retake rule -- RELAXED from V2 via two knobs:
#              Knob 1: PreCalc (6) dropped from the non-retakable set.
#                      Non-retakable set in V2a is {A1=2, GM=3, A2=4}.
#                      Rationale: a student who took PreCalc at G11 may
#                      legitimately repeat PreCalc at G12 before Calc.
#              Knob 2: consecutive-stage-only check. Instead of blocking
#                      L at stage k whenever L appears in ANY later-stage
#                      OTR (V2's cumulative rule), block only when L
#                      appears in the immediately-NEXT stage's OTR:
#                         G12: no constraint (first stage in backward).
#                         G11: block L in {2,3,4} appearing in OTRs_feas_4.
#                         G10: block L in {2,3,4} appearing in OTRs_feas_3.
#                         G9 : block L in {2,3,4} appearing in OTRs_feas_2.
#            V2's stricter cumulative rule ate too much utility (CI for
#            final value covered the observed mean). V2a loosens the rule
#            while preserving "no back-to-back retake of foundational
#            algebra/geometry" as the defensible core. BM=1 and AM=5
#            remain re-takeable at every stage.
#      Updated 2026-04-19 from the V2.0 hard-coded indices that the
#      V2.2 run inadvertently kept (script targeted PreCalc-as-level-5
#      after V2.1_A_Redesign moved PreCalc to level 6). See memo §3.9.
#   2. Structurally based on 352_6_Run_TMLE_OTR_MathAch_260112.R (the
#      paper's constrained version). V2a ports a relaxed no-retake rule
#      from 349_feasibility_constraints.R (constraint (c) above);
#      no-triple-take / forced-prereq logic still omitted.
#   ** Value updating (02_1_value_updating_constrained.R) is appended at
#      the bottom of this file so the full learn-then-evaluate pipeline
#      for V2a runs as a single sourced script. **
#   3. Requires PS matrices already saved by 01_multi_stage_MA.R to
#      ../data/tmle_results/01_multistage_PS_df_CV_and_NoCV_all_grades_20260418.rdata


# load the packages
library(fastDummies)
library(data.table)
library(dplyr)
library(tmle3)
library(sl3)
library(tlverse)
library(tmle3mopttx)
library(devtools)
library(caret)
library(dplyr)
library(purrr)

library(future)
library(future.apply)
library(doFuture)   # <-- needed so foreach / origami::cross_validate() dispatch
                    #     through the future plan. Without this, sl3 silently
                    #     falls back to sequential even with plan(multisession).

#source("253_D_TMLE_functions.r")
#source("286_TMLE_flexible_using_GLM.R")
# Set parallel plan (adjust workers based on available cores)
options(future.globals.maxSize = 13 * 1024^3)

#setwd("H:/My Drive/DTR_Project/Synthetic_data/script")

# Cache directory for TMLE checkpoints (replaces the paper's
# `data_files/` path, which does not exist in this project).
dir.create("data/tmle_results", showWarnings = FALSE, recursive = TRUE)

df_sim <- read.csv("data/synthetic_odtr_n10000.csv")

# Treatments in the synthetic CSV are stored as integers 1..J_k for
# portability. `tmle3mopttx` expects them as factors with integer-string
# levels starting at 1, so convert them immediately after load.
df_sim$course_g9  <- factor(df_sim$course_g9,  levels = as.character(1:4))
df_sim$course_g10 <- factor(df_sim$course_g10, levels = as.character(1:5))
df_sim$course_g11 <- factor(df_sim$course_g11, levels = as.character(1:6))
df_sim$course_g12 <- factor(df_sim$course_g12, levels = as.character(1:7))


# first_run = FALSE means the unconstrained run (01_multi_stage_MA.R)
# has already been done and its propensity-score matrices have been
# saved to disk. The constrained run below loads those matrices and
# re-uses them for (a) PS trimming at the current stage and (b) the
# a priori PreCalc-prerequisite look-up at the prior stage.
#
# If you ever want a fully cold run, set first_run = TRUE and remove
# the `if (!first_run)` guards around the a priori blocks -- the
# script will then do PS trimming only (no prerequisite check).
first_run <- FALSE

if (!first_run){
  # Loads: ps_df1..ps_df4 (CV) and ps_noCV_df1..ps_noCV_df4 (non-CV).
  # Non-CV PS is what TMLE uses for trimming / feasibility.
  load("data/tmle_results/01_multistage_PS_df_CV_and_NoCV_all_grades_260421_N10000.rdata")
}

time_start_all <- Sys.time()

# Three feasibility constraints are applied in this script (see Sec 2.6.2
# below and the analogous blocks at each stage):
#   (1) PS trimming -- set blip_df{k}[ps_noCV_df{k} < cut_off{k}] <- -Inf
#       at every stage k.
#   (2) A priori prerequisite -- Calculus (level 6) and AP Calculus
#       (level 7) require PreCalculus (level 5) in the prior grade.
#       Enforced at G12 (checks PS for PreCalc at G11) and G11 (checks
#       PS for PreCalc at G10). At G10 and G9 only constraint (1) applies.
#   (3) [V2a] No-retake rule (RELAXED) -- a non-retakable level L in
#       {A1=2, GM=3, A2=4} may not be recommended in two consecutive
#       stages. BM=1, AM=5, PC=6, CL=7 are unconstrained by this rule.
#       Because backward induction fixes later stages first, at each
#       stage k we block L only where L == OTRs_feas_{k+1}:
#         G12: no-op (no later stages to check).
#         G11: block L in {2,3,4} that equal OTRs_feas_4.
#         G10: block L in {2,3,4} that equal OTRs_feas_3.
#         G9 : block L in {2,3,4} that equal OTRs_feas_2.
# The paper's `349_feasibility_constraints.R` also contains no-triple-take
# and forced-prereq rules that remain deliberately NOT ported here;
# see memo.md for the scope decision.

cut_off4 <- 0.03
cut_off3 <- 0.03
cut_off2 <- 0.04
cut_off1 <- 0.05

# ------------------------------------------------------------------
# set up the multisession computing
plan(multisession, workers = parallel::detectCores()-8)
registerDoFuture()   # <-- registers future as the foreach backend so that
                     #     origami::cross_validate() (used internally by sl3)
                     #     dispatches its fold-level foreach loop to the
                     #     multisession workers. Without this line, the plan
                     #     exists but foreach runs sequentially.

# Sanity-check: number of workers actually active.
cat(sprintf("[future] nbrOfWorkers = %d\n", future::nbrOfWorkers()))

##########################################################################
# STAGE 1: DATA PREPARATION                                              #
##########################################################################


#std_TF <- T
# load the training and the testing dataset
#weighted <- TRUE

#load("data_file\\340_cleaned_train_A_4stage_251121_GS_wGPA_Y6_all_pmm.rdata")


# Based on the Longitudinal TMLE paper, we should fix the CV folds across all stages
library(origami)

# 1) Global folds (do ONCE)
set.seed(20260418)
V <- 10
folds_global <- make_folds(n = nrow(df_sim), fold_fun = folds_vfold, V = V)
foldvec_global <- folds2foldvec(folds_global)   # length n, values in 1:V

# Add stable row id if you ever subset/reorder later
df_sim$.row_id <- seq_len(nrow(df_sim))

# helper: rebuild fold list for a stage dataset using the global foldvec
make_stage_folds <- function(stage_df, foldvec_global, V) {
  fv <- foldvec_global[stage_df$.row_id]
  lapply(seq_len(V), function(v) fold_from_foldvec(v, fv))
}


folds_fixed <- make_stage_folds(df_sim, foldvec_global, V)

str(folds_fixed)

##########################################################################
# STAGE 2: run TMLE on the 12th grade                                    #
##########################################################################

# -----------------------------
# 2.1 TMLE specification
# -----------------------------

# --- node_list for Grade 12 stage (final, real outcome) ------------------
# W at Grade 12 = all baseline covariates + full observed history
# (identity, course, cumulative GPA at every prior stage) + the
# Grade-12 identity state. A = course_g12. Y = math_gpa_g12_cum (the
# real final outcome; in backward induction the downstream stages use
# pseudo-outcomes).
node_list_12 <- list(
  W = c(# -- baseline --
    "pre_hs_math", "student_ses", "student_male",
    # -- Grade 9 wave --
    "math_identity_g9",  "course_g9",  "math_gpa_g9",
    # -- Grade 10 wave --
    "math_identity_g10", "course_g10", "math_gpa_g10_cum",
    # -- Grade 11 wave --
    "math_identity_g11", "course_g11", "math_gpa_g11_cum",
    # -- Grade 12 pre-treatment state --
    "math_identity_g12"),
  A = "course_g12",
  Y = "math_gpa_g12_cum"
)

# Initialize few of the learners:
# nthread = 1 on xgboost: xgboost's default is "use all cores", which double-
# books once sl3 already parallelizes the CV folds across future workers.
# Pin to 1 so the effective parallelism is workers (fold-level) x 1 (xgboost
# internal) = workers. If you want more cores in play, increase `workers`
# above rather than bumping nthread here.
lrn_xgboost_50  <- Lrnr_xgboost$new(nrounds = 50,  nthread = 1)
lrn_xgboost_100 <- Lrnr_xgboost$new(nrounds = 100, nthread = 1)
lrn_xgboost_500 <- Lrnr_xgboost$new(nrounds = 500, nthread = 1)

lrn_mean <- Lrnr_mean$new()
lrn_glm <- Lrnr_glm_fast$new()
lrn_rf <- Lrnr_randomForest$new()
lrn_lasso <- Lrnr_glmnet$new(alpha = 1)


# --------------------------------

## Define the Q learner, which is just a regular learner:
Q_learner <- Lrnr_sl$new(
  learners = list(lrn_xgboost_50, lrn_xgboost_100,lrn_xgboost_500, lrn_mean, lrn_glm, lrn_lasso),
  metalearner = Lrnr_nnls$new()
)

## Define the g learner, which is a multinomial learner:
# specify the appropriate loss of the multinomial learner:
mn_metalearner <- make_learner(Lrnr_solnp,
                               eval_function = loss_loglik_multinomial,
                               learner_function = metalearner_linear_multinomial
)

g_learner <- make_learner(Lrnr_sl, list(lrn_xgboost_50, lrn_xgboost_100,lrn_xgboost_500,
                                        Lrnr_glmnet$new(),
                                        Lrnr_ranger$new(mtry=2,
                                                        min.node.size = 5,
                                                        num.trees = 100,
                                                        classification=TRUE,
                                                        num.threads = 1)), mn_metalearner)
# ^^ num.threads = 1 on ranger: default is "all detected cores", which inside
# a multisession worker would mean (workers * all_cores) threads competing for
# the same physical cores. Pin to 1 for the same reason as xgboost nthread.

## Define the Blip learner, which is a multivariate learner:
learners <- list(lrn_xgboost_50,lrn_xgboost_100,lrn_xgboost_500, lrn_glm)

# learners <- list(lrn_rf)
b_learner <- create_mv_learners(learners = learners)

# specify outcome and treatment regressions and create learner list
learner_list <- list(Y = Q_learner, A = g_learner, B = b_learner)


source("ODTR_Functions_V1.0.0.R")


if (first_run){

  
  out <- NULL
  success <- FALSE
  attempts <- 0
  max_attempts <- 50
  seed <- 19
  
  
  # -----------------------------
  # 2.2 fit TMLE model
  # -----------------------------
  
  tmle_spec_cat4 <- tmle3_mopttx_blip_revere(
    # Moderators at Grade 12: current identity state + baseline prep +
    # most recent cumulative GPA + SES. Must be a subset of W.
    V = c("math_identity_g12", "pre_hs_math", "math_gpa_g11_cum", "student_ses"),
    type = 'blip1',
    learners = learner_list,
    maximize = T, complex=T,
    realistic=F, resource=1,
    interpret = F
  )
  
  tmle_task4 <- tmle_spec_cat4$make_tmle_task(
    data = df_sim,
    node_list = node_list_12,
    folds = folds_fixed
  )
  
  
  
  time4_0 <- Sys.time()
  while (!success && attempts < max_attempts) {
    attempts <- attempts + 1
    seed <- seed + 1
    set.seed(seed)
    tryCatch({
      # 2) initial likelihood
      initial_likelihood <- tmle_spec_cat4$make_initial_likelihood(
        tmle_task4,
        learner_list = learner_list
      )
      
      
      # 3) updater + targeted likelihood
      updater <- tmle_spec_cat4$make_updater()
      targeted_likelihood <- tmle_spec_cat4$make_targeted_likelihood(
        initial_likelihood,
        updater
      )
      
      # 4) params
      tmle_params <- tmle_spec_cat4$make_params(
        tmle_task4,
        likelihood = targeted_likelihood
      )
      updater$tmle_params <- tmle_params
      
      # 5) fit
      fit_cat4 <- fit_tmle3(
        tmle_task4,
        targeted_likelihood,
        tmle_params,
        updater
      )
      success <- T
    }, error =  function(e){
      message(e)
      cat("Error occurred, attempt: ", attempts, "\n")
    }
    )
  }
  
  
  tmle_task4 <- fit_cat4$tmle_task
  
  
  #####################################################################
  #                                                                   #
  # 2025.11.13 New method to retrieve the cf and propensity matrix    #
  #                                                                   #
  #####################################################################
  

  
  results_task4 <- extract_tmle_components(
    node_list = node_list_12,
    tmle_fit = fit_cat4,
    tmle_spec = tmle_spec_cat4,
    tmle_task = tmle_task4,
    original_data = df_sim
  )
  
  
  outcome_node <- tmle_task4$npsem$Y
  outcome_bounds <- outcome_node$variable_type$bounds
  Y_min <- outcome_bounds[1]
  Y_max <- outcome_bounds[2]
  
  
  blip_df4 <- results_task4$blip_df
  ps_df4 <- results_task4$ps_df
  ps_noCV_df4 <- results_task4$ps_noCV_df
  cf_df4 <- results_task4$cf_df
  Y_min <- results_task4$Y_min
  Y_max <- results_task4$Y_max
}




# ----------------------------------------------------------
# 2.6.1 no constraints
# ----------------------------------------------------------

# OTRs_4_unconstrained <- apply(blip_df4, 1, which.max)

# OTRs_pkg <- as.integer(as.character(tmle_spec_cat4$return_rule))
# table(OTRs_4_unconstrained, OTRs_pkg)


# A4 <- as.integer(as.character(df_sim$course_g12))
#
# opt4 <- Optimizer$new(
#   treat_obs = A4,
#   y_obs = df_sim$math_gpa_g12_cum,
#   y_pre = results_task4$QAW,
#   y_global_min = Y_min,
#   y_global_max = Y_max,
#   propensity_matrix = as.matrix(ps_df4),
#   OTR_pre = OTRs_4_unconstrained,
#   values_est_init = cf_matrix_original_scale_4[cbind(1:nrow(df_sim), OTRs_4_unconstrained)]
# )
# 
# 
# out_cv <- opt4$update()
# out_cv$psi_hat
# out_cv$se


# ----------------------------------------------------------
# 2.6.2 with constraints (Grade 12)
# ----------------------------------------------------------
#
# Two feasibility constraints are applied at each stage:
#   (1) PS trimming -- set blip to -Inf wherever the NON-CV propensity
#       for that (student, treatment) cell falls below the stage cut-off.
#       Non-CV PS is the TMLE-official choice for the trimming step;
#       CV PS is reserved for the (not-done-here) value-update step.
#   (2) A priori prerequisite: Calculus (level 6) and AP Calculus (level 7)
#       require PreCalculus (level 5) as a prerequisite in the PRIOR grade.
#       A G12 student cannot be routed to Calc / APCalc unless their
#       PreCalc-at-G11 propensity is itself above the G11 cut-off -- i.e.
#       the student is plausibly able to satisfy the prerequisite in the
#       prior year. If not, Calc / APCalc are disallowed for this student.
#
# The a priori check at G12 reads ps_noCV_df3, which was loaded at the
# top of the script from the unconstrained run. G11 has not been re-fit
# yet in this backward-induction loop, so the loaded ps_noCV_df3 is
# still in memory and is the correct reference.


#save(results_task4, file = "../data/tmle_results/02_stage4_results_260422_N10000.rdata")


# --- Constraint 2: a priori chain check at G12 (V2.1_A_Redesign rules) ---
# Two separate rules (the V2.0 single-rule check was wrong under the
# new taxonomy where PreCalc moved from level 5 to level 6 and Calc
# from level 6 to level 7):
#   (a) Calc (level 7) requires PreCalc (level 6) at G11.
#   (b) AdvMath (5) or PreCalc (6) at G12 require Alg-2 (level 4)
#       in history (G10 or G11).
# Loaded ps_noCV_df3 (from unconstrained 01) is authoritative for the
# G11-prereq look-up; ps_noCV_df2 from the loaded image for G10.
if (!first_run) {
  load("data/tmle_results/02_stage4_results_260422_N10000.rdata")
  blip_df4 <- results_task4$blip_df
  ps_df4 <- results_task4$ps_df
  ps_noCV_df4 <- results_task4$ps_noCV_df
  cf_df4 <- results_task4$cf_df
  Y_min <- results_task4$Y_min
  Y_max <- results_task4$Y_max
  
  time_start <- Sys.time()
  
  # --- Constraint 1: PS trimming at G12 ---
  blip_df4_feas <- blip_df4
  blip_df4_feas[ps_noCV_df4 < cut_off4] <- -Inf
  OTRs_feas_4 <- apply(blip_df4_feas, 1, which.max)
  cat(sprintf("\n[G12] After PS trimming (cut_off4 = %.3f):\n", cut_off4))
  print(table(OTRs_feas_4))

  # --- Rule (a): Calc-needs-PreCalc-at-G11 ---
  stu_calc_g12 <- which(OTRs_feas_4 == 7L)
  cat(sprintf("[G12-a] %d students have OTR = Calc (level 7).\n",
              length(stu_calc_g12)))
  if (length(stu_calc_g12) > 0L) {
    ps_precalc_g11 <- ps_noCV_df3[cbind(stu_calc_g12, 6L)]   # PreCalc = col 6 at G11
    blocked_calc   <- stu_calc_g12[ps_precalc_g11 < cut_off3]
    cat(sprintf("[G12-a]   blocked %d (P(PreCalc@G11) < %.3f) -> Calc removed.\n",
                length(blocked_calc), cut_off3))
    if (length(blocked_calc) > 0L) {
      blip_df4_feas[cbind(blocked_calc, 7L)] <- -Inf
    }
  }

  # --- Rule (b): AdvMath/PreCalc-need-Alg2-in-history ---
  stu_advpc_g12 <- which(OTRs_feas_4 %in% c(5L, 6L))
  cat(sprintf("[G12-b] %d students have OTR in {AdvMath (5), PreCalc (6)}.\n",
              length(stu_advpc_g12)))
  if (length(stu_advpc_g12) > 0L) {
    ps_alg2_g10 <- ps_noCV_df2[cbind(stu_advpc_g12, 4L)]    # Alg2 = col 4 at G10
    ps_alg2_g11 <- ps_noCV_df3[cbind(stu_advpc_g12, 4L)]    # Alg2 = col 4 at G11
    ps_alg2_max <- pmax(ps_alg2_g10, ps_alg2_g11)
    blocked_advpc <- stu_advpc_g12[ps_alg2_max < cut_off3]
    cat(sprintf("[G12-b]   blocked %d (max P(Alg2 @ G10|G11) < %.3f) -> AdvMath/PreCalc removed.\n",
                length(blocked_advpc), cut_off3))
    if (length(blocked_advpc) > 0L) {
      blip_df4_feas[cbind(blocked_advpc, 5L)] <- -Inf
      blip_df4_feas[cbind(blocked_advpc, 6L)] <- -Inf
    }
  }

  OTRs_feas_4 <- apply(blip_df4_feas, 1, which.max)
  cat("[G12] After a priori checks:\n")
  print(table(OTRs_feas_4))
}

# --- Constraint 3 (V2): no-retake check at G12 -----------------------------
# G12 is the FIRST stage in backward induction, so no later-stage OTRs
# exist yet to constrain against. The retake check is therefore a no-op
# at G12; earlier stages (G11, G10, G9) will each see OTRs_feas_4 in
# their own retake blocks and block accordingly so the full recommended
# sequence has at most one occurrence of each non-retakable level.

pseudo_out_4 <- cf_df4[cbind(1:nrow(df_sim), OTRs_feas_4)]

summary(pseudo_out_4)

# 2025.11.16
# you can start running this file from here to save time
# load("data_file/333_multistage_after_training_stage_4.rdata")


##########################################################################
# STAGE 3: run TMLE on the 11th grade                                    #
##########################################################################

summary(pseudo_out_4)

df_sim$Y3_pseudo <- pseudo_out_4


# --- node_list for Grade 11 stage (pseudo-outcome from backward induction) --
# W at Grade 11 = baseline + Grades 9-10 history + Grade-11 identity
# state. A = course_g11. Y = Y3_pseudo (constructed at line ~338 from
# cf_df4 evaluated at the stage-12 optimal rule).
node_list_11 <- list(
  W = c("pre_hs_math", "student_ses", "student_male",
        "math_identity_g9",  "course_g9",  "math_gpa_g9",
        "math_identity_g10", "course_g10", "math_gpa_g10_cum",
        "math_identity_g11"),
  A = "course_g11",
  Y = "Y3_pseudo"
)


tmle_spec_cat3 <- tmle3_mopttx_blip_revere(
  V = c("math_identity_g11", "pre_hs_math", "math_gpa_g10_cum", "student_ses"),
  type = 'blip1',
  learners = learner_list,
  maximize = T, complex=T,
  realistic = F, resource=1,
  interpret = F
)

tmle_task3 <- tmle_spec_cat3$make_tmle_task(
  data = df_sim,
  node_list = node_list_11,
  folds = folds_fixed
)



success <- FALSE
attempts <- 0
max_attempts <- 50
seed <- 666

# seed = 13 for learners <- list(lrn_xgboost_50, lrn_mean, lrn_glm) blip2
time_0 <- Sys.time()
while (!success && attempts < max_attempts) {
  attempts <- attempts + 1
  seed <- seed + 1
  set.seed(seed)
  tryCatch({
    # 2) initial likelihood
    initial_likelihood <- tmle_spec_cat3$make_initial_likelihood(
      tmle_task3,
      learner_list = learner_list
    )
    
    
    # 3) updater + targeted likelihood
    updater <- tmle_spec_cat3$make_updater()
    targeted_likelihood <- tmle_spec_cat3$make_targeted_likelihood(
      initial_likelihood,
      updater
    )
    
    # 4) params
    tmle_params <- tmle_spec_cat3$make_params(
      tmle_task3,
      likelihood = targeted_likelihood
    )
    updater$tmle_params <- tmle_params
    
    # 5) fit
    fit_cat3 <- fit_tmle3(
      tmle_task3,
      targeted_likelihood,
      tmle_params,
      updater
    )
    success <- T
  }, error =  function(e){
    message(e)
    cat("Error occurred, attempt: ", attempts, "\n")
  }
  )
}

if (success){
  print("Function executed successfully")
} else{
  print("Function failed after several attempts")
}

time_1 <- Sys.time()
time_1-time_0

tmle_task3 <- fit_cat3$tmle_task



#####################################################################
#                                                                   #
# 2025.11.13 New method to retrieve the cf and propensity matrix    #
#                                                                   #
#####################################################################


results_task3 <- extract_tmle_components(
  node_list = node_list_11,
  tmle_fit = fit_cat3,
  tmle_spec = tmle_spec_cat3,
  tmle_task = tmle_task3,
  original_data = df_sim
)



blip_df3 <- results_task3$blip_df
ps_df3 <- results_task3$ps_df
ps_noCV_df3 <- results_task3$ps_noCV_df
cf_df3 <- results_task3$cf_df
Y_min <- results_task3$Y_min
Y_max <- results_task3$Y_max


# --- Constraint 1: PS trimming at G11 ---
blip_df3_feas <- blip_df3
blip_df3_feas[ps_noCV_df3 < cut_off3] <- -Inf
OTRs_feas_3 <- apply(blip_df3_feas, 1, which.max)
cat(sprintf("\n[G11] After PS trimming (cut_off3 = %.3f):\n", cut_off3))
print(table(OTRs_feas_3))

# --- Constraint 2: a priori chain check at G11 (V2.1_A_Redesign rules) ---
# AdvMath (5) or PreCalc (6) at G11 require Alg-2 (4) or AdvMath (5)
# at G10. (Note: in V2.1_A_Redesign G11's max level is 6, not 7;
# Calc moved to level 7 and is offered only at G12.)
# G10 has not been re-fit yet in this backward-induction loop, so the
# loaded ps_noCV_df2 from the unconstrained 01 is authoritative.
if (!first_run) {
  stu_advpc_g11 <- which(OTRs_feas_3 %in% c(5L, 6L))
  cat(sprintf("[G11] %d students have OTR in {AdvMath (5), PreCalc (6)}.\n",
              length(stu_advpc_g11)))
  if (length(stu_advpc_g11) > 0L) {
    ps_alg2_g10    <- ps_noCV_df2[cbind(stu_advpc_g11, 4L)]   # Alg2 = col 4 at G10
    ps_advmath_g10 <- ps_noCV_df2[cbind(stu_advpc_g11, 5L)]   # AdvMath = col 5 at G10
    ps_max         <- pmax(ps_alg2_g10, ps_advmath_g10)
    blocked        <- stu_advpc_g11[ps_max < cut_off2]
    cat(sprintf("[G11]   blocked %d (max P(Alg2|AdvMath @ G10) < %.3f) -> AdvMath/PreCalc removed.\n",
                length(blocked), cut_off2))
    if (length(blocked) > 0L) {
      blip_df3_feas[cbind(blocked, 5L)] <- -Inf
      blip_df3_feas[cbind(blocked, 6L)] <- -Inf
      OTRs_feas_3 <- apply(blip_df3_feas, 1, which.max)
      cat("[G11] After a priori check:\n")
      print(table(OTRs_feas_3))
    }
  }
}

# --- Constraint 3 (V2a): no-retake check at G11 --------------------------
# Rule: if a non-retakable level L (A1=2, GM=3, A2=4) is already
# recommended at the IMMEDIATELY-NEXT stage (OTRs_feas_4 at G12), block
# L at G11 so the G11->G12 transition does not repeat L.
# [Knob 1] PreCalc (6) removed from the non-retakable set in V2a.
# [Knob 2] Check is consecutive-only: G11 looks at G12 only.
no_retake_g11 <- c(2L, 3L, 4L)
for (L in no_retake_g11) {
  retake_students <- which(OTRs_feas_4 == L)
  cat(sprintf("[G11-noretake] level %d scheduled at G12 for %d students -> blocked at G11.\n",
              L, length(retake_students)))
  if (length(retake_students) > 0L) {
    blip_df3_feas[cbind(retake_students, L)] <- -Inf
  }
}
OTRs_feas_3 <- apply(blip_df3_feas, 1, which.max)
cat("[G11] After no-retake check:\n")
print(table(OTRs_feas_3))


pseudo_out_3 <- cf_df3[cbind(1:nrow(df_sim), OTRs_feas_3)]


##########################################################################
# STAGE 4: run TMLE on the 10th grade                                    #
##########################################################################

df_sim$Y2_pseudo <- pseudo_out_3


# --- node_list for Grade 10 stage (pseudo-outcome from backward induction) --
# W at Grade 10 = baseline + Grade-9 history + Grade-10 identity state.
# A = course_g10. Y = Y2_pseudo (constructed from cf_df3 evaluated at
# the stage-11 optimal rule).
node_list_10 <- list(
  W = c("pre_hs_math", "student_ses", "student_male",
        "math_identity_g9",  "course_g9",  "math_gpa_g9",
        "math_identity_g10"),
  A = "course_g10",
  Y = "Y2_pseudo"
)


tmle_spec_cat2 <- tmle3_mopttx_blip_revere(
  V = c("math_identity_g10", "pre_hs_math", "math_gpa_g9", "student_ses"),
  type = 'blip1',
  learners = learner_list,
  maximize = T, complex=T,
  realistic=F, resource=1,
  interpret = F
)

tmle_task2 <- tmle_spec_cat2$make_tmle_task(
  data = df_sim,
  node_list = node_list_10,
  folds = folds_fixed
)



# seed = 13 for learners <- list(lrn_xgboost_50, lrn_mean, lrn_glm) blip2
time_0 <- Sys.time()
success <- FALSE
attempts <- 0
max_attempts <- 50
seed <- 14

while (!success && attempts < max_attempts) {
  attempts <- attempts + 1
  seed <- seed + 3
  set.seed(seed)
  tryCatch({
    # 2) initial likelihood
    initial_likelihood <- tmle_spec_cat2$make_initial_likelihood(
      tmle_task2,
      learner_list = learner_list
    )
    
    # 3) updater + targeted likelihood
    updater <- tmle_spec_cat2$make_updater()
    targeted_likelihood <- tmle_spec_cat2$make_targeted_likelihood(
      initial_likelihood,
      updater
    )
    
    # 4) params
    tmle_params <- tmle_spec_cat2$make_params(
      tmle_task2,
      likelihood = targeted_likelihood
    )
    updater$tmle_params <- tmle_params
    
    # 5) fit
    fit_cat2 <- fit_tmle3(
      tmle_task2,
      targeted_likelihood,
      tmle_params,
      updater
    )
    success <- T
  }, error =  function(e){
    message(e)
    cat("Error occurred, attempt: ", attempts, "\n")
  }
  )
}

if (success){
  print("Function executed successfully")
} else{
  print("Function failed after several attempts")
}

time_1 <- Sys.time()
time_1-time_0

tmle_task2 <- fit_cat2$tmle_task

# retrieve the TMLE estimated value for the current stage
fit_cat2


#####################################################################
#                                                                   #
# 2025.11.13 New method to retrieve the cf and propensity matrix    #
#                                                                   #
#####################################################################

results_task2 <- extract_tmle_components(
  node_list = node_list_10,
  tmle_fit = fit_cat2,
  tmle_spec = tmle_spec_cat2,
  tmle_task = tmle_task2,
  original_data = df_sim
)



blip_df2 <- results_task2$blip_df
ps_df2 <- results_task2$ps_df
ps_noCV_df2 <- results_task2$ps_noCV_df
cf_df2 <- results_task2$cf_df
Y_min <- results_task2$Y_min
Y_max <- results_task2$Y_max


# --- Constraint 1: PS trimming at G10 ---
blip_df2_feas <- blip_df2
blip_df2_feas[ps_noCV_df2 < cut_off2] <- -Inf
OTRs_feas_2 <- apply(blip_df2_feas, 1, which.max)
cat(sprintf("\n[G10] After PS trimming (cut_off2 = %.3f):\n", cut_off2))
print(table(OTRs_feas_2))

# --- Constraint 2: a priori check at G10 (V2.1_A_Redesign rule) ---
# AdvMath (level 5) at G10 requires Alg-2 (level 4) at G9.
# G9 has not been re-fit yet, so loaded ps_noCV_df1 is authoritative.
if (!first_run) {
  stu_adv_g10 <- which(OTRs_feas_2 == 5L)
  cat(sprintf("[G10] %d students have OTR = AdvMath (5).\n", length(stu_adv_g10)))
  if (length(stu_adv_g10) > 0L) {
    ps_alg2_g9 <- ps_noCV_df1[cbind(stu_adv_g10, 4L)]    # Alg2 = col 4 at G9
    blocked    <- stu_adv_g10[ps_alg2_g9 < cut_off1]
    cat(sprintf("[G10]   blocked %d (P(Alg2 @ G9) < %.3f) -> AdvMath removed.\n",
                length(blocked), cut_off1))
    if (length(blocked) > 0L) {
      blip_df2_feas[cbind(blocked, 5L)] <- -Inf
      OTRs_feas_2 <- apply(blip_df2_feas, 1, which.max)
      cat("[G10] After a priori check:\n")
      print(table(OTRs_feas_2))
    }
  }
}

# --- Constraint 3 (V2a): no-retake check at G10 --------------------------
# [Knob 2] Consecutive-only check: G10 looks only at G11 (OTRs_feas_3).
# Long-range duplicates (G10 level == G12 level) are tolerated in V2a.
no_retake_g10 <- c(2L, 3L, 4L)
for (L in no_retake_g10) {
  retake_students <- which(OTRs_feas_3 == L)
  cat(sprintf("[G10-noretake] level %d scheduled at G11 for %d students -> blocked at G10.\n",
              L, length(retake_students)))
  if (length(retake_students) > 0L) {
    blip_df2_feas[cbind(retake_students, L)] <- -Inf
  }
}
OTRs_feas_2 <- apply(blip_df2_feas, 1, which.max)
cat("[G10] After no-retake check:\n")
print(table(OTRs_feas_2))

addmargins(table(df_sim$course_g10, OTRs_feas_2))

pseudo_out_2 <- cf_df2[cbind(1:nrow(df_sim), OTRs_feas_2)]

#########################################################################
# STAGE 5: run TMLE on the 9th grade                                    #
#########################################################################

df_sim$Y1_pseudo <- pseudo_out_2


# --- node_list for Grade 9 stage (pseudo-outcome from backward induction) --
# W at Grade 9 = baseline covariates only (no prior treatment history).
# A = course_g9. Y = Y1_pseudo (constructed from cf_df2 evaluated at
# the stage-10 optimal rule).
node_list_09 <- list(
  W = c("pre_hs_math", "student_ses", "student_male",
        "math_identity_g9"),
  A = "course_g9",
  Y = "Y1_pseudo"
)



success <- FALSE
attempts <- 0
max_attempts <- 50
seed <- 14

tmle_spec_cat1 <- tmle3_mopttx_blip_revere(
  V = c("math_identity_g9", "pre_hs_math", "student_ses"),
  type = 'blip1',
  learners = learner_list,
  maximize = T, complex=T,
  realistic=F, resource=1,
  interpret = F
)

tmle_task1 <- tmle_spec_cat1$make_tmle_task(
  data = df_sim,
  node_list = node_list_09,
  folds = folds_fixed
)



# seed = 13 for learners <- list(lrn_xgboost_50, lrn_mean, lrn_glm) blip2
time_0 <- Sys.time()
while (!success && attempts < max_attempts) {
  attempts <- attempts + 1
  seed <- seed + 2
  set.seed(seed)
  tryCatch({
    
    # 2) initial likelihood
    initial_likelihood <- tmle_spec_cat1$make_initial_likelihood(
      tmle_task1,
      learner_list = learner_list
    )
    
    
    # 3) updater + targeted likelihood
    updater <- tmle_spec_cat1$make_updater()
    targeted_likelihood <- tmle_spec_cat1$make_targeted_likelihood(
      initial_likelihood,
      updater
    )
    
    # 4) params
    tmle_params <- tmle_spec_cat1$make_params(
      tmle_task1,
      likelihood = targeted_likelihood
    )
    updater$tmle_params <- tmle_params
    
    # 5) fit
    fit_cat1 <- fit_tmle3(
      tmle_task1,
      targeted_likelihood,
      tmle_params,
      updater
    )
    success <- T
  }, error =  function(e){
    message(e)
    cat("Error occurred, attempt: ", attempts, "\n")
  }
  )
}

if (success){
  print("Function executed successfully")
} else{
  print("Function failed after several attempts")
}

fit_cat1

time_1 <- Sys.time()
time_1-time_0

tmle_task1 <- fit_cat1$tmle_task

# retrieve the TMLE estimated value for the current stage
fit_cat1



#####################################################################
#                                                                   #
# 2025.11.13 New method to retrieve the cf and propensity matrix    #
#                                                                   #
#####################################################################

results_task1 <- extract_tmle_components(
  node_list = node_list_09,
  tmle_fit = fit_cat1,
  tmle_spec = tmle_spec_cat1,
  tmle_task = tmle_task1,
  original_data = df_sim
)



blip_df1 <- results_task1$blip_df
ps_df1 <- results_task1$ps_df
ps_noCV_df1 <- results_task1$ps_noCV_df
cf_df1 <- results_task1$cf_df
Y_min <- results_task1$Y_min
Y_max <- results_task1$Y_max


# --- Constraint 1: PS trimming at G9 ---
# No a priori rule at G9: the menu is {BasicMath, Alg1, Geom, Alg2}
# (levels 1-4); no advanced courses here.
blip_df1_feas <- blip_df1
blip_df1_feas[ps_noCV_df1 < cut_off1] <- -Inf
OTRs_feas_1 <- apply(blip_df1_feas, 1, which.max)
cat(sprintf("\n[G9] After PS trimming (cut_off1 = %.3f):\n", cut_off1))
print(table(OTRs_feas_1))

# --- Constraint 3 (V2a): no-retake check at G9 ---------------------------
# [Knob 2] Consecutive-only check: G9 looks only at G10 (OTRs_feas_2).
# Long-range duplicates (G9 level == G11 or G12 level) are tolerated.
no_retake_g9 <- c(2L, 3L, 4L)
for (L in no_retake_g9) {
  retake_students <- which(OTRs_feas_2 == L)
  cat(sprintf("[G9-noretake] level %d scheduled at G10 for %d students -> blocked at G9.\n",
              L, length(retake_students)))
  if (length(retake_students) > 0L) {
    blip_df1_feas[cbind(retake_students, L)] <- -Inf
  }
}
OTRs_feas_1 <- apply(blip_df1_feas, 1, which.max)
cat("[G9] After no-retake check:\n")
print(table(OTRs_feas_1))

addmargins(table(df_sim$course_g9, OTRs_feas_1))

pseudo_out_1 <- cf_df1[cbind(1:nrow(df_sim), OTRs_feas_1)]

summary(pseudo_out_1)
# save the four ps-matrices for future use
# save(ps_df1, ps_df2, ps_df3, ps_df4,
#      ps_noCV_df1, ps_noCV_df2, ps_noCV_df3, ps_noCV_df4,
#      file = "data_files/352_6_PS_df_CV_and_NoCV_all_grades_260112.rdata")

beepr::beep(1)


# ----------------------------------------------------------
# 3.0 check the recommendation sequences
# ----------------------------------------------------------


OTRs_feas_all <- cbind(OTRs_feas_1, OTRs_feas_2, OTRs_feas_3, OTRs_feas_4)

rec_mult_MA_con <- data.frame(A1 = OTRs_feas_1,
                              A2 = OTRs_feas_2,
                              A3 = OTRs_feas_3,
                              A4 = OTRs_feas_4)

if (success){
  print("Function executed successfully")
} else{
  print("Function failed after several attempts")
}

# --- sanity check on the a priori chain rules (V2.1_A_Redesign) ---
# Each count should be 0 if the constraints were applied correctly.
n_rec <- nrow(rec_mult_MA_con)
v_g10       <- sum(rec_mult_MA_con$A2 == 5L &
                   ps_noCV_df1[cbind(seq_len(n_rec), 4L)] < cut_off1)
ps_max_g10  <- pmax(ps_noCV_df2[cbind(seq_len(n_rec), 4L)],
                    ps_noCV_df2[cbind(seq_len(n_rec), 5L)])
v_g11       <- sum(rec_mult_MA_con$A3 %in% c(5L, 6L) & ps_max_g10 < cut_off2)
v_g12_calc  <- sum(rec_mult_MA_con$A4 == 7L &
                   ps_noCV_df3[cbind(seq_len(n_rec), 6L)] < cut_off3)
ps_max_alg2 <- pmax(ps_noCV_df2[cbind(seq_len(n_rec), 4L)],
                    ps_noCV_df3[cbind(seq_len(n_rec), 4L)])
v_g12_advpc <- sum(rec_mult_MA_con$A4 %in% c(5L, 6L) & ps_max_alg2 < cut_off3)
cat(sprintf("\n[sanity] G10 AdvMath w/o feasible Alg2@G9              : %d (expect 0)\n", v_g10))
cat(sprintf("[sanity] G11 AdvMath/PreCalc w/o feasible Alg2|AM@G10  : %d (expect 0)\n", v_g11))
cat(sprintf("[sanity] G12 Calc            w/o feasible PreCalc@G11  : %d (expect 0)\n", v_g12_calc))
cat(sprintf("[sanity] G12 AdvMath/PreCalc w/o feasible Alg2 hist    : %d (expect 0)\n", v_g12_advpc))

# --- V2a sanity check: no-retake rule (consecutive-only) -----------------
# The V2a rule enforces: for each non-retakable L in {A1=2, GM=3, A2=4},
# no two CONSECUTIVE stages may both recommend L. Long-range duplicates
# (e.g. A1 at G9 and A1 at G11) are allowed.
non_retakable_consec <- c(2L, 3L, 4L)
rec_mat <- as.matrix(rec_mult_MA_con[, c("A1", "A2", "A3", "A4")])

# Consecutive-duplicate check (should be 0 for each pair):
v_consec_12 <- sum(rec_mat[, 1] == rec_mat[, 2] & rec_mat[, 1] %in% non_retakable_consec)
v_consec_23 <- sum(rec_mat[, 2] == rec_mat[, 3] & rec_mat[, 2] %in% non_retakable_consec)
v_consec_34 <- sum(rec_mat[, 3] == rec_mat[, 4] & rec_mat[, 3] %in% non_retakable_consec)
cat(sprintf("[sanity] consecutive re-take G9->G10 in {A1,GM,A2}     : %d (expect 0)\n", v_consec_12))
cat(sprintf("[sanity] consecutive re-take G10->G11 in {A1,GM,A2}    : %d (expect 0)\n", v_consec_23))
cat(sprintf("[sanity] consecutive re-take G11->G12 in {A1,GM,A2}    : %d (expect 0)\n", v_consec_34))

# Informational: long-range duplicates that are NOT blocked by V2a rule.
# (Non-zero values are expected; listed so we can see how permissive V2a is.)
n_long_pc <- sum(rec_mat[, 3] == 6L & rec_mat[, 4] == 6L)  # PC at G11 and G12 (allowed by Knob 1)
n_any_dup <- sum(apply(rec_mat, 1, function(r) any(duplicated(r[r %in% c(2L,3L,4L,6L,7L)]))))
cat(sprintf("[info]   PC recommended at both G11 and G12             : %d (allowed in V2a)\n",
            n_long_pc))
cat(sprintf("[info]   students with any non-retakable duplicate      : %d (includes long-range)\n",
            n_any_dup))

#save(rec_mult_MA_con, file = "../data/tmle_results/02_multistage_rec_synth_MA_constrained_260421_N10000_V2a.rdata")


# save the R image
#save.image(file = "../data/tmle_results/02_multistage_image_synth_MA_constrained_260421_N10000_V2a.rdata")


time_end_all <- Sys.time()

time_end_all - time_start_all


# ============================================================================
# ============================================================================
#
#                VALUE UPDATING  (appended from 02_1_value_updating_constrained.R)
#
# This section performs prefix-weighted backward-induction TMLE targeting
# on the V2a constrained rule learned above. It is identical to
# 02_1_value_updating_constrained.R except that:
#   - the initial load() is commented out (the workspace is already in
#     memory from the policy-learning section above);
#   - the final save.image() path is retargeted to the _V2a file.
#
# ============================================================================
# ============================================================================

# date: 2026.04.18
# purpose:
#   Value updating for the CONSTRAINED multi-stage ODTR learned by
#   02_multi_stage_MA_constrianted.R (which applies PS trimming at every
#   stage and an a priori PreCalc-prerequisite check at G12 / G11).
#   Mirrors the logic of 352_7_Value_updating_V2.R: backward induction
#   with prefix-weighted TMLE targeting at each stage, using CV
#   propensity scores.
#
# Inputs:
#   ../data/tmle_results/02_multistage_image_synth_MA_constrained_20260418.rdata
#     (produced by save.image() at the end of 02_multi_stage_MA_constrianted.R;
#      contains df_sim, OTRs_feas_{1..4} [constrained], ps_df{1..4} (CV),
#      cf_df4, results_task4$QAW, node_list_{09,10,11,12}, folds_fixed,
#      learner primitives.)
#
# Outputs:
#   Final value estimate psi_hat and SE printed to the console; full
#   workspace saved to
#   ../data/tmle_results/02_1_value_updating_constrained_260421_N10000_V2a.rdata
#
# Notes:
#   - This script is structurally identical to
#     01_1_value_updating_unconstrained.R. The only differences are the
#     input image, the console banner, and the output filename.
#   - OTRs_feas_k here are the CONSTRAINED rules: they already respect
#     PS trimming (blip -> -Inf where ps_noCV < cut_off) and the a priori
#     PreCalc-prerequisite for Calc / APCalc at G12 / G11. The value
#     update is blind to the constraint mechanism: it only needs the
#     final rule d_t and the CV propensities.
#   - CV propensity matrices ps_df{1..4} are used both for the prefix
#     weights and as the `propensity_matrix` argument to Optimizer$new.
#     (Non-CV PS was only used during the policy-learning step for
#     feasibility trimming; value updating uses CV PS throughout.)


# -----------------------------------------------------------------
# 0. Load learned policy + fitted models
# -----------------------------------------------------------------
# load() intentionally commented out: the policy-learning section above
# has already populated df_sim, OTRs_feas_{1..4}, ps_df{1..4}, cf_df4,
# results_task4, node_list_*, folds_fixed, and learner primitives into
# the current workspace.
# load("../data/tmle_results/02_multistage_image_synth_MA_constrained_260421_N10000_V2a.rdata")

library(fastDummies)
library(data.table)
library(dplyr)
library(tmle3)
library(sl3)
library(tlverse)
library(tmle3mopttx)
library(devtools)
library(caret)
library(dplyr)
library(purrr)
library(future)
library(future.apply)

source("ODTR_Functions_V1.0.0.R")   # re-sources Optimizer, crossfit_Q_pseudo_stage

# -----------------------------------------------------------------
# 1. Prefix-weight helpers (ported verbatim from 352_7_Value_updating_V2.R)
# -----------------------------------------------------------------
get_g_obs <- function(ps_df, A_int, ps_floor = 1e-6) {
  ps <- as.matrix(ps_df)
  g  <- ps[cbind(seq_len(nrow(ps)), A_int)]
  pmax(g, ps_floor)
}

build_prefix_w_4stage <- function(A1, A2, A3, A4,
                                  d1, d2, d3, d4,
                                  ps_df1, ps_df2, ps_df3, ps_df4,
                                  ps_floor = 1e-6) {
  n <- length(A1)

  g1_obs <- get_g_obs(ps_df1, A1, ps_floor)
  g2_obs <- get_g_obs(ps_df2, A2, ps_floor)
  g3_obs <- get_g_obs(ps_df3, A3, ps_floor)

  prefix_w1 <- rep(1, n)                                          # W0
  prefix_w2 <- prefix_w1 * (as.numeric(A1 == d1) / g1_obs)         # W1
  prefix_w3 <- prefix_w2 * (as.numeric(A2 == d2) / g2_obs)         # W2
  prefix_w4 <- prefix_w3 * (as.numeric(A3 == d3) / g3_obs)         # W3

  prefix_w2[!is.finite(prefix_w2)] <- 0
  prefix_w3[!is.finite(prefix_w3)] <- 0
  prefix_w4[!is.finite(prefix_w4)] <- 0

  list(prefix_w1 = prefix_w1, prefix_w2 = prefix_w2,
       prefix_w3 = prefix_w3, prefix_w4 = prefix_w4)
}


# -----------------------------------------------------------------
# 2. Assemble inputs (integer-coded A_t, rule d_t, outcome bounds)
# -----------------------------------------------------------------
A1_int <- as.integer(df_sim$course_g9)
A2_int <- as.integer(df_sim$course_g10)
A3_int <- as.integer(df_sim$course_g11)
A4_int <- as.integer(df_sim$course_g12)

# Constrained rule d_t from the policy-learning step.
d1 <- OTRs_feas_1
d2 <- OTRs_feas_2
d3 <- OTRs_feas_3
d4 <- OTRs_feas_4

prefix_weights <- build_prefix_w_4stage(
  A1 = A1_int, A2 = A2_int, A3 = A3_int, A4 = A4_int,
  d1 = d1,     d2 = d2,     d3 = d3,     d4 = d4,
  ps_df1 = ps_df1, ps_df2 = ps_df2, ps_df3 = ps_df3, ps_df4 = ps_df4,
  ps_floor = 1e-6
)

y_global_min <- min(df_sim$math_gpa_g12_cum)
y_global_max <- max(df_sim$math_gpa_g12_cum)


# -----------------------------------------------------------------
# 3. Stage 12 update  (real outcome Y = math_gpa_g12_cum)
# -----------------------------------------------------------------
opt4 <- Optimizer$new(
  treat_obs         = A4_int,
  y_obs             = df_sim$math_gpa_g12_cum,
  y_pre             = results_task4$QAW,
  y_global_min      = y_global_min,
  y_global_max      = y_global_max,
  propensity_matrix = as.matrix(ps_df4),
  OTR_pre           = d4,
  values_est_init   = cf_df4[cbind(seq_len(nrow(cf_df4)), d4)],
  prefix_w          = prefix_weights$prefix_w4
)
fit4    <- opt4$update()
pseudo3 <- fit4$Qd_star
value_4 <- fit4$psi_hat
cat(sprintf("Stage-12 updated value : %.4f\n", value_4))


# -----------------------------------------------------------------
# 4. Stage 11 update  (pseudo-outcome = pseudo3 from fit4)
# -----------------------------------------------------------------
options(future.globals.maxSize = 13 * 1024^3)
plan(multisession, workers = parallel::detectCores() - 8)
registerDoFuture()   # defensive re-register (already global from the top of
                     # the script, but kept here so this value-updating block
                     # can be sourced standalone if needed).
cat(sprintf("[future] nbrOfWorkers (value-update) = %d\n", future::nbrOfWorkers()))

Q_learner <- Lrnr_sl$new(
  learners    = list(lrn_xgboost_100, lrn_glm, lrn_lasso),
  metalearner = Lrnr_nnls$new()
)

df_sim$Y3_pseudo <- pseudo3

Q3_init <- crossfit_Q_pseudo_stage(
  stage_df    = df_sim,
  Y_col       = "Y3_pseudo",
  A_col       = node_list_11$A,
  W_cols      = node_list_11$W,
  d_vec       = d3,
  folds_fixed = folds_fixed,
  learner     = Q_learner
)

opt3 <- Optimizer$new(
  treat_obs         = A3_int,
  y_obs             = pseudo3,
  y_pre             = Q3_init$QAW,
  y_global_min      = y_global_min,
  y_global_max      = y_global_max,
  propensity_matrix = as.matrix(ps_df3),
  OTR_pre           = d3,
  values_est_init   = Q3_init$Qd,
  prefix_w          = prefix_weights$prefix_w3
)
fit3    <- opt3$update()
pseudo2 <- fit3$Qd_star
value_3 <- fit3$psi_hat
cat(sprintf("Stage-11 updated value : %.4f\n", value_3))


# -----------------------------------------------------------------
# 5. Stage 10 update  (pseudo-outcome = pseudo2 from fit3)
# -----------------------------------------------------------------
df_sim$Y2_pseudo <- pseudo2

Q2_init <- crossfit_Q_pseudo_stage(
  stage_df    = df_sim,
  Y_col       = "Y2_pseudo",
  A_col       = node_list_10$A,
  W_cols      = node_list_10$W,
  d_vec       = d2,
  folds_fixed = folds_fixed,
  learner     = Q_learner
)

opt2 <- Optimizer$new(
  treat_obs         = A2_int,
  y_obs             = pseudo2,
  y_pre             = Q2_init$QAW,
  y_global_min      = y_global_min,
  y_global_max      = y_global_max,
  propensity_matrix = as.matrix(ps_df2),
  OTR_pre           = d2,
  values_est_init   = Q2_init$Qd,
  prefix_w          = prefix_weights$prefix_w2
)
fit2    <- opt2$update()
pseudo1 <- fit2$Qd_star
value_2 <- fit2$psi_hat
cat(sprintf("Stage-10 updated value : %.4f\n", value_2))


# -----------------------------------------------------------------
# 6. Stage 9 update  (pseudo-outcome = pseudo1 from fit2)
# -----------------------------------------------------------------
df_sim$Y1_pseudo <- pseudo1

Q1_init <- crossfit_Q_pseudo_stage(
  stage_df    = df_sim,
  Y_col       = "Y1_pseudo",
  A_col       = node_list_09$A,
  W_cols      = node_list_09$W,
  d_vec       = d1,
  folds_fixed = folds_fixed,
  learner     = Q_learner
)

opt1 <- Optimizer$new(
  treat_obs         = A1_int,
  y_obs             = pseudo1,
  y_pre             = Q1_init$QAW,
  y_global_min      = y_global_min,
  y_global_max      = y_global_max,
  propensity_matrix = as.matrix(ps_df1),
  OTR_pre           = d1,
  values_est_init   = Q1_init$Qd,
  prefix_w          = prefix_weights$prefix_w1
)
fit1        <- opt1$update()
value_final <- fit1$psi_hat


# -----------------------------------------------------------------
# 7. Four-stage influence curve + final SE
# -----------------------------------------------------------------
IC_final <- fit4$H_obs * (df_sim$math_gpa_g12_cum - fit4$Qobs_star) +
            fit3$H_obs * (pseudo3                 - fit3$Qobs_star) +
            fit2$H_obs * (pseudo2                 - fit2$Qobs_star) +
            fit1$H_obs * (pseudo1                 - fit1$Qobs_star) +
            fit1$Qd_star - value_final

se_final <- sd(IC_final) / sqrt(nrow(df_sim))
obs_Y    <- mean(df_sim$math_gpa_g12_cum)

cat("\n=====================================================================\n")
cat(sprintf(" CONSTRAINED ODTR (V2a): final value-updated estimate\n"))
cat(sprintf("   psi_hat (V^d)      : %.4f\n", value_final))
cat(sprintf("   SE (influence fn)  : %.4f\n", se_final))
cat(sprintf("   95%% Wald CI        : [%.4f, %.4f]\n",
            value_final - 1.96 * se_final, value_final + 1.96 * se_final))
cat(sprintf("   Observed mean Y    : %.4f\n", obs_Y))
cat(sprintf("   Gain over observed : %.4f\n", value_final - obs_Y))
cat("=====================================================================\n")

# save the the 
constrained_otr_V2a <- data.frame(
  A1 = OTRs_feas_1,
  A2 = OTRs_feas_2,
  A3 = OTRs_feas_3,
  A4 = OTRs_feas_4
)

save(constrained_otr_V2a, file = "data/Demo_02_multistage_rec_synth_MA_constrained_260421_N10000_V2a.rdata")
# -----------------------------------------------------------------
# 8. Save
# -----------------------------------------------------------------
# save.image(file = "../data/tmle_results/02_1_training_value_updating_constrained_260421_N10000_V2a.rdata")