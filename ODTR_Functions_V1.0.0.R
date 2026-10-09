# date: 2026.01.03  
# purpose:  
#  - this file ccontains key functions to extract the key info for:
#    policy learning and value updating in longitudinal setting



extract_tmle_components <- function(node_list, tmle_fit, tmle_spec, tmle_task, treatment,original_data) {
  
  # extract the treatment node
  A_node <- node_list$A
  
  
  # --------------------- Block 2 Setup: LF & Levels -------------------------
  lf_A <- tmle_fit$likelihood$factor_list[["A"]]
  lf_Y <- tmle_fit$likelihood$factor_list[["Y"]]
  
  # Get levels from the training task within the learner
  A_levels <- lf_A$learner$training_task$outcome_type$levels
  a_levels_count <- length(A_levels)
  
  message("Step 1/6: Extracting CVed blip matrix blip_df...")
  # ----------------------- Block 1: CV-ed Blip matrix -------------------------
  blip_mat_cv <- tmle_spec$get_blip_pred(tmle_task, fold_number = "validation")
  
  blip_cv_mat <- t(vapply(
    blip_mat_cv,
    function(x) unclass(x[[1]]),
    FUN.VALUE = numeric(a_levels_count)
  ))
  
  colnames(blip_cv_mat) <- names(blip_mat_cv[[1]][[1]])
  blip_df <- as.data.frame(blip_cv_mat, check.names = FALSE)
  
  
  message("Step 2/6: Extracting CVed PS matrix ps_df...")
  # ----------------------- Block 2: CV-ed PS Matrix ---------------------------
  # Get regression task for A (time variant true ensures strictly standard format)
  g_task <- tmle_task$get_regression_task("A", is_time_variant = TRUE)
  
  pred_g <- lf_A$learner$predict_fold(g_task, fold_number = "validation")
  
  g_mat <- as.data.frame(sl3::unpack_predictions(as.vector(pred_g)))
  
  # Enforce level order based on the task's variable type
  lev <- tmle_task$npsem$A$variable_type$levels
  g_mat <- as.matrix(g_mat[, lev, drop = FALSE]) 
  
  ps_df <- as.data.frame(g_mat)
  
  message("Step 3/6: Extracting Non-CVed PS matrix ps_df...")
  # ----------------------- Block 2.5: Non-CV-ed PS_noCV_df ---------------------
  # Used for feasibility trimming
  g_task_nocv <- tmle_task$get_regression_task("A")
  g_learner <- tmle_fit$likelihood$factor_list[["A"]]$learner
  g_mat_noCV <- sl3::unpack_predictions(g_learner$predict(g_task_nocv))
  ps_noCV_df <- as.data.frame(g_mat_noCV[, lev, drop = FALSE])
  
  message("Step 4/6: Extracting CV-ed predicted outcome Q(A|W)...")
  # ----------------------- Block 3: CV-ed predicted CFs (Observed A) ----------
  QAW_pred <- lf_Y$get_likelihood(tmle_task, fold_number = "validation")
  QAW <- as.numeric(QAW_pred)
  
  message("Step 5/6: Extracting CVed predicted CFs df...")
  # ----------------------- Block 4: CV-ed predicted CFs Matrix ----------------
  
  
  # Loop through levels to create counterfactual tasks
  cf_mat <- sapply(A_levels, function(a) {
    cf_data <- data.table::copy(original_data)
    
    # Force column assignment
    set(cf_data, j = A_node, value = factor(a, levels = A_levels))
    
    # Create new task reusing the original folds and graph
    cf_task <- tmle3::tmle3_Task$new(
      cf_data, 
      npsem = tmle_task$npsem, 
      folds = tmle_task$folds
    )
    
    as.numeric(lf_Y$get_likelihood(cf_task, fold_number = "validation"))
  })
  
  cf_matrix_original_scale <- as.data.frame(cf_mat)
  
  # ---------------Block 5: retrieve min max of outcome---------------
  message("Step 6/6: Retrieving outcome bounds...")
  outcome_node <- tmle_task$npsem$Y
  outcome_bounds <- outcome_node$variable_type$bounds
  Y_min <- outcome_bounds[1]
  Y_max <- outcome_bounds[2]
  
  message("...Success!!")
  
  # ----------------------- Return List ----------------------------------------
  return(list(
    blip_df = blip_df,
    ps_df = ps_df,
    ps_noCV_df = ps_noCV_df,
    QAW = QAW,
    cf_df = cf_matrix_original_scale,
    Y_min = Y_min,
    Y_max = Y_max
  ))
}


library(R6)

Optimizer <- R6Class(
  "Optimizer",
  public = list(
    treat_obs = NULL, # observed treatment assignments, integer vector
    OTR_pre = NULL,   # rule recommendations d_t(H_t), integer vector
    values_est_init = NULL, # initial Q_t^0(H_t, d_t)
    y_obs = NULL,   # outcome used at this stage (Y or pseudo-outcome)
    y_obs_min = NULL,  # global lower bound
    y_obs_max = NULL,  # global upper bound
    propensity_matrix = NULL, # PS matrix: n x K_t, row i is g_t(.|H_ti)
    y_pre = NULL,  # initial Q_t^0(H_t, A_t)
    bound_eps = 0.005,
    obs_num = NULL,
    
    # NEW: prefix weights W_{t-1}
    prefix_w = NULL,
    
    initialize = function(treat_obs,
                          y_obs,
                          y_pre,
                          y_global_min,
                          y_global_max,
                          propensity_matrix,
                          OTR_pre,
                          values_est_init,
                          prefix_w = NULL) {
      
      self$treat_obs <- treat_obs
      self$OTR_pre <- OTR_pre
      self$values_est_init <- values_est_init
      self$y_obs <- y_obs
      self$y_obs_min <- y_global_min
      self$y_obs_max <- y_global_max
      self$propensity_matrix <- propensity_matrix
      self$obs_num <- length(treat_obs)
      self$y_pre <- y_pre
      
      # NEW: set prefix_w (defaults to all 1s for stage 1)
      if (is.null(prefix_w)) {
        self$prefix_w <- rep(1, self$obs_num)
      } else {
        if (length(prefix_w) != self$obs_num) stop("prefix_w must have length n.")
        self$prefix_w <- prefix_w
      }
    },
    
    .bound01 = function(p, eps = self$bound_eps) pmin(pmax(p, eps), 1 - eps),
    .scale01 = function(y, lower, upper) (y - lower) / (upper - lower),
    .unscale01 = function(p, lower, upper) p * (upper - lower) + lower,
    
    # tmle3 bounds clever covariate to [-40, 40]
    get_H_obs = function() {
      n <- self$obs_num
      g_obs <- self$propensity_matrix[cbind(seq_len(n), self$treat_obs)]
      H <- self$prefix_w * (as.numeric(self$treat_obs == self$OTR_pre) / g_obs)
      pmin(pmax(H, -40), 40)
    },
    
    get_H_cf = function() {
      n <- self$obs_num
      g_d <- self$propensity_matrix[cbind(seq_len(n), self$OTR_pre)]
      H <- self$prefix_w * (1 / g_d)
      pmin(pmax(H, -40), 40)
    },
    
    update = function() {
      n <- self$obs_num
      
      message("Step 1/5: Getting the clever covariate(s)...")
      H_obs <- self$get_H_obs()
      H_cf  <- self$get_H_cf()
      lower <- self$y_obs_min
      upper <- self$y_obs_max
      
      if (upper == lower) stop("Outcome has zero range; cannot scale.")
      
      message("Step 2/5: Scaling the initial QAW and QdW...")
      
      y_scaled    <- self$.scale01(self$y_obs, lower, upper)
      Qobs_scaled <- self$.bound01(self$.scale01(self$y_pre, lower, upper))
      Qd_scaled   <- self$.bound01(self$.scale01(self$values_est_init, lower, upper))
      
      if (any(y_scaled < 0 | y_scaled > 1, na.rm = TRUE)) {
        stop("y_scaled outside [0,1]. Your y_obs bounds do not match y_obs.")
      }
      
      message("Step 3/5: Fitting the fluc model...")
      fit <- suppressWarnings(glm(
        y_scaled ~ H_obs - 1,
        family = binomial(),
        offset = qlogis(Qobs_scaled)
      ))
      
      eps <- unname(coef(fit)[1])
      if (is.na(eps)) eps <- 0
      eps_vector <- rep(eps, n)
      
      message("Step 4/5: Getting the updated estimates...")
      Qobs_star_scaled <- plogis(qlogis(Qobs_scaled) + eps * H_obs)
      Qd_star_scaled   <- plogis(qlogis(Qd_scaled)   + eps * H_cf)
      
      Qobs_star <- self$.unscale01(Qobs_star_scaled, lower, upper)
      Qd_star   <- self$.unscale01(Qd_star_scaled,   lower, upper)
      
      psi_hat <- mean(Qd_star)
      
      IC <- H_obs * (self$y_obs - Qobs_star) + Qd_star - psi_hat
      se <- sd(IC) / sqrt(n)
      
      message("Step 5/5: Done!")
      
      list(
        psi_hat = psi_hat,
        se = se,
        IC = IC,
        Qobs_star = Qobs_star,
        Qd_star = Qd_star,
        H_obs = H_obs,
        H_cf = H_cf,
        eps = eps_vector
      )
    }
  )
)


# helper: robustly pull the sl3 fit out of a likelihood factor
get_sl_fit <- function(lf) {
  out <- tryCatch(lf$fit_object, error = function(e) NULL)
  if (!is.null(out)) return(out)
  out <- tryCatch(lf$learner, error = function(e) NULL)
  if (!is.null(out)) return(out)
  stop("Could not find the learner fit inside this likelihood factor. Try str(lf).")
}

# 1) extract fitted SL objects for Y and A
#lf_Y <- fit_cat3$likelihood$factor_list[["Y"]]
#sl_Y <- get_sl_fit(lf_Y)
#w_Y <- sl_Y$coefficients

suppressPackageStartupMessages({
  library(sl3)
  library(origami)
})

# Helper to extract train/valid indices from either:
#  (a) origami folds objects, or
#  (b) a simple list of validation indices
.get_fold_idx <- function(folds, k, n) {
  fk <- folds[[k]]
  # origami fold case
  idx_valid <- tryCatch(origami::validation_set(fk), error = function(e) NULL)
  idx_train <- tryCatch(origami::training_set(fk),   error = function(e) NULL)
  if (!is.null(idx_valid) && !is.null(idx_train)) {
    return(list(train = idx_train, valid = idx_valid))
  }
  # list-of-validation-indices case
  if (is.integer(fk) || is.numeric(fk)) {
    idx_valid <- as.integer(fk)
    idx_train <- setdiff(seq_len(n), idx_valid)
    return(list(train = idx_train, valid = idx_valid))
  }
  stop("Unsupported folds format. Provide origami folds or list of validation indices.")
}


# ---------------------------------------------------------
#     Non-parallel and parallel cross-fitted Q functions
# --------------------------------------------------------

# Cross-fitted regression for Q(H,A) and Q(H,d): non parallel
crossfit_Q_pseudo_stage <- function(stage_df, Y_col, A_col, W_cols,
                                    d_vec, folds_fixed, learner) {
  n <- nrow(stage_df)
  stopifnot(length(d_vec) == n)
  
  # Ensure A is a factor with stable levels across folds
  if (!is.factor(stage_df[[A_col]])) stage_df[[A_col]] <- factor(stage_df[[A_col]])
  A_levels <- levels(stage_df[[A_col]])
  
  # Coerce d_vec to same coding as A
  # If your A is factor like "1","2",... and d_vec is integer 1..K, this is ok:
  if (is.numeric(d_vec) || is.integer(d_vec)) {
    dA <- factor(as.character(d_vec), levels = A_levels)
    # If your A levels are actually integers stored as characters, as.character works.
    # If not, change the mapping accordingly.
  } else {
    dA <- factor(d_vec, levels = A_levels)
  }
  if (anyNA(dA)) stop("d_vec cannot be matched to A levels. Check coding of d_vec vs levels(stage_df[[A_col]]).")
  
  QAW <- rep(NA_real_, n)
  Qd  <- rep(NA_real_, n)
  
  for (k in seq_along(folds_fixed)) {
    
    message("Fitting fold ", k, " of ", length(folds_fixed), "...\n")
    fold_k <- folds_fixed[[k]]
    tr <- fold_k$training_set
    va <- fold_k$validation_set
    
    # Fit on training fold
    message("  Training on ", length(tr), " observations...\n")
    time_k_0 <- Sys.time()
    task_tr <- sl3_Task$new(
      data       = stage_df[tr, , drop = FALSE],
      covariates = c(W_cols, A_col),
      outcome    = Y_col
    )
    fit_k <- learner$train(task_tr)
    
    # Predict on validation fold at observed A
    message("  Predicting QAW on validation ...\n")
    task_va_obs <- sl3_Task$new(
      data       = stage_df[va, , drop = FALSE],
      covariates = c(W_cols, A_col),
      outcome    = Y_col
    )
    QAW[va] <- as.numeric(fit_k$predict(task_va_obs))
    
    # Predict on validation fold at rule A = dA
    message("  Predicting Qd on validation ...\n")
    stage_va_d <- stage_df[va, , drop = FALSE]
    stage_va_d[[A_col]] <- dA[va]
    task_va_d <- sl3_Task$new(
      data       = stage_va_d,
      covariates = c(W_cols, A_col),
      outcome    = Y_col
    )
    Qd[va] <- as.numeric(fit_k$predict(task_va_d))
    
    time_k_1 <- Sys.time()
    message("  Fold ", k, " done in ", round(difftime(time_k_1, time_k_0, units = "secs"), 2), " seconds.\n")
  }
  
  if (anyNA(QAW) || anyNA(Qd)) stop("Missing fold predictions. Check folds_fixed coverage.")
  list(QAW = QAW, Qd = Qd)
}

# --------------------------------------------------------
#     Parallel cross-fitted Q functions
# --------------------------------------------------------
library(foreach)
library(doParallel)

crossfit_Q_pseudo_stage_parallel <- function(stage_df, Y_col, A_col, W_cols, 
                                             d_vec, folds_fixed, learner, 
                                             n_cores = parallel::detectCores() - 1) {
  n <- nrow(stage_df)
  
  # Ensure A is a factor
  if (!is.factor(stage_df[[A_col]])) stage_df[[A_col]] <- factor(stage_df[[A_col]])
  A_levels <- levels(stage_df[[A_col]])
  
  # Coerce d_vec
  dA <- factor(as.character(d_vec), levels = A_levels)
  if (anyNA(dA)) stop("d_vec cannot be matched to A levels.")
  
  # Set up parallel backend
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  on.exit(stopCluster(cl)) # Ensure cluster stops even if code fails
  
  # Run folds in parallel
  # We use .export to ensure the worker nodes see the sl3 objects and data
  results <- foreach(k = seq_along(folds_fixed), 
                     .packages = c("sl3"),
                     .combine = 'rbind') %dopar% {
                       
                       fold_k <- folds_fixed[[k]]
                       tr <- fold_k$training_set
                       va <- fold_k$validation_set
                       
                       message("Processing fold ", k, "...\n")
                       # Fit on training fold
                       task_tr <- sl3_Task$new(
                         data       = stage_df[tr, , drop = FALSE],
                         covariates = c(W_cols, A_col),
                         outcome    = Y_col
                       )
                       fit_k <- learner$train(task_tr)
                       
                       # Predict observed
                       task_va_obs <- sl3_Task$new(
                         data       = stage_df[va, , drop = FALSE],
                         covariates = c(W_cols, A_col),
                         outcome    = Y_col
                       )
                       pred_QAW <- as.numeric(fit_k$predict(task_va_obs))
                       
                       # Predict under rule
                       stage_va_d <- stage_df[va, , drop = FALSE]
                       stage_va_d[[A_col]] <- dA[va]
                       task_va_d <- sl3_Task$new(
                         data       = stage_va_d,
                         covariates = c(W_cols, A_col),
                         outcome    = Y_col
                       )
                       pred_Qd <- as.numeric(fit_k$predict(task_va_d))
                       
                       # Return data frame of predictions and their original indices
                       data.frame(idx = va, QAW = pred_QAW, Qd = pred_Qd)
                     }
  
  # Re-order results to match original data order
  results <- results[order(results$idx), ]
  
  list(QAW = results$QAW, Qd = results$Qd)
}

