
library(R6)

Optimizer <- R6Class(
  "Optimizer",
  public = list(
    treat_obs = NULL, # observed treatment assignments,a integar arrary
    OTR_pre = NULL,   # optimal treatment recommendations, an integer array
    values_est_init = NULL, # CV-ed CFs under the OTRs
    y_obs = NULL,   # observed outcomes
    y_obs_min = NULL,  # global min of observed outcomes
    y_obs_max = NULL,  # global max of observed outcomes
    propensity_matrix = NULL, # CV-ed propensity scores matrix
    y_pre = NULL,  # CV-ed predicted CFs under observed treatments
    bound_eps = 0.005, # epsilon for bounding qlogis() away from 0 and 1
    obs_num = NULL, # number of observations, you do need to specify
    
    initialize = function(treat_obs,
                          y_obs,
                          y_pre,
                          y_global_min,
                          y_global_max,
                          propensity_matrix,
                          OTR_pre,
                          values_est_init) {
      self$treat_obs <- treat_obs
      self$OTR_pre <- OTR_pre
      self$values_est_init <- values_est_init
      self$y_obs <- y_obs
      self$y_obs_min <- y_global_min
      self$y_obs_max <- y_global_max
      self$propensity_matrix <- propensity_matrix
      self$obs_num <- length(treat_obs)
      self$y_pre <- y_pre
    },
    
    .bound01 = function(p, eps = self$bound_eps) pmin(pmax(p, eps), 1 - eps),
    
    .scale01 = function(y, lower, upper) (y - lower) / (upper - lower),
    
    .unscale01 = function(p, lower, upper) p * (upper - lower) + lower,
    
    # tmle3::Param_TSM bounds clever covariate to [-40, 40]
    get_H_obs = function() {
      n <- self$obs_num
      g_obs <- self$propensity_matrix[cbind(seq_len(n), self$treat_obs)]
      H <- as.numeric(self$treat_obs == self$OTR_pre) / g_obs
      pmin(pmax(H, -40), 40)
    },
    
    get_H_cf = function() {
      n <- self$obs_num
      g_d <- self$propensity_matrix[cbind(seq_len(n), self$OTR_pre)]
      H <- 1 / g_d
      pmin(pmax(H, -40), 40)
    },
    # 
    # Consolidated Update Method
    update = function() {
      n <- self$obs_num
      
      # 1. Prepare Data
      H_obs <- self$get_H_obs()
      H_cf  <- self$get_H_cf()
      lower <- self$y_obs_min
      upper <- self$y_obs_max
      
      if (upper == lower) stop("Outcome has zero range; cannot scale.")
      
      y_scaled    <- self$.scale01(self$y_obs, lower, upper)
      Qobs_scaled <- self$.bound01(self$.scale01(self$y_pre, lower, upper))
      Qd_scaled   <- self$.bound01(self$.scale01(self$values_est_init, lower, upper))
      
      if (any(y_scaled < 0 | y_scaled > 1, na.rm = TRUE)) {
        stop("y_scaled outside [0,1]. Your y_obs bounds do not match y_obs.")
      }
      
      # Vectors to store updated values
      Qobs_star_scaled <- rep(NA_real_, n)
      Qd_star_scaled   <- rep(NA_real_, n)
      eps_vector       <- rep(NA_real_, n) # specific epsilon used for each row
      
      # 2. Fit and Update based on Approach
      # --- POOLED UPDATE (Standard TMLE3) ---
      # Fits one global epsilon using all stacked CV predictions
      fit <- suppressWarnings(glm(
        y_scaled ~ H_obs - 1,
        family = binomial(),
        offset = qlogis(Qobs_scaled)
      ))
      eps <- unname(coef(fit)[1])
      if (is.na(eps)) eps <- 0
      
      # Update everyone with the same epsilon
      Qobs_star_scaled <- plogis(qlogis(Qobs_scaled) + eps * H_obs)
      Qd_star_scaled   <- plogis(qlogis(Qd_scaled)   + eps * H_cf)
      eps_vector[]     <- eps
      
      # 3. Finalize and Unscale
      Qobs_star <- self$.unscale01(Qobs_star_scaled, lower, upper)
      Qd_star   <- self$.unscale01(Qd_star_scaled,   lower, upper)
      
      # --- Two equivalent ways to compute psi_hat (test) ---
      psi_hat <- mean(Qd_star)
      
      # IC/SE with the chosen psi_hat
      IC <- H_obs * (self$y_obs - Qobs_star) + Qd_star - psi_hat
      se <- sd(IC) / sqrt(n)
      
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