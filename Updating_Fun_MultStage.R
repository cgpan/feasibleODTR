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
