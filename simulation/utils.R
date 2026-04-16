###### ======== autocorrelation: matsigma in traindtv_autocorr_gnrt.R ======== ######
# Mantain 2TI4TV + Strong (to financial domain)
create_matsigma <- function(){
  matsigma <- .7 * diag(6) + matrix(.2, 6, 6)
  # to make the first two variables time-independent
  matsigma[1, ] <- c(1, 0, 0, 0, 0, 0)
  matsigma[2, ] <- c(0, 1, 0, 0, 0, 0) 
  return(matsigma)
}


###### ======== censoring rate: Censor.time in traindtv_autocorr_gnrt.R ======== ######
# Mantain nperiod=8, model=linear and distribution=Exp (easiest), censor.rate = 10% (more realistic) 
# and SNR (How strong is the signal compared to the noise) = "low" (to financial domain)
create_ctime <- function(nsub){
  Censor.time <- rep(Inf, nsub)
  return(Censor.time)
}

###### ======== create coefficient lists: Coeff in Timevarying_gnrt.R ======== ######
# Mantain nperiod=8 (using 12), model=linear and distribution=Exp (easiest), censor.rate = 10% (more realistic) 
# and SNR (How strong is the signal compared to the noise) = "low" (to financial domain)
# Add 4 scenario:
# 
# Z_t = matsigma Z_{t-1} + Gamma * S + epsilon_t; 
# theta = exp(data * Beta1 + Beta0 + S * BetaS)
#
# FAIR: No discrimination -> BetaS=0 Gamma=0
# DIRECT: Sensitive variable directly influences risk Theta -> BetaS=0.3 Gamma=0
# PROXY: Sensitive variable influences covariates Zt -> BetaS=0 Gamma=0.3
# TEMPORAL:  Sensitive variable influences covariates Zt and bias increases over time -> BetaS=0 Gamma=0.3 (later * t)

create_coeff <- function(nsub, scenario){
  nperiod <- 12
  Beta1 <- c(1, -1, 1, -1, -0.25, 0.5) 
  Lambda = 0.5
  Alpha = 0
  V = 0
  Beta0 = -5
  TS <- as.vector(replicate(nsub, 
                            c(0, sort(rtrunc(nperiod - 1, spec = "beta", a = 0.0001, b = 1, shape1 = 0.1, shape2 = 2)) * 900)
  ))
  if(scenario == "fair") {
    Gamma=0 
    BetaS=0
  }
  else if(scenario == "direct"){
    Gamma=0
    BetaS=1
  }
  else if(scenario == "proxy" | scenario == "temporal"){
    Gamma= 0.3  # X2↓, X4↓, X6↑ → tutti aumentano Fstar per S=1
    BetaS=0
  }
  else {
      stop("Wrong scenario is specified.")
  }
  Coeff <- list(Lambda = Lambda, Alpha = Alpha, V = V, 
                Beta1 = Beta1, Beta0 = Beta0, BetaS=BetaS, Gamma=Gamma)
  return(list(TS = TS, Coeff = Coeff))
}

