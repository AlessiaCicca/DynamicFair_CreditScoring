#####################################################################################################
create_theta <- function(data, scenario, coeff){
  X <- data[, -ncol(data)] 
  S <- data[, ncol(data)] 
  if (scenario == "direct"){
    Fstar <- X %*% coeff$Beta1 + coeff$Beta0[1]
    Fstar <- Fstar + coeff$BetaS * S
  }
  else if (scenario == "fair" | scenario == "proxy" | scenario == "temporal"){
     Fstar <- X %*% coeff$Beta1 + coeff$Beta0[1] 
  } else {
    stop("Wrong model is set.")
  }
  return(exp(Fstar))
}

## Compute the cumulative hazards function
ExpHfunc <- function(ts1, ts2, theta, coeff){
  coeff$Lambda * theta * (ts1 - ts2)
}

## Compute the continuous times
Exptfunc <- function(tall, theta, coeff, t0, rid){
  t0 / coeff$Lambda / theta[rid] + tall[rid]
}


findsurvint <- function(y, nper, rate) {
  # INPUT # 
  # y = true survival times (not censored) from the existing DGP used in the previous paper
  # nper = the number of intervals / periods we want
  # rate = proportion of censoring at the end that we want 
  #        (we will also add censoring at any moment later with the DGP itself)
  
  # OUTPUT # 
  # The interval limits to define the periods
  
  int <- quantile(y, probs = seq((1 - rate) / nper, 1 - rate, length.out = nper))
  return(int)
}

#####################======== MAIN FUNCTION ===============#####################
tvstimegnrt <- function(nsub = 200, 
                        scenario = c("fair", "direct", "proxy", "temporal"), 
                        matsigma = NULL){

  nperiod <- 12
  Data <- matrix(NA, nperiod * nsub, 8)
  colnames(Data) <- c("ID","X1","X2","X3","X4","X5","X6","S")
  Data[, 1] <- rep(1:nsub, each = nperiod)
  Data[, 2:8] <- genvar(nsub = nsub, 
                        matsigma = matsigma,
                        scenario = scenario)
  
  ## Set the coefficients and compute the Theta = exp(f(X))
  coeffTS <- create_coeff(scenario = scenario, 
                          nsub = nsub)
  Coeff <- coeffTS$Coeff
  TS <- coeffTS$TS
  rm(coeffTS)
  
  Theta <- create_theta(data = Data[, 2:8], 
                        coeff = Coeff,
                        scenario = scenario)
                       
  Hfunc <- ExpHfunc
  tfunc <- Exptfunc
  
  tlen <- length(TS)
  seqt2 <- nperiod * c(1:(tlen / nperiod)) # each : -nperiod
  seqt1 <- nperiod * c(0:((tlen - 1) / nperiod)) + 1 # each : -1
  
  R <- Hfunc(ts1 = TS[-seqt1], 
             ts2 = TS[-seqt2], 
             theta = Theta[-seqt2], 
             coeff = Coeff)
  # each row belongs to a subject
  R <- matrix(R, ncol = nperiod - 1, byrow = TRUE)
  
  U <- runif(nsub)
  survtime <- rep(0, nsub)
  survnrow <- rep(0, nsub)
  for (Count in 1:nsub) {
    idxC <- which(Data[, "ID"] == Count)
    VEC <- c(0, cumsum(R[Count, ]), Inf)
    R.ID <- findInterval(-log(U[Count]), VEC)
    TT <- -log(U[Count]) - VEC[R.ID] 
    survnrow[Count] <- R.ID
    survtime[Count] <- tfunc(tall = TS[idxC], 
                             theta = Theta[idxC], 
                             coeff = Coeff, 
                             t0 = TT, 
                             rid = R.ID)
  }
  rm(R)
  rm(Theta)
  rm(U)
  gc()
  RET = list(survtime = survtime,
             coeff = Coeff)
  return(RET)
}

