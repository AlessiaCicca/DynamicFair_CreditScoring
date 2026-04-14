## Updated: July 8th
######################################################################################################
genvar <- function(nsub = 1000, 
                   matsigma = NULL,
                   scenario= c("fair", "direct", "proxy", "dynamic", "interaction"),
                   nperiod = c(4, 8),
                   ncov = 6,
                   ncovfixed = 2){
  #######################################################################
  # GOAL #
  # Produce dataset of subject size nsub, with ncov number of covariates, 
  # with ncovfixed number of time-invariant covariates 
  # The values of covariates are correlated.
  # In addition, the values of time-varying covariates are autoregressive.
  # The coefficients are determined by a autocorrelation matrix 
  #
  # DETAIL #
  # Note that, this function is only worked for ncov = 6 and ncovfixed = 2
  #
  # INPUT # 
  # nsub = number of subjects
  # nepriod = number of periods
  # matsigma = matrix for the VAR process, of size ncov x ncov
  #            (Z_t = matsigma Z_{t-1} + epsilon_t; 
  #             where epsilon_t are iid standard multivariate normal)
  # 
  # OUTPUT # 
  # A dataframe with nperiod * nsub rows and 6 columns 
  #######################################################################
  #Definition of scenario: NumberOfTimeInvariante and NumberOfTimeVariant
  if (scenario == "0TI2TV") {
   
  } else if (scenario %in% c("0TI4TV", "1TI4TV", "2TI4TV")) {
    ## First is to generate a normal VAR process
    # z0 = e_0                                                       # Time 0: e is of size ncov times nsub
    # z1 = A * e_0 + e_1                                             # Time 1: e is of size ncov times nsub
    # z2 = A * z1 + e2 = A^2 * e_0 +   A * e_1 +     e_2             # Time 2: e is of size ncov times nsub
    # z3 = A * z2 + e3 = A^3 * e_0 + A^2 * e_1 + A * e_2 + e_3       # Time 3: e is of size ncov times nsub
    
    z <- rep(list(0), nperiod)
    ## Each of size ncov x nsub, for ncov covariates and nsub subjects
    z[[1]] <- matrix(rnorm(ncov * nsub), nrow = ncov, ncol = nsub)   
    n1seq <- ncovfixed * nsub
    n2seq <- (ncov - ncovfixed) * nsub
    
    for (pp in 2:nperiod) {
      z[[pp]] <- matsigma %*% z[[pp - 1]] + matrix(c(rep(0, n1seq), rnorm(n2seq)), nrow = ncov, ncol = nsub, byrow = TRUE)    
    }
    
    Data <- matrix(NA, nrow = nperiod * nsub, ncol = ncov)
    colnames(Data) <- paste0("X", 1:ncov)
    rownames(Data) <- rep(1:nsub, each = nperiod)
    
    ## Transform the normal variables to mimic the Yao et al. (2020) covariates.
    Data[, -c(1:ncovfixed)] <- sapply((ncovfixed + 1):ncov, function(jj) as.vector(t(sapply(z, function(zz) zz[jj, ]))))
    
    Data[, "X1"] <- rep(as.numeric(z[[1]][1, ] > 0), each = nperiod)
    Data[, "X2"] <- rep(pnorm(z[[1]][2, ]), each = nperiod)
    rm(z)
    Data[, "X3"] <- as.numeric(Data[, "X3"] > 0)
    Data[, "X4"] <- pnorm(Data[, "X4"])
    Data[, "X5"] <- (Data[, "X5"] < qnorm(.2)) + 
      2 * (Data[, "X5"] >= qnorm(.2)) * (Data[, "X5"] < qnorm(.4)) + 
      3 * (Data[, "X5"] >= qnorm(.4)) * (Data[, "X5"] < qnorm(.6)) + 
      4 * (Data[, "X5"] >= qnorm(.6)) * (Data[, "X5"] < qnorm(.8)) +
      5 * (Data[, "X5"] >= qnorm(.8))
    Data[, "X6"] <- pnorm(Data[, "X6"]) * 2
  }
  
  return(Data)
}







######################################################################################################
