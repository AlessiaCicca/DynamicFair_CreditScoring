# 1.Load file
source("utils.R")        
source("genera_dati.R")  
source("time_varying.R") 
source("train.R")        

# 2. Create matsigma
matsigma <- create_matsigma()


scenarios <- c("fair", "direct", "proxy", "temporal")

for (sc in scenarios) {
  
  cat("Running scenario:", sc, "\n")
  
  result <- traindtv_autocorr_gnrt(nsub = 200, 
                                    matsigma = matsigma, 
                                    scenario = sc)
  

  write.csv(result$fullData, 
            file = paste0("data_", sc, ".csv"), 
            row.names = FALSE)
  
  cat("Saved: data_", sc, ".csv\n", sep = "")
  cat("Death rate:", result$Info$DRate, "\n\n")
}

cat("Done!\n")
