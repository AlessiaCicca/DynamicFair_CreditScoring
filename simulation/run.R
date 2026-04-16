# 1.Load file
source("utils.R")        
source("genvar.R")  
source("Timevarying_gnrt.R") 
source("traindtv_autocorr_gnrt.R")   
source("testdtv_gnrt.R")  

# 2. Create matsigma
matsigma <- create_matsigma()


scenarios <- c("fair", "direct", "proxy", "temporal")

for (sc in scenarios) {
  
  cat("Running scenario:", sc, "\n")
  
  result <- traindtv_autocorr_gnrt(nsub = 300, 
                                    matsigma = matsigma, 
                                    scenario = sc)

  test_list <- testdtv_gnrt(data = result$fullData, 
                             ntest = 50,        
                             id = "ID", 
                             period = "Time", 
                             y = "Event")
 

  write.csv(result$fullData, 
            file = paste0("data_", sc, ".csv"), 
            row.names = FALSE)
  
  cat("Saved: data_", sc, ".csv\n", sep = "")
  cat("Death rate:", result$Info$DRate, "\n\n")

  # Save each period as a separate csv to landmarking analysis
  for (t in 1:length(test_list)) {
    write.csv(test_list[[t]], 
              file = paste0("test_", sc, "_t", t, ".csv"), 
              row.names = FALSE)
  }
 
}

cat("Done!\n")
