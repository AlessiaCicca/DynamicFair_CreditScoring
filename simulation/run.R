#I CAMBIAMENTI SONO STATI IL t DA QUALCHE PARTE IL RoW TRUE IN GENVAR E LO 0.10 DEL INVARIANT
#bisogna controlalre il senso di censor.rate inf

install.packages("openxlsx")
library(openxlsx)
library(truncdist)

source("simulation/utils.R")        
source("simulation/genvar.R")  
source("simulation/timevarying_gnrt.R") 
source("simulation/traindtv_autocorr_gnrt.R")   
source("simulation/testdtv_gnrt.R")  

matsigma  <- create_matsigma()
scenarios <- c("fair", "direct", "proxy", "temporal")

wb <- createWorkbook()

for (sc in scenarios) {
  
  cat("Running scenario:", sc, "\n")
  
  result <- traindtv_autocorr_gnrt(nsub = 1000, 
                                   matsigma = matsigma, 
                                   scenario = sc)
  

  df        <- result$fullData
  df_sorted <- df[order(df$ID, df$Time), ]
  df_id     <- df_sorted[!duplicated(df_sorted$ID, fromLast = TRUE), ]
  counts    <- table(df_id$S, df_id$Event)
  percent   <- prop.table(counts, margin = 1) * 100
  
  cat("\n=== EVENT COUNTS ===\n");      print(counts)
  cat("\n=== EVENT PERCENTAGES ===\n"); print(round(percent, 2))
  

  df_counts  <- as.data.frame.matrix(counts)
  df_percent <- as.data.frame.matrix(round(percent, 2))
  df_counts$S  <- rownames(df_counts);  df_counts  <- df_counts[, c("S", setdiff(names(df_counts), "S"))]
  df_percent$S <- rownames(df_percent); df_percent <- df_percent[, c("S", setdiff(names(df_percent), "S"))]
  

  df_info <- data.frame(Metric = "Death Rate", Value = result$Info$DRate)
  cat("Death rate:", result$Info$DRate, "\n\n")
  
  # ---- COEFF----
  coefficients <- create_coeff(scenario = sc,  
                               nsub = 1000)
  coeff_list <- coefficients$Coeff
  df_coeff <- do.call(rbind, lapply(names(coeff_list), function(nm) {
    vals <- as.vector(coeff_list[[nm]])
    data.frame(
      Coefficient = if (length(vals) == 1) nm else paste0(nm, "_", seq_along(vals)),
      Value       = vals,
      stringsAsFactors = FALSE
    )
  }))
  
  # ---- EXCEL ----
  addWorksheet(wb, sheetName = sc)
  writeData(wb, sc, "SCENARIO INFO",        startRow = 1,  startCol = 1)
  writeData(wb, sc, df_info,                startRow = 2,  startCol = 1)
  writeData(wb, sc, "EVENT COUNTS",         startRow = 6,  startCol = 1)
  writeData(wb, sc, df_counts,              startRow = 7,  startCol = 1)
  row_pct <- 7 + nrow(df_counts) + 2
  writeData(wb, sc, "EVENT PERCENTAGES (%)", startRow = row_pct,     startCol = 1)
  writeData(wb, sc, df_percent,              startRow = row_pct + 1, startCol = 1)
  row_coeff <- row_pct + nrow(df_percent) + 3
  writeData(wb, sc, "COEFFICIENTS",         startRow = row_coeff,     startCol = 1)
  writeData(wb, sc, df_coeff,               startRow = row_coeff + 1, startCol = 1)
  bold_style <- createStyle(textDecoration = "bold", fontSize = 12)
  addStyle(wb, sc, bold_style, rows = 1,          cols = 1)
  addStyle(wb, sc, bold_style, rows = 6,          cols = 1)
  addStyle(wb, sc, bold_style, rows = row_pct,    cols = 1)
  addStyle(wb, sc, bold_style, rows = row_coeff,  cols = 1)
  data_file <- paste0("data_", sc, ".csv")
  write.csv(result$fullData, file = data_file, row.names = FALSE)
  cat("Saved:", data_file, "\n")
  train_sheet <- paste0("train_", sc)
  addWorksheet(wb, train_sheet)
  writeData(wb, train_sheet, result$fullData)
  
  
  # ---- TEST DATA ----
  test_list <- testdtv_gnrt(data = result$fullData, ntest  = 100, id = "ID", period = "Time", y  = "Event")
  for (t in seq_along(test_list)) {
    write.csv(test_list[[t]], 
              file = paste0("test_", sc, "_t", t, ".csv"),row.names = FALSE)
  }
}


excel_file <- "simulation_results.xlsx"
saveWorkbook(wb, excel_file, overwrite = TRUE)
cat("\nDone! File saved in", excel_file, "\n")
shell.exec(normalizePath(excel_file))  

