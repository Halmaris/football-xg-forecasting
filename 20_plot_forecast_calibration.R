# Rscript --vanilla 20_plot_forecast_calibration.R [diagnostics_dir]
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args)) args[1] else 'results/diagnostics'
df <- read.csv(file.path(output_dir, 'forecast_binned_means.csv'))
stopifnot(nrow(df) == 40L, all(is.finite(df$forecast)), all(is.finite(df$observed)))
lims <- range(c(df$forecast, df$observed)) + c(-0.08, 0.08)
cairo_pdf(file.path(output_dir, 'forecast_calibration.pdf'), width = 8.5, height = 3.55,
  family = 'Arial', pointsize = 10)
par(mfrow = c(1, 2), mar = c(4.0, 4.0, 2.5, 0.8), mgp = c(2.4, 0.7, 0),
  tcl = -0.25, las = 1, bty = 'l')
for (i in seq_along(c('LMM', 'XGBoost'))) {
  model <- c('LMM', 'XGBoost')[i]
  plot(NA, xlim = lims, ylim = lims, xlab = 'Mean forecast xG',
    ylab = if (i == 1L) 'Mean observed xG' else '', axes = FALSE)
  abline(h = pretty(lims), v = pretty(lims), col = '#EEEEEE', lwd = 0.5)
  abline(a = 0, b = 1, lty = 2, col = '#888888', lwd = 0.8)
  for (j in 1:2) {
    d <- df[df$model == paste0(model, c('', ' smearing')[j]), ]
    lines(d$forecast, d$observed, type = 'o', pch = c(16, 15)[j],
      col = c('#0057B8', '#D55E00')[j], lwd = 1.1, cex = 0.7)
  }
  axis(1); axis(2); box(bty = 'l')
  title(main = paste(c('(A)', '(B)')[i], model), adj = 0, cex.main = 1.05)
  legend('topleft', c('Uncorrected', 'Smearing'), col = c('#0057B8', '#D55E00'),
    pch = c(16, 15), lty = 1, lwd = 1.1, bty = 'n', cex = 0.85)
}
invisible(dev.off())
