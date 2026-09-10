# Usage: Rscript --vanilla 17_plot_diagnostics_stability.R results_dir
# Publication graphics and table from saved numerical results.

render_diagnostics_stability <- function(output_dir) {
  out <- function(name) file.path(output_dir, name)
  read_result <- function(name) read.csv(out(name), check.names = FALSE)
  ink <- '#222222'
  gray <- '#888888'
  league_colors <- c('Ekstraklasa' = '#0072B2', 'La Liga 2' = '#D55E00')
  start_plot <- function(name, height, title) {
    pdf(out(name), width = 9.1, height = height, family = 'Helvetica',
      useDingbats = FALSE, title = title)
    par(mfrow = c(1, 2), mar = c(4.2, 4.3, 2.1, 0.9), mgp = c(2.6, 0.7, 0),
      cex = 0.86, col.axis = ink, col.lab = ink, col.main = ink, fg = ink)
  }

  df_resid <- read_result('lmm_conditional_residuals.csv')
  z <- df_resid$standardized_residual
  qq_indices <- unique(round(seq(1, length(z), length.out = 3000)))
  start_plot('lmm_residual_diagnostics.pdf', 3.8,
    'Conditional residual diagnostics for the mixed-effects model')
  plot(qnorm(ppoints(length(z)))[qq_indices], sort(z)[qq_indices], pch = 16,
    cex = 0.42, col = adjustcolor(ink, alpha.f = 0.55),
    xlab = 'Standard normal quantile', ylab = 'Conditional residual / residual SD',
    main = '(A)  Normal Q-Q plot')
  abline(a = 0, b = 1, col = gray, lty = 2, lwd = 1.1)
  plot(df_resid$fitted_log, df_resid$residual_log, pch = 16, cex = 0.25,
    col = adjustcolor(ink, alpha.f = 0.11), xaxs = 'i',
    xlab = expression('Fitted ' * log(1 + xG^F)), ylab = 'Conditional residual',
    main = '(B)  Residuals versus fitted values')
  abline(h = 0, col = gray, lty = 2)
  smooth_x <- seq(min(df_resid$fitted_log), max(df_resid$fitted_log), length.out = 401)
  bandwidth <- 0.10
  # Gaussian weights estimate the local mean and SD from all observed residuals.
  smooth <- vapply(smooth_x, function(x) {
    weights <- exp(-0.5 * ((df_resid$fitted_log - x) / bandwidth)^2)
    weights <- weights / sum(weights)
    mean <- sum(weights * df_resid$residual_log)
    c(mean = mean, sd = sqrt(sum(weights * (df_resid$residual_log - mean)^2)))
  }, numeric(2))
  stopifnot(all(is.finite(smooth)), all(smooth['sd', ] > 0))
  lines(smooth_x, smooth['mean', ], col = 'black', lty = 1, lwd = 1.35)
  for (sign in c(-1, 1)) {
    lines(smooth_x, smooth['mean', ] + sign * smooth['sd', ],
      col = 'black', lty = 2, lwd = 1.05)
  }
  dev.off()

  by_competition <- read_result('rolling_errors_by_competition.csv')
  long_panels <- read_result('rolling_errors_long_coverage.csv')
  stopifnot(requireNamespace('ggplot2', quietly = TRUE),
    requireNamespace('ggrepel', quietly = TRUE))
  points_data <- by_competition[order(by_competition$n_seasons, by_competition$mae_xgf), ]
  offsets <- ave(points_data$n_seasons, points_data$n_seasons,
    FUN = function(x) if (length(x) == 1L) 0 else seq(-0.16, 0.16, length.out = length(x)))
  points_data$x_position <- points_data$n_seasons + offsets
  points_data$highlight <- ifelse(points_data$competition_name %in% names(league_colors),
    points_data$competition_name, 'Other competitions')
  points_data$display_label <- points_data$competition_name
  labels <- c('Bundesliga' = 'Bundesliga (AUT)', 'Superliga' = 'Superliga (DEN)',
    'Super Liga' = 'Super Liga (SVK)', 'Primera División' = 'Primera Division (URU)')
  matched <- points_data$competition_name %in% names(labels)
  points_data$display_label[matched] <- labels[points_data$competition_name[matched]]
  style <- ggplot2::theme_classic(base_size = 11, base_family = 'Helvetica') +
    ggplot2::theme(text = ggplot2::element_text(colour = ink),
      axis.text = ggplot2::element_text(colour = ink),
      panel.border = ggplot2::element_rect(colour = ink, fill = NA, linewidth = 0.4),
      axis.line = ggplot2::element_blank(), plot.title = ggplot2::element_text(size = 12),
      plot.margin = ggplot2::margin(12, 8, 8, 6))
  left <- ggplot2::ggplot(points_data, ggplot2::aes(x_position, mae_xgf)) +
    ggplot2::geom_point(ggplot2::aes(colour = highlight, shape = highlight), size = 2.1) +
    ggrepel::geom_text_repel(ggplot2::aes(label = display_label),
      colour = ink, size = 3.5, box.padding = 0.20, point.padding = 0.16,
      min.segment.length = 0, segment.colour = '#999999', segment.size = 0.25,
      max.overlaps = Inf, max.iter = 20000, max.time = Inf, seed = 20260910,
      force = 2, force_pull = 0.12) +
    ggplot2::scale_colour_manual(values = c(league_colors, 'Other competitions' = 'black')) +
    ggplot2::scale_shape_manual(values = c('Ekstraklasa' = 16, 'La Liga 2' = 17,
      'Other competitions' = 16)) +
    ggplot2::scale_x_continuous(breaks = 2:8, limits = c(0.5, 9.2),
      expand = ggplot2::expansion(mult = 0)) +
    ggplot2::scale_y_continuous(limits = c(0.47, 0.72), breaks = seq(0.50, 0.70, 0.05),
      expand = ggplot2::expansion(mult = 0)) +
    ggplot2::labs(x = 'Number of observed seasons', y = expression('Test MAE for ' * xG^F),
      title = '(A)  Competition-level error') + style +
    ggplot2::theme(legend.position = 'none')
  right <- ggplot2::ggplot(long_panels, ggplot2::aes(season_end, mae_xgf,
    colour = competition_name, linetype = competition_name, shape = competition_name)) +
    ggplot2::geom_line(linewidth = 0.55) + ggplot2::geom_point(size = 2.1) +
    ggplot2::scale_colour_manual(values = league_colors) +
    ggplot2::scale_linetype_manual(values = c('Ekstraklasa' = 1, 'La Liga 2' = 2)) +
    ggplot2::scale_shape_manual(values = c('Ekstraklasa' = 16, 'La Liga 2' = 17)) +
    ggplot2::scale_x_continuous(breaks = 2019:2026) +
    ggplot2::scale_y_continuous(limits = c(0.47, 0.72), breaks = seq(0.50, 0.70, 0.05),
      expand = ggplot2::expansion(mult = 0)) +
    ggplot2::labs(x = 'Season ending year', y = expression('Test MAE for ' * xG^F),
      title = '(B)  Seasonal errors in two leagues') + style +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, hjust = 1, vjust = 0.5),
      legend.position = 'inside', legend.position.inside = c(0.97, 0.97),
      legend.justification = c(1, 1), legend.title = ggplot2::element_blank(),
      legend.background = ggplot2::element_blank(), legend.key.height = grid::unit(0.45, 'cm'))
  pdf(out('rolling_error_stability.pdf'), width = 10.6, height = 5.2,
    family = 'Helvetica', useDingbats = FALSE, title = 'Rolling forecast error by coverage and season')
  grid::grid.newpage()
  left_grob <- ggplot2::ggplotGrob(left)
  right_grob <- ggplot2::ggplotGrob(right)
  heights <- grid::unit.pmax(left_grob$heights, right_grob$heights)
  left_grob$heights <- heights
  right_grob$heights <- heights
  grid::pushViewport(grid::viewport(x = 0.29, width = 0.58))
  grid::grid.draw(left_grob)
  grid::popViewport()
  grid::pushViewport(grid::viewport(x = 0.79, width = 0.42))
  grid::grid.draw(right_grob)
  grid::popViewport()
  dev.off()

  table_rows <- read_result('rolling_errors_by_coverage.csv')
  table_rows <- table_rows[order(table_rows$length_group), ]
  table_body <- sprintf('%s & %d & %d & %s & %.4f & %.4f & $%.4f$ \\\\',
    gsub('-', '--', table_rows$length_group, fixed = TRUE),
    table_rows$n_competitions, table_rows$n_panels,
    format(table_rows$n, big.mark = ',', trim = TRUE),
    table_rows$mae_xgf, table_rows$mae_xgd, table_rows$bias_xgf)
  writeLines(c(
    '\\begin{table}[!htbp]', '\\color{reviewerThree}', '\\arrayrulecolor{black}', '\\centering',
    '\\captionsetup{labelfont={bf,color=reviewerThree},textfont={color=reviewerThree}}',
    paste0('\\caption{Observation-weighted rolling-forecast error by competition coverage ',
      '(9,686 team--match records; 4,843 matches).}'),
    '\\label{tab:coverage_errors}', '\\setlength{\\tabcolsep}{4pt}',
    '\\begin{tabular}{lrrrrrr}', '\\toprule',
    'Coverage & Competitions & Panels & $N$ & MAE $\\mathrm{xG}^{F}$ & MAE $\\mathrm{xG}^{D}$ & Bias $\\mathrm{xG}^{F}$ \\\\',
    '\\midrule', table_body, '\\bottomrule', '\\end{tabular}', '\\end{table}'),
    out('rolling_coverage.tex'))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  stopifnot(length(args) == 1L)
  render_diagnostics_stability(args[1])
}
