library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(slider)
library(readr)
library(purrr)
library(patchwork)
library(scales)
library(tidytext)

response_scale <- 'log' # main analysis: 'log'; sensitivity analysis: 'raw'

results_dir <- 'results'
figure_dir <- 'figs'

dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

df <- read_csv('df.csv', show_col_types = FALSE)
df_model <- read_csv('df_model.csv', show_col_types = FALSE)

# Table 2 ####
df %>%
  filter(!is.na(statsbomb_xg)) %>%
  mutate(
    statsbomb_xg = as.numeric(statsbomb_xg),
    is_goal_num = ifelse(is_goal %in% c(1, TRUE, 'TRUE', '1'), 1, 0)
  ) %>%
  bind_rows(mutate(., country_name = 'Total', competition_name = 'All competitions')) %>%
  group_by(country_name, competition_name) %>%
  summarise(
    seasons = n_distinct(season_id),
    seasons_included = paste(sort(unique(season_name)), collapse = ', '),
    matches = n_distinct(match_id),
    teams = n_distinct(team_id),
    team_match_observations = n_distinct(paste(match_id, team_id, sep = '_')),
    shots = n(),
    goals = sum(is_goal_num, na.rm = TRUE),
    total_xg = sum(statsbomb_xg, na.rm = TRUE),
    mean_shots_per_match = shots / matches,
    mean_xg_per_match = total_xg / matches,
    mean_xg_per_shot = total_xg / shots,
    .groups = 'drop'
  ) %>%
  arrange(country_name == 'Total', country_name, competition_name)

# Table 3 ####
df_model %>%
  arrange(match_date, match_id, desc(home_away == 'home'), team_name) %>%
  transmute(
    match_date,
    competition = competition_name,
    season = season_name,
    team = team_name,
    opponent = opponent_name,
    HA = home_away,
    xG_for = round(xG_for, 2),
    xG_against = round(xG_against, 2),
    xG_diff = round(xG_diff, 2)
    # shots_for,
    # shots_against
  ) %>%
  slice_head(n = 6)

# Table 7 ####

df_plot <- df %>%
  group_by(match_id) %>%
  filter(!any(period > 2, na.rm = TRUE)) %>%
  ungroup()

team_match_xg <- df_plot %>%
  filter(!is.na(statsbomb_xg)) %>%
  group_by(match_id, team_id) %>%
  summarise(
    country_name = first(country_name),
    competition_name = first(competition_name),
    season_name = first(season_name),
    season_id = first(season_id),
    competition_id = first(competition_id),
    match_date = first(match_date),
    match_week = first(match_week),
    team_name = first(team_name),
    home_away = first(home_away),
    opponent_id = first(opponent_id),
    opponent_name = first(opponent_name),
    goals_for = first(goals_for),
    goals_against = first(goals_against),
    xG_for = sum(statsbomb_xg, na.rm = TRUE),
    shots_for = n(),
    .groups = 'drop'
  )

opponent_xg <- team_match_xg %>%
  select(
    match_id,
    opponent_id = team_id,
    xG_against = xG_for,
    shots_against = shots_for
  )

team_match_xg <- team_match_xg %>%
  left_join(
    opponent_xg,
    by = c('match_id', 'opponent_id')
  ) %>%
  mutate(
    xG_diff = xG_for - xG_against
  )

team_match_xg %>%
  filter(!is.na(xG_against)) %>%
  group_by(
    country_name,
    competition_name
  ) %>%
  summarise(
    Mean_xG_F = mean(xG_for, na.rm = TRUE),
    SD_xG_F = sd(xG_for, na.rm = TRUE),
    Mean_xG_D = mean(xG_diff, na.rm = TRUE),
    SD_xG_D = sd(xG_diff, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(country_name, competition_name) %>%
  transmute(
    Country = country_name,
    Competition = competition_name,
    `Mean xG^F` = round(Mean_xG_F, 3),
    `SD xG^F` = round(SD_xG_F, 3),
    `SD xG^D` = round(SD_xG_D, 3))


# Figure 5 ####

full_mean_xg_for <- team_match_xg %>%
  summarise(mean_xG_for = mean(xG_for, na.rm = TRUE)) %>%
  pull(mean_xG_for)

selected_df <- tibble::tribble(
  ~team_name,              ~competition_name, ~country_name, ~team_display, ~league_display,
  'Lech Poznan',           'Ekstraklasa',     'Poland',      'Lech Poznań', 'Ekstraklasa',
  'Barcelona',             'La Liga',         'Spain',       'FC Barcelona', 'La Liga',
  'Bayern Munich',         '1. Bundesliga',   'Germany',     'Bayern Munich', 'Bundesliga',
  'Paris Saint-Germain',   'Ligue 1',         'France',      'Paris Saint-Germain', 'Ligue 1'
)

example_series <- team_match_xg %>%
  filter(season_name == '2025/2026') %>%
  inner_join(
    selected_df,
    by = c('team_name', 'competition_name', 'country_name')
  ) %>%
  arrange(team_id, match_date, match_id) %>%
  group_by(team_id) %>%
  mutate(
    match_number = row_number(),
    rolling5_xG_for = if (response_scale == 'log') {
      expm1(
        slide_dbl(
          lag(log1p(xG_for)),
          mean,
          .before = 4,
          .complete = TRUE,
          na.rm = TRUE
        )
      )
    } else {
      slide_dbl(
        lag(xG_for),
        mean,
        .before = 4,
        .complete = TRUE,
        na.rm = TRUE
      )
    },
    team_label = paste0(team_display, ' (', league_display, ')'),
    HA_label = recode(
      home_away,
      'home' = 'Home',
      'away' = 'Away'
    )
  ) %>%
  ungroup()

p <- ggplot(example_series, aes(x = match_number, y = xG_for)) +
  geom_point(
    aes(color = HA_label),
    alpha = 0.75,
    size = 2.2
  ) +
  geom_line(
    aes(y = rolling5_xG_for),
    linewidth = 0.8,
    color = 'black'
  ) +
  geom_hline(
    yintercept = full_mean_xg_for,
    linewidth = 0.7,
    linetype = 'dashed',
    alpha = 0.85
  ) +
  facet_wrap(~ team_label, ncol = 2, scales = 'free') +
  scale_color_manual(
    values = c(
      'Home' = '#1f77b4',
      'Away' = '#ff7f0e'
    )
  ) +
  labs(
    x = 'Match number',
    y = expression(xG^F),
    color = 'Venue',
    title = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = 'bold'),
    legend.position = 'bottom'
  )

ggsave(filename = file.path(figure_dir, 'example_xg_series.pdf'),
       plot = p,
       width = 7.2,
       height = 4.8,
       device = cairo_pdf)

# Figure 6 ####

prediction_files <- tibble::tribble(
  ~model,                 ~file_prefix,
  'Rolling mean',         'rolling',
  'ARIMA',                'arima',
  'Temporal ConvNet',     'tcn',
  'XGBoost',              'xgb',
  'Mixed-effects model',  'lmm'
) %>%
  mutate(
    file = file.path(
      results_dir,
      paste0(
        file_prefix,
        '_',
        response_scale,
        '_test_predictions_long.csv'
      )
    )
  )

plot_df <- map2_dfr(
  prediction_files$model,
  prediction_files$file,
  ~ read_csv(.y, show_col_types = FALSE) %>%
    filter(target == 'xG_for') %>%
    transmute(
      model = .x,
      match_id,
      team_id,
      actual,
      predicted
    )
)

model_order <- c(
  'Rolling mean',
  'ARIMA',
  'Temporal ConvNet',
  'XGBoost',
  'Mixed-effects model'
)

plot_df <- plot_df %>%
  mutate(
    model = factor(model, levels = model_order)
  )

lims <- range(
  c(plot_df$actual, plot_df$predicted),
  na.rm = TRUE
)

make_model_plot <- function(model_name) {
  plot_df %>%
    filter(model == model_name) %>%
    ggplot(aes(x = actual, y = predicted)) +
    geom_bin2d(bins = 100) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = 'dashed',
      linewidth = 0.5,
      color = 'red'
    ) +
    # coord_equal(xlim = lims, ylim = lims) +
    labs(
      x = expression(Observed~xG^F),
      y = expression(Predicted~xG^F),
      fill = 'Count',
      title = as.character(model_name)
    ) +
    guides(fill = 'none') +
    theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(face = 'bold', hjust = 0.5, size = 10),
      legend.position = 'bottom'
    )
}

p1 <- make_model_plot('Rolling mean')
p2 <- make_model_plot('ARIMA')
p3 <- make_model_plot('Temporal ConvNet')
p4 <- make_model_plot('XGBoost')
p5 <- make_model_plot('Mixed-effects model')

top_row <- p1 | p2
middle_row <- p3 | p4

bottom_row <- plot_spacer() | p5 | plot_spacer()
bottom_row <- bottom_row + plot_layout(widths = c(0.5, 1, 0.5))

p <- (top_row / middle_row / bottom_row) +
  plot_layout(
    guides = 'collect',
    heights = c(1, 1, 1)
  ) &
  theme(
    legend.position = 'bottom'
  )

ggsave(
  filename = file.path(figure_dir, 'observed_vs_predicted_xgfor_test.pdf'),
  plot = p,
  width = 7.2,
  height = 6.2,
  device = cairo_pdf
)

# Figure: XGBoost grouped permutation importance ####

importance <- read_csv(
  file.path(results_dir, 'xgb_log_importance.csv'),
  show_col_types = FALSE
)

plot_df <- importance %>%
  mutate(
    relative_importance =
      100 * delta_mae_mean / max(delta_mae_mean, na.rm = TRUE),
    
    variable_type = case_when(
      str_detect(feature_group, 'identity$') ~
        'Identity variables',
      feature_group %in% c(
        'Venue (home/away)',
        'Match timing and experience'
      ) ~
        'Match context',
      TRUE ~
        'Performance history'
    ),
    
    feature_label = reorder(
      feature_group,
      relative_importance
    )
  )

p <- ggplot(
  plot_df,
  aes(
    x = relative_importance,
    y = feature_label,
    fill = variable_type
  )
) +
  geom_col(
    width = 0.80
  ) +
  geom_text(
    aes(
      label = sprintf(
        '%.1f%%',
        relative_importance
      )
    ),
    hjust = -0.15,
    size = 3.4
  ) +
  scale_fill_manual(
    name = NULL,
    values = c(
      'Performance history' = '#0072B2',
      'Identity variables' = '#D55E00',
      'Match context' = '#009E73'
    )
  ) +
  guides(
    fill = guide_legend(
      ncol = 1,
      direction = 'vertical'
    )
  ) +
  scale_x_continuous(
    limits = c(0, 112),
    breaks = seq(0, 100, 20),
    labels = function(x) {
      paste0(x, '%')
    },
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  labs(
    x = 'Relative permutation importance',
    y = NULL
  ) +
  theme_classic() +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    
    axis.text.x = element_text(
      size = 12
    ),
    
    axis.text.y = element_text(
      size = 12,
      lineheight = 0.95,
      margin = margin(r = 7)
    ),
    
    axis.title.x = element_text(
      size = 12
    ),
    
    legend.position = 'inside',
    legend.position.inside = c(0.98, 0.03),
    legend.justification = c('right', 'bottom'),
    
    legend.background = element_rect(
      fill = 'white',
      colour = 'grey70',
      linewidth = 0.5
    ),
    
    legend.text = element_text(
      size = 11
    ),
    
    legend.key.height = grid::unit(
      0.35,
      'cm'
    ),
    
    legend.key.width = grid::unit(
      0.45,
      'cm'
    ),
    
    legend.key.spacing.y = grid::unit(
      0.1,
      'cm'
    )
  )

ggsave(
  filename = file.path(
    figure_dir,
    'xgb_log_grouped_feature_importance.pdf'
  ),
  plot = p,
  width = 7.2,
  height = 6.2,
  device = cairo_pdf
)

# Figure: ARIMA models ####

arima_counts <- read_csv(
  file.path(results_dir, 'arima_log_model_counts.csv'),
  show_col_types = FALSE
) %>%
  filter(
    response_scale == 'log',
    target == 'xG_for',
    split == 'test',
    !is.na(arima_model)
  ) %>%
  group_by(arima_model) %>%
  summarise(
    n = sum(n),
    .groups = 'drop'
  ) %>%
  arrange(desc(n)) %>% 
  filter(arima_model != 'Unclassified')

total_n <- sum(arima_counts$n)

top9 <- arima_counts %>%
  slice_head(n = 9)

other_n <- arima_counts %>%
  filter(!arima_model %in% top9$arima_model) %>%
  summarise(n = sum(n)) %>%
  pull(n)

plot_df <- bind_rows(
  top9,
  tibble(arima_model = 'Other', n = other_n)
)

orders <- str_match(
  plot_df$arima_model,
  '^ARIMA\\((\\d+),(\\d+),(\\d+)\\)$'
)

plot_df <- plot_df %>%
  mutate(
    p = as.integer(orders[, 2]),
    d = as.integer(orders[, 3]),
    q = as.integer(orders[, 4]),
    model_family = case_when(
      arima_model == 'Other' ~ 'Other',
      d > 0 ~ 'ARIMA',
      p == 0 & q == 0 ~ 'No AR/MA terms',
      p > 0 & q == 0 ~ 'AR',
      p == 0 & q > 0 ~ 'MA',
      p > 0 & q > 0 ~ 'ARMA',
      TRUE ~ 'Other'
    ),
    model_family = factor(
      model_family,
      levels = c(
        'No AR/MA terms',
        'AR',
        'MA',
        'ARMA',
        'ARIMA',
        'Other'
      )
    ),
    percentage = 100 * n / total_n,
    count_label = format(
      n,
      big.mark = ',',
      scientific = FALSE,
      trim = TRUE
    ),
    value_label = paste0(
      count_label,
      ' (',
      sprintf('%.1f', percentage),
      '%)'
    ),
    log_n = log10(n),
    arima_model = factor(
      arima_model,
      levels = rev(c(top9$arima_model, 'Other'))
    )
  )

p <- ggplot(
  plot_df,
  aes(
    x = log_n,
    y = arima_model,
    fill = model_family
  )
) +
  geom_col(width = 0.80) +
  geom_text(
    aes(
      x = log_n + 0.06,
      label = value_label
    ),
    hjust = 0,
    size = 3.4
  ) +
  scale_fill_manual(
    name = NULL,
    values = c(
      'No AR/MA terms' = '#666666',
      'AR' = '#0072B2',
      'MA' = '#D55E00',
      'ARMA' = '#CC79A7',
      'ARIMA' = '#009E73',
      'Other' = '#BDBDBD'
    )
  ) +
  guides(
    fill = guide_legend(
      ncol = 1,
      direction = 'vertical'
    )
  ) +
  scale_x_continuous(
    limits = c(0, 5.2),
    breaks = log10(c(1, 10, 30, 100, 300, 1000, 3000, 10000)),
    labels = c('1', '10', '30', '100', '300', '1,000', '3,000', '10,000'),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_discrete(
    labels = function(x) str_remove(x, '^ARIMA')
  ) +
  labs(
    x = 'Number of selections (logarithmic scale)',
    y = 'ARIMA order (p, d, q)'
  ) + 
  theme_classic() +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(
      size = 12,
      margin = margin(r = 7)
    ),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(
      size = 12,
      margin = margin(r = 12)
    ),
    legend.position = 'inside',
    legend.position.inside = c(0.98, 0.03),
    legend.justification = c('right', 'bottom'),
    legend.background = element_rect(
      fill = 'white',
      colour = 'grey70',
      linewidth = 0.5
    ),
    legend.text = element_text(size = 11),
    legend.key.height = grid::unit(0.35, 'cm'),
    legend.key.width = grid::unit(0.45, 'cm'),
    legend.key.spacing.y = grid::unit(0.1, 'cm'),
    plot.margin = margin(5.5, 15, 5.5, 5.5)
  )

ggsave(filename = file.path(figure_dir, 'arima_log_model_distribution.pdf'),
       plot = p,
       width = 7.2,
       height = 6.2,
       device = cairo_pdf)

# Figure: Forest plot for CI ####

plot_df <- read_csv(
  file.path(results_dir, 'bootstrap_log_mae_differences_vs_rolling.csv'),
  show_col_types = FALSE
) %>%
  filter(
    response_scale == 'log',
    target %in% c('xG_for', 'xG_diff')
  ) %>%
  mutate(
    model = factor(
      model,
      levels = rev(c(
        'Mixed-effects model',
        'XGBoost',
        'Temporal ConvNet',
        'ARIMA'
      ))
    ),
    target_label = factor(
      target,
      levels = c('xG_for', 'xG_diff'),
      labels = c('xG^F', 'xG^D')
    ),
    significant = (
      mae_difference_ci_low > 0 |
        mae_difference_ci_high < 0
    )
  )

x_range <- range(
  c(
    plot_df$mae_difference_ci_low,
    plot_df$mae_difference_ci_high,
    0
  ),
  na.rm = TRUE
)

x_padding <- 0.08 * diff(x_range)
x_limits <- x_range + c(-x_padding, x_padding)

p <- ggplot(
  plot_df,
  aes(
    x = mae_difference,
    y = model
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = 'dashed',
    linewidth = 0.6
  ) +
  geom_segment(
    aes(
      x = mae_difference_ci_low,
      xend = mae_difference_ci_high,
      yend = model
    ),
    linewidth = 0.9
  ) +
  geom_point(
    aes(shape = significant),
    size = 3.2,
    stroke = 0.9
  ) +
  facet_wrap(
    vars(target_label),
    ncol = 1,
    scales = 'free_y',
    labeller = label_parsed
  ) +
  scale_shape_manual(
    name = NULL,
    values = c(
      'TRUE' = 16,
      'FALSE' = 1
    ),
labels = c(
  'TRUE' = 'Simultaneous 95% CI excludes zero',
  'FALSE' = 'Simultaneous 95% CI includes zero'
)
  ) +
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 6),
    labels = scales::label_number(
      accuracy = 0.01
    ),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_cartesian(
    xlim = x_limits,
    clip = 'off'
  ) +
  labs(
    x = expression(
      Delta * 'MAE relative to the rolling mean'
    ),
    y = NULL
  ) +
  theme_classic() +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_text(size = 11),
    axis.text.y = element_text(
      size = 11,
      margin = margin(r = 8)
    ),
    axis.title.x = element_text(
      size = 12,
      margin = margin(t = 8)
    ),
    # strip.background = element_blank(),
    strip.text = element_text(
      size = 13,
      face = 'bold',
      margin = margin(t = 6, b = 6)
    ),
    panel.spacing.y = grid::unit(1.1, 'cm'),
    legend.position = 'bottom',
    legend.justification = 'center',
    legend.location = 'plot',
    legend.box.just = 'center',
    legend.text = element_text(size = 10),
    plot.margin = margin(6, 12, 6, 6)
  )

ggsave(filename = file.path(figure_dir, 'bootstrap_mae_difference_forest.pdf'),
  plot = p,
  width = 7.2,
  height = 5.6,
  device = cairo_pdf)

# Figure: TCN training

training_history <- read_csv(
  file.path(results_dir, 'tcn_log_selected_trial_training_history.csv'),
  show_col_types = FALSE
)

best_epoch <- training_history %>%
  filter(is.finite(val_loss)) %>%
  slice_min(
    order_by = val_loss,
    n = 1,
    with_ties = FALSE
  )

plot_df <- training_history %>%
  select(epoch, train_loss, val_loss) %>%
  pivot_longer(
    cols = c(train_loss, val_loss),
    names_to = 'loss_type',
    values_to = 'loss'
  ) %>%
  mutate(
    loss_type = factor(
      loss_type,
      levels = c('train_loss', 'val_loss'),
      labels = c('Training loss', 'Validation loss')
    )
  )

p <- ggplot(
  plot_df,
  aes(
    x = epoch,
    y = loss,
    color = loss_type,
    linetype = loss_type
  )
) +
  geom_line(linewidth = 0.9) +
  geom_point(
    data = best_epoch,
    aes(
      x = epoch,
      y = val_loss
    ),
    inherit.aes = FALSE,
    shape = 20,
    size = 5,
    stroke = 0.9,
    fill = 'white',
    color = '#D55E00'
  ) +
  geom_vline(
    xintercept = best_epoch$epoch,
    linewidth = 0.6,
    linetype = 'dashed',
    color = '#666666'
  ) +
  annotate(
    'text',
    x = best_epoch$epoch + 0.8,
    y = Inf,
    label = paste0(
      'Selected epoch: ',
      best_epoch$epoch
    ),
    hjust = 0,
    vjust = 1.4,
    size = 4
  ) +
  scale_color_manual(
    name = NULL,
    values = c(
      'Training loss' = '#0072B2',
      'Validation loss' = '#D55E00'
    )
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c(
      'Training loss' = 'solid',
      'Validation loss' = 'solid'
    )
  ) +
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 8)
  ) +
  labs(
    x = 'Epoch',
    y = 'Loss'
  ) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    
    legend.position = 'inside',
    legend.position.inside = c(0.5, 0.98),
    legend.justification = c('center', 'top'),
    legend.direction = 'horizontal',
    legend.text = element_text(size = 11),
    
    legend.background = element_rect(
      fill = scales::alpha('white', 0.8),
      colour = NA
    ),
    plot.margin = margin(6, 12, 6, 6)
  )

ggsave(filename = file.path(figure_dir, 'tcn_log_selected_trial_learning_curve.pdf'),
  plot = p,
  width = 7.2,
  height = 4.8,
  device = cairo_pdf)

# Figure: LOCO ####
competition_order <- c(
  'Bundesliga (AUT)',
  'Challenger Pro League (BEL)',
  'Jupiler Pro League (BEL)',
  '1. HNL (HRV)',
  'Czech Liga (CZE)',
  '1. Division (DNK)',
  'Superliga (DNK)',
  'Ligue 1 (FRA)',
  'Ligue 2 (FRA)',
  '1. Bundesliga (DEU)',
  '2. Bundesliga (DEU)',
  'Greek Super League (GRC)',
  'NB I (HUN)',
  'Serie B (ITA)',
  'J1 League (JPN)',
  'Eredivisie (NLD)',
  'Eliteserien (NOR)',
  'Ekstraklasa (POL)',
  'Primeira Liga (PRT)',
  'Segunda Liga (PRT)',
  'Premiership (SCO)',
  'Super Liga (SVK)',
  'La Liga (ESP)',
  'La Liga 2 (ESP)',
  'Allsvenskan (SWE)',
  'Superettan (SWE)',
  'Swiss Super League (CHE)',
  'Süper Lig (TUR)',
  'Major League Soccer (USA)',
  'Primera División (URY)'
)

second_tier <- c(
  'Challenger Pro League',
  '1. Division',
  'Ligue 2',
  '2. Bundesliga',
  'Serie B',
  'Segunda Liga',
  'La Liga 2',
  'Superettan'
)

non_europe <- c(
  'J1 League',
  'Major League Soccer',
  'Primera División'
)

competition_labels <- tibble(
  held_out_competition_name = c(
    'Bundesliga',
    'Challenger Pro League',
    'Jupiler Pro League',
    '1. HNL',
    'Czech Liga',
    '1. Division',
    'Superliga',
    'Ligue 1',
    'Ligue 2',
    '1. Bundesliga',
    '2. Bundesliga',
    'Greek Super League',
    'NB I',
    'Serie B',
    'J1 League',
    'Eredivisie',
    'Eliteserien',
    'Ekstraklasa',
    'Primeira Liga',
    'Segunda Liga',
    'Premiership',
    'Super Liga',
    'La Liga',
    'La Liga 2',
    'Allsvenskan',
    'Superettan',
    'Swiss Super League',
    'Süper Lig',
    'Major League Soccer',
    'Primera División'
  ),
  competition_label_plain = competition_order
) %>%
  mutate(
    competition_label = case_when(
      held_out_competition_name %in% second_tier ~ paste0(
        '<span style="color:#0057B8"><i>',
        competition_label_plain,
        '</i></span>'
      ),
      TRUE ~ paste0(
        '<span>',
        competition_label_plain,
        '</span>'
      )
    )
  )

df <- read_csv(
  file.path(results_dir, 'xgb_loco_log_vs_rolling_by_competition.csv')
) %>%
  filter(target %in% c('xG_for', 'xG_diff')) %>%
  left_join(
    competition_labels,
    by = 'held_out_competition_name'
  ) %>%
  mutate(
    target = factor(
      target,
      levels = c('xG_for', 'xG_diff'),
      labels = c('xG^F', 'xG^D')
    ),
    competition_label = factor(
      competition_label,
      levels = rev(competition_labels$competition_label)
    )
  )

non_europe_rows <- competition_labels %>%
  filter(held_out_competition_name %in% non_europe) %>%
  mutate(
    y = match(
      competition_label,
      rev(competition_labels$competition_label)
    ),
    ymin = y - 0.5,
    ymax = y + 0.5
  )

df %>%
  ggplot(
    aes(
      x = delta_mae_xgb_minus_rolling,
      y = competition_label
    )
  ) +
  geom_rect(
    data = non_europe_rows,
    aes(ymin = ymin, ymax = ymax),
    xmin = -Inf,
    xmax = Inf,
    fill = '#FFF2CC',
    inherit.aes = FALSE
  ) +
  geom_vline(
    xintercept = 0,
    linetype = 'dashed'
  ) +
  geom_segment(
    aes(
      x = 0,
      xend = delta_mae_xgb_minus_rolling,
      yend = competition_label
    )
  ) +
  geom_point() +
  facet_wrap(
    ~target,
    nrow = 1,
    labeller = label_parsed
  ) +
  labs(
    x = expression(
      Delta * MAE~'(XGBoost - rolling mean)'
    ),
    y = NULL
  ) +
  theme_classic() +
  theme(
    axis.text.y = element_markdown()
  ) -> p

ggsave(filename = file.path(figure_dir, 'loco_delta_mae.pdf'),
       plot = p,
       width = 7.2,
       height = 6.8,
       device = cairo_pdf)

# Calendar leakage ####
df_model <- read_csv(
  'df_model.csv',
  show_col_types = FALSE
) %>%
  mutate(
    match_date = as.Date(match_date)
  )

fit_df <- df_model %>%
  filter(
    split %in% c('train', 'validation'),
    !is.na(match_date)
  )

test_df <- df_model %>%
  filter(
    split == 'test',
    !is.na(match_date)
  )

fit_dates <- sort(as.numeric(fit_df$match_date))
n_fit <- length(fit_dates)

panel_fit_end <- fit_df %>%
  group_by(
    competition_id,
    season_id
  ) %>%
  summarise(
    last_fit_date_same_panel = max(match_date),
    .groups = 'drop'
  )

panel_end_dates <- panel_fit_end$last_fit_date_same_panel

calendar_overlap <- test_df %>%
  select(
    competition_id,
    season_id,
    match_id,
    team_id,
    match_date
  ) %>%
  left_join(
    panel_fit_end,
    by = c(
      'competition_id',
      'season_id'
    )
  ) %>%
  mutate(
    # Liczba wszystkich obserwacji dopasowania późniejszych
    # niż dana obserwacja testowa
    n_fit_later = n_fit - findInterval(
      as.numeric(match_date),
      fit_dates
    ),
    
    pct_fit_later = 100 * n_fit_later / n_fit,
    
    any_fit_later = n_fit_later > 0,
    
    # Liczba par liga–sezon zawierających późniejsze dane
    n_panels_later = vapply(
      match_date,
      function(x) {
        sum(panel_end_dates > x)
      },
      integer(1)
    ),
    
    # Kontrola, czy problem występuje również
    # wewnątrz tej samej pary liga–sezon
    same_panel_future = (
      last_fit_date_same_panel > match_date
    )
  )

calendar_overlap_summary <- calendar_overlap %>%
  summarise(
    n_test = n(),
    
    n_test_with_later_fit = sum(any_fit_later),
    pct_test_with_later_fit = 100 * mean(any_fit_later),
    
    median_n_fit_later = median(n_fit_later),
    q1_n_fit_later = quantile(n_fit_later, 0.25),
    q3_n_fit_later = quantile(n_fit_later, 0.75),
    
    median_pct_fit_later = median(pct_fit_later),
    q1_pct_fit_later = quantile(pct_fit_later, 0.25),
    q3_pct_fit_later = quantile(pct_fit_later, 0.75),
    
    median_n_panels_later = median(n_panels_later),
    q1_n_panels_later = quantile(n_panels_later, 0.25),
    q3_n_panels_later = quantile(n_panels_later, 0.75),
    
    n_same_panel_future = sum(
      same_panel_future,
      na.rm = TRUE
    ),
    pct_same_panel_future = 100 * mean(
      same_panel_future,
      na.rm = TRUE
    )
  )

write_csv(
  calendar_overlap,
  file.path(
    'results',
    'calendar_overlap_test_observations.csv'
  )
write_csv(calendar_overlap_summary,
          file.path(results_dir, 'calendar_overlap_summary.csv'))
