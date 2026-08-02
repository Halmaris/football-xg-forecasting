library(dplyr)
library(tidyr)
library(slider)
library(stringr)
library(readr)

# Prepares team-match data for xG forecasting.
# Input:  df.csv in the current directory (shot-level data).
# Output: df_model.csv in the current directory, containing team-match outcomes,
#         lagged and rolling features, opponent features, and temporal data splits.
# Matches containing extra time or penalty shoot-outs are excluded entirely.

input_file <- 'df.csv'
output_file <- 'df_model.csv'
lag_orders <- c(1, 2, 3, 5)
rolling_windows <- c(3, 5, 8, 10)
min_history <- 3
train_prop <- 0.70
validation_prop <- 0.15

first_non_missing <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) NA else x[1]
}

safe_mean <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

add_history <- function(df) {
  vars <- c(
    'xG_for', 'xG_against', 'xG_diff',
    'shots_for', 'shots_against',
    'goals_for', 'goals_against', 'goal_diff'
  )

  df <- df %>%
    arrange(match_date, match_week, match_id) %>%
    mutate(
      team_match_order = row_number(),
      n_previous_matches = row_number() - 1
    )

  for (var in vars) {
    for (k in lag_orders) {
      df[[paste0('lag', k, '_', var)]] <- lag(df[[var]], k)
    }

    for (k in rolling_windows) {
      df[[paste0('rolling', k, '_', var)]] <- slide_dbl(
        lag(df[[var]]),
        safe_mean,
        .before = k - 1,
        .complete = FALSE
      )
    }
  }

  df
}

# Shot-level data ####

df <- read_csv(input_file, show_col_types = FALSE) %>%
  mutate(
    match_date = as.Date(match_date),
    across(
      c(season_id, competition_id, match_id, team_id, opponent_id),
      as.character
    ),
    match_week = as.integer(match_week),
    period = as.integer(period),
    goals_for = as.numeric(goals_for),
    goals_against = as.numeric(goals_against),
    is_goal = as.integer(is_goal),
    statsbomb_xg = as.numeric(statsbomb_xg)
  )

# Exclude complete matches containing extra time or penalty shoot-outs ####

excluded_matches <- df %>%
  group_by(competition_id, season_id, match_id) %>%
  summarise(
    has_extra_time = any(period %in% c(3L, 4L), na.rm = TRUE),
    has_shootout = any(period == 5L, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  filter(has_extra_time | has_shootout)

exclusion_summary <- excluded_matches %>%
  summarise(
    excluded_matches = n(),
    with_extra_time = sum(has_extra_time),
    with_shootout = sum(has_shootout)
  )

print(exclusion_summary)

df <- df %>%
  anti_join(
    excluded_matches,
    by = c('competition_id', 'season_id', 'match_id')
  )

stopifnot(!any(df$period > 2L, na.rm = TRUE))

# One row per team and match, including teams with no shots ####

base_rows <- df %>%
  select(
    country_name, competition_name, season_name,
    season_id, competition_id, match_id, match_date, match_week,
    team_id, team_name, home_away, opponent_id, opponent_name,
    goals_for, goals_against
  ) %>%
  distinct()

opponent_rows <- base_rows %>%
  rename(
    original_team_id = team_id,
    original_team_name = team_name,
    original_opponent_id = opponent_id,
    original_opponent_name = opponent_name,
    original_goals_for = goals_for,
    original_goals_against = goals_against
  ) %>%
  transmute(
    country_name,
    competition_name,
    season_name,
    season_id,
    competition_id,
    match_id,
    match_date,
    match_week,
    team_id = original_opponent_id,
    team_name = original_opponent_name,
    home_away = recode(home_away, home = 'away', away = 'home'),
    opponent_id = original_team_id,
    opponent_name = original_team_name,
    goals_for = original_goals_against,
    goals_against = original_goals_for
  )

team_match <- bind_rows(base_rows, opponent_rows) %>%
  filter(!is.na(match_id), !is.na(team_id), !is.na(opponent_id)) %>%
  group_by(match_id, team_id) %>%
  summarise(
    across(
      c(
        country_name, competition_name, season_name,
        season_id, competition_id, match_date, match_week,
        team_name, home_away, opponent_id, opponent_name,
        goals_for, goals_against
      ),
      first_non_missing
    ),
    .groups = 'drop'
  ) %>%
  mutate(
    result = case_when(
      goals_for > goals_against ~ 'win',
      goals_for == goals_against ~ 'draw',
      goals_for < goals_against ~ 'loss'
    ),
    home_away = factor(home_away, levels = c('home', 'away')),
    result = factor(result, levels = c('win', 'draw', 'loss'))
  )

shot_summary <- df %>%
  group_by(match_id, team_id) %>%
  summarise(
    shots_for = n(),
    goals_from_shots = sum(is_goal, na.rm = TRUE),
    xG_for = sum(statsbomb_xg, na.rm = TRUE),
    mean_xG_per_shot = safe_mean(statsbomb_xg),
    high_xG_shots = sum(statsbomb_xg >= 0.20, na.rm = TRUE),
    open_goal_shots = sum(is_open_goal == TRUE, na.rm = TRUE),
    one_on_one_shots = sum(is_one_on_one == TRUE, na.rm = TRUE),
    under_pressure_shots = sum(under_pressure == TRUE, na.rm = TRUE),
    headers = sum(body_part == 'Head', na.rm = TRUE),
    .groups = 'drop'
  )

team_match <- team_match %>%
  left_join(shot_summary, by = c('match_id', 'team_id')) %>%
  mutate(
    across(
      c(
        shots_for, goals_from_shots, xG_for, high_xG_shots,
        open_goal_shots, one_on_one_shots,
        under_pressure_shots, headers
      ),
      ~ replace_na(.x, 0)
    ),
    mean_xG_per_shot = replace_na(mean_xG_per_shot, 0)
  )

opponent_stats <- team_match %>%
  transmute(
    match_id,
    opponent_id = team_id,
    xG_against = xG_for,
    shots_against = shots_for,
    goals_from_shots_against = goals_from_shots,
    high_xG_shots_against = high_xG_shots
  )

team_match <- team_match %>%
  left_join(opponent_stats, by = c('match_id', 'opponent_id')) %>%
  mutate(
    across(
      c(
        xG_against, shots_against,
        goals_from_shots_against, high_xG_shots_against
      ),
      ~ replace_na(.x, 0)
    ),
    xG_diff = xG_for - xG_against,
    goal_diff = goals_for - goals_against,
    shot_diff = shots_for - shots_against,
    team_season_id = paste(team_id, season_id, sep = '_'),
    opponent_season_id = paste(opponent_id, season_id, sep = '_')
  )

# Historical and opponent features ####

df_model <- team_match %>%
  arrange(competition_id, season_id, team_id, match_date, match_week, match_id) %>%
  group_by(competition_id, season_id, team_id) %>%
  group_modify(~ add_history(.x)) %>%
  ungroup()

history_cols <- names(df_model) %>%
  str_subset('^(lag|rolling)[0-9]+_')

opponent_history <- df_model %>%
  select(match_id, opponent_id = team_id, all_of(history_cols)) %>%
  rename_with(~ paste0('opponent_', .x), all_of(history_cols))

df_model <- df_model %>%
  left_join(opponent_history, by = c('match_id', 'opponent_id'))

# Temporal train-validation-test split ####

match_split <- df_model %>%
  distinct(
    competition_id, season_id, match_id,
    match_date, match_week
  ) %>%
  arrange(competition_id, season_id, match_date, match_week, match_id) %>%
  group_by(competition_id, season_id) %>%
  mutate(
    match_order_in_season = row_number(),
    n_matches_in_season = n(),
    train_end = floor(train_prop * n_matches_in_season),
    validation_end = floor(
      (train_prop + validation_prop) * n_matches_in_season
    ),
    split = case_when(
      match_order_in_season <= train_end ~ 'train',
      match_order_in_season <= validation_end ~ 'validation',
      TRUE ~ 'test'
    )
  ) %>%
  ungroup() %>%
  select(
    competition_id, season_id, match_id,
    match_order_in_season, n_matches_in_season, split
  )

df_model <- df_model %>%
  left_join(
    match_split,
    by = c('competition_id', 'season_id', 'match_id')
  ) %>%
  mutate(split = factor(split, levels = c('train', 'validation', 'test'))) %>%
  filter(n_previous_matches >= min_history) %>%
  arrange(
    competition_id, season_id, match_date,
    match_week, match_id, team_id
  )

write_csv(df_model, output_file)

# Final data audit ####

final_audit <- df_model %>%
  count(split, name = 'n_team_matches')

print(final_audit)
