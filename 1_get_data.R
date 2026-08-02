# Libs ####
library(dplyr)
library(tidyr)
library(purrr)
library(StatsBombR)
library(readr)

# Params ####
username = ''
password = ''

# Games ####
competitions(username, password) %>% 
  filter(str_detect(competition_name, 'Play-offs', negate = TRUE),
         str_detect(competition_name, 'Superpuchar', negate = TRUE),
         str_detect(competition_name, 'Lech Poznan', negate = TRUE),
         str_detect(country_name, 'Europe', negate = TRUE)) %>% 
  select(country_name,
         competition_name, 
         season_name,
         season_id,
         competition_id) %>% 
  arrange(country_name, season_name) -> comp_ids

# Functions ####
get_events_one_comp <- function(country_name, competition_name, season_name,
                                season_id, competition_id, username, password) {
  
  add_missing_cols <- function(df, cols) {
    missing_cols <- setdiff(cols, names(df))
    
    for (col in missing_cols) {
      df[[col]] <- NA
    }
    
    df
  }
  
  message('Pobieram: ', country_name, ' | ', competition_name, ' | ', season_name)
  
  df_matches <- get.matches(
    username,
    password,
    season_id = season_id,
    competition_id = competition_id
  )
  
  if (nrow(df_matches) == 0) {
    return(tibble())
  }
  
  match_cols <- c(
    'match_id',
    'match_date',
    'match_week',
    'home_score',
    'away_score',
    'home_team.home_team_id',
    'home_team.home_team_name',
    'away_team.away_team_id',
    'away_team.away_team_name'
  )
  
  event_cols <- c(
    'match_id',
    'team.id',
    'team.name',
    'player.id',
    'player.name',
    'type.name',
    'position.name',
    'shot.open_goal',
    'shot.one_on_one',
    'shot.type.name',
    'under_pressure',
    'shot.outcome.name',
    'shot.body_part.name',
    'period',
    'minute',
    'second',
    'DistToGoal',
    'AngleToGoal',
    'AngleDeviation',
    'location.x',
    'location.y',
    'density.incone',
    'DefendersInCone',
    'shot.statsbomb_xg'
  )
  
  df_matches <- add_missing_cols(df_matches, match_cols)
  
  df_match_meta <- df_matches %>%
    select(all_of(match_cols)) %>%
    mutate(
      match_date = as.Date(match_date),
      home_score = suppressWarnings(as.numeric(home_score)),
      away_score = suppressWarnings(as.numeric(away_score)),
      home_team_id = as.character(home_team.home_team_id),
      away_team_id = as.character(away_team.away_team_id)
    )
  
  df_events <- allevents(
    username,
    password,
    matches = df_matches$match_id
  ) %>%
    allclean()
  
  df_events <- add_missing_cols(df_events, event_cols)
  
  df_events %>%
    filter(type.name == 'Shot') %>%
    select(all_of(event_cols)) %>%
    left_join(df_match_meta, by = 'match_id') %>%
    mutate(
      team_id = as.character(team.id),
      team_name = team.name,
      
      home_away = case_when(
        team_id == home_team_id ~ 'home',
        team_id == away_team_id ~ 'away',
        team.name == home_team.home_team_name ~ 'home',
        team.name == away_team.away_team_name ~ 'away',
        TRUE ~ NA_character_
      ),
      
      opponent_id = case_when(
        home_away == 'home' ~ away_team_id,
        home_away == 'away' ~ home_team_id,
        TRUE ~ NA_character_
      ),
      
      opponent_name = case_when(
        home_away == 'home' ~ away_team.away_team_name,
        home_away == 'away' ~ home_team.home_team_name,
        TRUE ~ NA_character_
      ),
      
      goals_for = case_when(
        home_away == 'home' ~ home_score,
        home_away == 'away' ~ away_score,
        TRUE ~ NA_real_
      ),
      
      goals_against = case_when(
        home_away == 'home' ~ away_score,
        home_away == 'away' ~ home_score,
        TRUE ~ NA_real_
      ),
      
      result = case_when(
        goals_for > goals_against ~ 'win',
        goals_for == goals_against ~ 'draw',
        goals_for < goals_against ~ 'loss',
        TRUE ~ NA_character_
      ),
      
      player_id = as.character(player.id),
      player_name = player.name,
      position_name = position.name,
      
      event_type = type.name,
      is_open_goal = shot.open_goal,
      is_one_on_one = shot.one_on_one,
      shot_type = shot.type.name,
      shot_outcome = shot.outcome.name,
      body_part = shot.body_part.name,
      is_goal = as.integer(shot_outcome == 'Goal'),
      
      distance_to_goal = DistToGoal,
      angle_to_goal = AngleToGoal,
      angle_deviation = AngleDeviation,
      location_x = location.x,
      location_y = location.y,
      defender_density_in_cone = density.incone,
      defenders_in_cone = DefendersInCone,
      statsbomb_xg = shot.statsbomb_xg
    ) %>%
    transmute(
      country_name = country_name,
      competition_name = competition_name,
      season_name = season_name,
      season_id = season_id,
      competition_id = competition_id,
      
      match_id,
      match_date,
      match_week,
      
      team_id,
      team_name,
      home_away,
      opponent_id,
      opponent_name,
      goals_for,
      goals_against,
      result,
      
      player_id,
      player_name,
      position_name,
      
      event_type,
      is_open_goal,
      is_one_on_one,
      shot_type,
      under_pressure,
      shot_outcome,
      is_goal,
      body_part,
      
      period,
      minute,
      second,
      
      distance_to_goal,
      angle_to_goal,
      angle_deviation,
      location_x,
      location_y,
      defender_density_in_cone,
      defenders_in_cone,
      statsbomb_xg
    )
}


safe_get_events_one_comp <- safely(get_events_one_comp, 
                                   otherwise = tibble())

# safe_get_events_one_comp(comp_ids$country_name[1], comp_ids$competition_name[1],
#                          comp_ids$season_name[1], comp_ids$season_id[1],
#                          comp_ids$competition_id[1], username, password) -> A

# Data ####
results <- comp_ids %>%
  mutate(result = pmap(list(country_name,
                            competition_name,
                            season_name,
                            season_id,
                            competition_id),
                       function(country_name,
                                competition_name,
                                season_name,
                                season_id,
                                competition_id) {
                         safe_get_events_one_comp(country_name = country_name,
                                                  competition_name = competition_name,
                                                  season_name = season_name,
                                                  season_id = season_id,
                                                  competition_id = competition_id,
                                                  username = username,
                                                  password = password)
                       }))

df_events_all <- results %>%
  mutate(events = map(result, 'result')) %>%
  select(events) %>%
  unnest(events)

df_events_all %>% 
  mutate(competition_name = case_when(country_name == 'Japan' & competition_name %in% c('J1 100 Year Vision League', 'J1 League') ~ 'J1 League',
                                      country_name == 'Greece' & competition_name == 'Super League' ~ 'Greek Super League',
                                      country_name == 'Switzerland' & competition_name == 'Super League' ~ 'Swiss Super League',
                                      TRUE ~ competition_name),
         competition_id = if_else(country_name == 'Japan', 108, competition_id)) -> df_events_all

write_csv(df_events_all, 'df.csv')
