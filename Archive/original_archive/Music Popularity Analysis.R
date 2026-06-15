library(tidyverse)

similarity_matrix <- read_csv("results/similarity_matrix.csv")

similarity_matrix <- similarity_matrix |> 
  pivot_longer(-1, names_to = "lyrics", values_to = "similarity") |> 
  rename(audio = `...1`) |> 
  mutate(audio = str_remove_all(audio, " \\(Audio\\)"),
         lyrics = str_remove_all(lyrics, " \\(Lyrics\\)")) |> 
  filter(audio == lyrics)

# Popularity Data
popularity_data <- tibble(
  audio = c(
    "RTJ_01",
    "RTJ_02",
    "RTJ_03",
    "RTJ_04",
    "RTJ_05",
    "RTJ_06",
    "RTJ_07",
    "RTJ_08",
    "RTJ_09",
    "RTJ_10",
    "GD_01",
    "GD_02",
    "GD_03",
    "GD_04",
    "GD_05",
    "GD_06",
    "GD_07",
    "GD_08",
    "GD_09",
    "GD_10"
  ),
  name = c(
    "Run the Jewels",
    "Banana Clipper",
    "36 Inch Chain",
    "DDHF",
    "Sea Legs",
    "Job Well Done",
    "No Come Down",
    "Get It",
    "Twin Hype Back",
    "A Christmas F*cking Miracle",
    "Minglewood Blues",
    "They Love Each Other",
    "Cassidy",
    "Loser",
    "Jack Straw",
    "Tennessee Jed",
    "Passenger",
    "Peggy-O",
    "Me & My Uncle",
    "Friend of the Devil"
  ),
  streams = c(
    45072119,
    18241755,
    9501708,
    12334740,
    10119782,
    6831525,
    6402405,
    13083897,
    4573336,
    9814085,
    7488184, 
    8329976,
    7077816,
    6938267, # Loser
    11381611, # Jack Straw
    14685522, # Tennessee Jed
    1361394, # Passenger
    2288557, # Peggy-O
    2588595,  # Me & My Uncle
    136975594   # Friend of the Devil
  ),
  artist = c(
    rep("Run the Jewels", 10),
    rep("Grateful Dead", 10)
  )
)

# Join
analysis_data <- similarity_matrix |> 
  left_join(popularity_data, by = "audio")

# Plot
ggplot(analysis_data, 
       aes(x = similarity, 
           y = streams)) +
  geom_smooth(method = "lm", 
              se = TRUE,
              data = subset(
                analysis_data,
                streams < 40000000
              )) +
  geom_point(aes(color = artist)) +
  geom_text(aes(label = name, color = artist), vjust = -1, size = 3) +
  labs(title = "Semantic Similarity of Lyrics and Audio vs. Song Popularity",
       x = "Similarity Score",
       y = "Spotify Streams",
       ) +
  theme_minimal()
  
# Linear Regression without Outliers
lm_model <- lm(streams ~ similarity, data = subset(analysis_data, streams < 40000000))

summary(lm_model)  

28235929/100
