"""
config.py — Master configuration for Musical Congruence experiment.

SETUP BEFORE RUNNING:
  export GENIUS_API_TOKEN="your_token"       # https://genius.com/api-clients
  export SPOTIFY_CLIENT_ID="your_id"         # https://developer.spotify.com/dashboard
  export SPOTIFY_CLIENT_SECRET="your_secret"

Artist selection rationale
--------------------------
12 artists across 4 genre clusters, calibrated to test the narrative-forward
vs. production-forward moderation hypothesis (Section 2.7 of lit review).

  Narrative-forward  → hip-hop (RTJ, Kendrick, J. Cole), folk/country (Grateful Dead, Dylan, Tyler Childers)
  Production-forward → pop (Taylor Swift, The Weeknd, Dua Lipa), electronic (Daft Punk, Disclosure, Caribou)

~180 songs total provides sufficient power for mixed-effects modelling
with song nested in artist nested in genre-cluster.
"""

import os

# ── API Keys ──────────────────────────────────────────────────────────────────
GENIUS_API_TOKEN    = os.getenv("GENIUS_API_TOKEN",    "YOUR_GENIUS_TOKEN_HERE")
SPOTIFY_CLIENT_ID   = os.getenv("SPOTIFY_CLIENT_ID",   "YOUR_SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "YOUR_SPOTIFY_CLIENT_SECRET")

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
AUDIO_BASE_DIR = os.path.join(BASE_DIR, "audio")
LYRICS_DIR     = os.path.join(BASE_DIR, "lyrics")
SCHEMA_DIR     = os.path.join(BASE_DIR, "schemas")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "mulan": {
        "name":            "MuQ-MuLan",
        "hf_id":           "OpenMuQ/MuQ-MuLan-large",
        "audio_sr":        24_000,
        "joint_embedding": True,
        "description":     "Music-language joint embedding — music + playlist text annotations",
    },
    "clap": {
        "name":            "LAION-CLAP-Music",
        "hf_id":           "laion/larger_clap_music",
        "audio_sr":        48_000,
        "chunk_s":         10,       # CLAP processes 10-second windows
        "joint_embedding": True,
        "description":     "Contrastive Language-Audio Pretraining — music + Audioset + LAION-630K",
    },
    "mert_sbert": {
        "name":            "MERT+SBERT (Late Fusion)",
        "audio_hf_id":     "m-a-p/MERT-v1-95M",
        "text_hf_id":      "sentence-transformers/all-mpnet-base-v2",
        "audio_sr":        24_000,
        "joint_embedding": False,
        "description":     "Independent audio (MERT) + text (SBERT) encoders — late-fusion baseline",
    },
}

# ── yt-dlp defaults ───────────────────────────────────────────────────────────
YTDLP_CONFIG = {
    "format":      "bestaudio/best",
    "postprocessors": [{
        "key":              "FFmpegExtractAudio",
        "preferredcodec":   "mp3",
        "preferredquality": "192",
    }],
    "default_search":     "ytsearch1:",
    "max_sleep_interval": 5,
    "min_sleep_interval": 2,
    "quiet":              True,
    "no_warnings":        True,
    "ignoreerrors":       True,
}

# ── lyricsgenius defaults ─────────────────────────────────────────────────────
GENIUS_CONFIG = {
    "sleep_time":            0.5,
    "timeout":               15,
    "retries":               3,
    "remove_section_headers": False,   # KEEP [Verse], [Chorus] etc. for segment analysis
    "skip_non_english":      False,
}

# ── Segment analysis ──────────────────────────────────────────────────────────
SEGMENT_CONFIG = {
    "enabled":             True,
    "min_sections":        2,
    "min_segment_secs":    10.0,
    "section_markers":     ["verse", "chorus", "bridge", "hook",
                             "pre-chorus", "outro", "intro", "refrain", "break"],
}

# ─────────────────────────────────────────────────────────────────────────────
# ARTIST CATALOG
# Each song entry: title | yt_query (YouTube search) | genius_query
# IDs follow the pattern {ArtistCode}_{NN}
# ─────────────────────────────────────────────────────────────────────────────

CATALOG = {

    # ══════════════════════════════════════════════
    #  NARRATIVE-FORWARD — HIP-HOP
    # ══════════════════════════════════════════════

    "RTJ": {
        "name":        "Run the Jewels",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "RTJ",
        "songs": {
            "RTJ_01": {"title": "Run the Jewels",                 "yt_query": "Run the Jewels Run the Jewels song audio",               "genius_query": "Run the Jewels Run the Jewels"},
            "RTJ_02": {"title": "Banana Clipper",                 "yt_query": "Run the Jewels Banana Clipper Big Boi audio",             "genius_query": "Banana Clipper Run the Jewels"},
            "RTJ_03": {"title": "36 Inch Chain",                  "yt_query": "Run the Jewels 36 Inch Chain audio",                     "genius_query": "36 Inch Chain Run the Jewels"},
            "RTJ_04": {"title": "DDHF",                           "yt_query": "Run the Jewels DDHF audio",                              "genius_query": "DDHF Run the Jewels"},
            "RTJ_05": {"title": "Sea Legs",                       "yt_query": "Run the Jewels Sea Legs audio",                          "genius_query": "Sea Legs Run the Jewels"},
            "RTJ_06": {"title": "Job Well Done",                  "yt_query": "Run the Jewels Job Well Done audio",                     "genius_query": "Job Well Done Run the Jewels"},
            "RTJ_07": {"title": "No Come Down",                   "yt_query": "Run the Jewels No Come Down audio",                      "genius_query": "No Come Down Run the Jewels"},
            "RTJ_08": {"title": "Get It",                         "yt_query": "Run the Jewels Get It audio",                            "genius_query": "Get It Run the Jewels"},
            "RTJ_09": {"title": "Twin Hype Back",                 "yt_query": "Run the Jewels Twin Hype Back audio",                    "genius_query": "Twin Hype Back Run the Jewels"},
            "RTJ_10": {"title": "A Christmas Fucking Miracle",    "yt_query": "Run the Jewels Christmas Fucking Miracle audio",         "genius_query": "A Christmas Fucking Miracle Run the Jewels"},
            "RTJ_11": {"title": "Blockbuster Night Part 1",       "yt_query": "Run the Jewels Blockbuster Night Part 1 audio",          "genius_query": "Blockbuster Night Part 1 Run the Jewels"},
            "RTJ_12": {"title": "Close Your Eyes (And Count to Fuck)", "yt_query": "Run the Jewels Close Your Eyes Count Fuck audio",  "genius_query": "Close Your Eyes Run the Jewels"},
            "RTJ_13": {"title": "Legend Has It",                  "yt_query": "Run the Jewels Legend Has It audio",                    "genius_query": "Legend Has It Run the Jewels"},
            "RTJ_14": {"title": "Talk to Me",                     "yt_query": "Run the Jewels Talk to Me audio",                       "genius_query": "Talk to Me Run the Jewels"},
            "RTJ_15": {"title": "JU$T",                           "yt_query": "Run the Jewels JUST Pharrell Williams audio",            "genius_query": "JUST Run the Jewels"},
        },
    },

    "KL": {
        "name":        "Kendrick Lamar",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "KL",
        "songs": {
            "KL_01": {"title": "HUMBLE.",                   "yt_query": "Kendrick Lamar HUMBLE official audio",                  "genius_query": "HUMBLE Kendrick Lamar"},
            "KL_02": {"title": "DNA.",                      "yt_query": "Kendrick Lamar DNA official audio",                     "genius_query": "DNA Kendrick Lamar"},
            "KL_03": {"title": "Alright",                   "yt_query": "Kendrick Lamar Alright official audio",                 "genius_query": "Alright Kendrick Lamar"},
            "KL_04": {"title": "Swimming Pools (Drank)",    "yt_query": "Kendrick Lamar Swimming Pools Drank audio",             "genius_query": "Swimming Pools Kendrick Lamar"},
            "KL_05": {"title": "Money Trees",               "yt_query": "Kendrick Lamar Money Trees Jay Rock audio",             "genius_query": "Money Trees Kendrick Lamar"},
            "KL_06": {"title": "m.A.A.d city",              "yt_query": "Kendrick Lamar maad city audio",                        "genius_query": "maad city Kendrick Lamar"},
            "KL_07": {"title": "Backseat Freestyle",        "yt_query": "Kendrick Lamar Backseat Freestyle audio",               "genius_query": "Backseat Freestyle Kendrick Lamar"},
            "KL_08": {"title": "King Kunta",                "yt_query": "Kendrick Lamar King Kunta audio",                       "genius_query": "King Kunta Kendrick Lamar"},
            "KL_09": {"title": "Bitch, Don't Kill My Vibe", "yt_query": "Kendrick Lamar Bitch Dont Kill My Vibe audio",         "genius_query": "Bitch Dont Kill My Vibe Kendrick Lamar"},
            "KL_10": {"title": "Poetic Justice",            "yt_query": "Kendrick Lamar Poetic Justice Drake audio",             "genius_query": "Poetic Justice Kendrick Lamar"},
            "KL_11": {"title": "These Walls",               "yt_query": "Kendrick Lamar These Walls audio",                     "genius_query": "These Walls Kendrick Lamar"},
            "KL_12": {"title": "i",                         "yt_query": "Kendrick Lamar i official audio",                      "genius_query": "i Kendrick Lamar"},
            "KL_13": {"title": "Die Hard",                  "yt_query": "Kendrick Lamar Die Hard Blxst Amanda Reifer audio",     "genius_query": "Die Hard Kendrick Lamar"},
            "KL_14": {"title": "Rich Spirit",               "yt_query": "Kendrick Lamar Rich Spirit audio",                     "genius_query": "Rich Spirit Kendrick Lamar"},
            "KL_15": {"title": "N95",                       "yt_query": "Kendrick Lamar N95 audio",                              "genius_query": "N95 Kendrick Lamar"},
        },
    },

    "JC": {
        "name":        "J. Cole",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "JC",
        "songs": {
            "JC_01": {"title": "No Role Modelz",        "yt_query": "J Cole No Role Modelz audio",             "genius_query": "No Role Modelz J Cole"},
            "JC_02": {"title": "Love Yourz",            "yt_query": "J Cole Love Yourz audio",                 "genius_query": "Love Yourz J Cole"},
            "JC_03": {"title": "Power Trip",            "yt_query": "J Cole Power Trip Miguel audio",          "genius_query": "Power Trip J Cole"},
            "JC_04": {"title": "Middle Child",          "yt_query": "J Cole Middle Child audio",               "genius_query": "Middle Child J Cole"},
            "JC_05": {"title": "ATM",                   "yt_query": "J Cole ATM audio",                        "genius_query": "ATM J Cole"},
            "JC_06": {"title": "Apparently",            "yt_query": "J Cole Apparently audio",                 "genius_query": "Apparently J Cole"},
            "JC_07": {"title": "Kevin's Heart",         "yt_query": "J Cole Kevins Heart audio",               "genius_query": "Kevins Heart J Cole"},
            "JC_08": {"title": "Wet Dreamz",            "yt_query": "J Cole Wet Dreamz audio",                 "genius_query": "Wet Dreamz J Cole"},
            "JC_09": {"title": "January 28th",          "yt_query": "J Cole January 28th audio",               "genius_query": "January 28th J Cole"},
            "JC_10": {"title": "HiiiPoWeR",             "yt_query": "J Cole HiiiPoWeR audio",                  "genius_query": "HiiiPoWeR J Cole"},
            "JC_11": {"title": "Work Out",              "yt_query": "J Cole Work Out audio",                   "genius_query": "Work Out J Cole"},
            "JC_12": {"title": "For Whom the Bell Tolls","yt_query": "J Cole For Whom the Bell Tolls audio",   "genius_query": "For Whom the Bell Tolls J Cole"},
            "JC_13": {"title": "G.O.M.D.",              "yt_query": "J Cole GOMD audio",                       "genius_query": "GOMD J Cole"},
            "JC_14": {"title": "She Knows",             "yt_query": "J Cole She Knows audio",                  "genius_query": "She Knows J Cole"},
            "JC_15": {"title": "Forbidden Fruit",       "yt_query": "J Cole Forbidden Fruit Jhene Aiko audio", "genius_query": "Forbidden Fruit J Cole"},
        },
    },

    "DR": {
        "name":        "Drake",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "DR",
        "songs": {
            "DR_01": {"title": "God's Plan",              "yt_query": "Drake Gods Plan official audio",               "genius_query": "Gods Plan Drake"},
            "DR_02": {"title": "Hotline Bling",           "yt_query": "Drake Hotline Bling official audio",           "genius_query": "Hotline Bling Drake"},
            "DR_03": {"title": "One Dance",               "yt_query": "Drake One Dance official audio",               "genius_query": "One Dance Drake"},
            "DR_04": {"title": "Passionfruit",            "yt_query": "Drake Passionfruit official audio",            "genius_query": "Passionfruit Drake"},
            "DR_05": {"title": "Started From the Bottom", "yt_query": "Drake Started From the Bottom audio",          "genius_query": "Started From the Bottom Drake"},
            "DR_06": {"title": "Take Care",               "yt_query": "Drake Take Care Rihanna audio",                "genius_query": "Take Care Drake"},
            "DR_07": {"title": "Marvins Room",            "yt_query": "Drake Marvins Room audio",                     "genius_query": "Marvins Room Drake"},
            "DR_08": {"title": "Hold On We're Going Home","yt_query": "Drake Hold On Were Going Home audio",           "genius_query": "Hold On Were Going Home Drake"},
            "DR_09": {"title": "Forever",                 "yt_query": "Drake Forever Kanye Lil Wayne Eminem audio",   "genius_query": "Forever Drake"},
            "DR_10": {"title": "Best I Ever Had",         "yt_query": "Drake Best I Ever Had audio",                  "genius_query": "Best I Ever Had Drake"},
            "DR_11": {"title": "Fancy",                   "yt_query": "Drake Fancy audio",                           "genius_query": "Fancy Drake"},
            "DR_12": {"title": "From Time",               "yt_query": "Drake From Time Jhene Aiko audio",            "genius_query": "From Time Drake"},
        },
    },

    "EM": {
        "name":        "Eminem",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "EM",
        "songs": {
            "EM_01": {"title": "Lose Yourself",           "yt_query": "Eminem Lose Yourself official audio",          "genius_query": "Lose Yourself Eminem"},
            "EM_02": {"title": "Stan",                    "yt_query": "Eminem Stan Dido audio",                       "genius_query": "Stan Eminem"},
            "EM_03": {"title": "The Real Slim Shady",     "yt_query": "Eminem The Real Slim Shady audio",             "genius_query": "The Real Slim Shady Eminem"},
            "EM_04": {"title": "Without Me",              "yt_query": "Eminem Without Me audio",                      "genius_query": "Without Me Eminem"},
            "EM_05": {"title": "Not Afraid",              "yt_query": "Eminem Not Afraid audio",                      "genius_query": "Not Afraid Eminem"},
            "EM_06": {"title": "Love The Way You Lie",    "yt_query": "Eminem Love The Way You Lie Rihanna audio",    "genius_query": "Love The Way You Lie Eminem"},
            "EM_07": {"title": "Rap God",                 "yt_query": "Eminem Rap God audio",                        "genius_query": "Rap God Eminem"},
            "EM_08": {"title": "River",                   "yt_query": "Eminem River Ed Sheeran audio",                "genius_query": "River Eminem"},
            "EM_09": {"title": "Cleanin Out My Closet",   "yt_query": "Eminem Cleanin Out My Closet audio",           "genius_query": "Cleanin Out My Closet Eminem"},
            "EM_10": {"title": "When I'm Gone",           "yt_query": "Eminem When Im Gone audio",                   "genius_query": "When Im Gone Eminem"},
            "EM_11": {"title": "Mockingbird",             "yt_query": "Eminem Mockingbird audio",                    "genius_query": "Mockingbird Eminem"},
            "EM_12": {"title": "Beautiful",               "yt_query": "Eminem Beautiful audio",                      "genius_query": "Beautiful Eminem"},
        },
    },

    "TC2": {
        "name":        "Tyler, the Creator",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "TC2",
        "songs": {
            "TC2_01": {"title": "See You Again",           "yt_query": "Tyler the Creator See You Again Kali Uchis audio",  "genius_query": "See You Again Tyler the Creator"},
            "TC2_02": {"title": "EARFQUAKE",               "yt_query": "Tyler the Creator EARFQUAKE audio",                "genius_query": "EARFQUAKE Tyler the Creator"},
            "TC2_03": {"title": "Enjoy Right Now Today",   "yt_query": "Tyler the Creator Enjoy Right Now Today audio",    "genius_query": "Enjoy Right Now Today Tyler the Creator"},
            "TC2_04": {"title": "I THINK",                 "yt_query": "Tyler the Creator I THINK audio",                 "genius_query": "I THINK Tyler the Creator"},
            "TC2_05": {"title": "NEW MAGIC WAND",          "yt_query": "Tyler the Creator NEW MAGIC WAND audio",           "genius_query": "NEW MAGIC WAND Tyler the Creator"},
            "TC2_06": {"title": "WUSYANAME",               "yt_query": "Tyler the Creator WUSYANAME audio",               "genius_query": "WUSYANAME Tyler the Creator"},
            "TC2_07": {"title": "Lumberjack",              "yt_query": "Tyler the Creator Lumberjack audio",              "genius_query": "Lumberjack Tyler the Creator"},
            "TC2_08": {"title": "911 / Mr. Lonely",        "yt_query": "Tyler the Creator 911 Mr Lonely audio",           "genius_query": "911 Mr Lonely Tyler the Creator"},
            "TC2_09": {"title": "Who Dat Boy",             "yt_query": "Tyler the Creator Who Dat Boy audio",             "genius_query": "Who Dat Boy Tyler the Creator"},
            "TC2_10": {"title": "Gone Gone / Thank You",   "yt_query": "Tyler the Creator Gone Gone Thank You audio",     "genius_query": "Gone Gone Tyler the Creator"},
            "TC2_11": {"title": "Foreword",                "yt_query": "Tyler the Creator Foreword Rex Orange County audio","genius_query": "Foreword Tyler the Creator"},
            "TC2_12": {"title": "MASSA",                   "yt_query": "Tyler the Creator MASSA audio",                   "genius_query": "MASSA Tyler the Creator"},
        },
    },

    "MM": {
        "name":        "Mac Miller",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "MM",
        "songs": {
            "MM_01": {"title": "Self Care",            "yt_query": "Mac Miller Self Care audio",               "genius_query": "Self Care Mac Miller"},
            "MM_02": {"title": "Small Worlds",         "yt_query": "Mac Miller Small Worlds audio",            "genius_query": "Small Worlds Mac Miller"},
            "MM_03": {"title": "Circles",              "yt_query": "Mac Miller Circles audio",                 "genius_query": "Circles Mac Miller"},
            "MM_04": {"title": "2009",                 "yt_query": "Mac Miller 2009 audio",                    "genius_query": "2009 Mac Miller"},
            "MM_05": {"title": "Objects in the Mirror","yt_query": "Mac Miller Objects in the Mirror audio",   "genius_query": "Objects in the Mirror Mac Miller"},
            "MM_06": {"title": "Diablo",               "yt_query": "Mac Miller Diablo audio",                  "genius_query": "Diablo Mac Miller"},
            "MM_07": {"title": "Programs",             "yt_query": "Mac Miller Programs audio",                "genius_query": "Programs Mac Miller"},
            "MM_08": {"title": "Dunno",                "yt_query": "Mac Miller Dunno audio",                   "genius_query": "Dunno Mac Miller"},
            "MM_09": {"title": "Come Back to Earth",   "yt_query": "Mac Miller Come Back to Earth audio",      "genius_query": "Come Back to Earth Mac Miller"},
            "MM_10": {"title": "Good News",            "yt_query": "Mac Miller Good News audio",               "genius_query": "Good News Mac Miller"},
            "MM_11": {"title": "What's the Use?",      "yt_query": "Mac Miller Whats the Use audio",           "genius_query": "Whats the Use Mac Miller"},
            "MM_12": {"title": "Grand Finale",         "yt_query": "Mac Miller Grand Finale audio",            "genius_query": "Grand Finale Mac Miller"},
        },
    },

    "CH": {
        "name":        "Chance the Rapper",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "CH",
        "songs": {
            "CH_01": {"title": "Blessings",              "yt_query": "Chance the Rapper Blessings audio",                    "genius_query": "Blessings Chance the Rapper"},
            "CH_02": {"title": "No Problem",             "yt_query": "Chance the Rapper No Problem Lil Wayne 2 Chainz audio","genius_query": "No Problem Chance the Rapper"},
            "CH_03": {"title": "Same Drugs",             "yt_query": "Chance the Rapper Same Drugs audio",                   "genius_query": "Same Drugs Chance the Rapper"},
            "CH_04": {"title": "Angels",                 "yt_query": "Chance the Rapper Angels Saba audio",                  "genius_query": "Angels Chance the Rapper"},
            "CH_05": {"title": "Paranoia",               "yt_query": "Chance the Rapper Paranoia audio",                     "genius_query": "Paranoia Chance the Rapper"},
            "CH_06": {"title": "Sunday Candy",           "yt_query": "Chance the Rapper Sunday Candy audio",                 "genius_query": "Sunday Candy Chance the Rapper"},
            "CH_07": {"title": "Favorite Song",          "yt_query": "Chance the Rapper Favorite Song Childish Gambino audio","genius_query": "Favorite Song Chance the Rapper"},
            "CH_08": {"title": "All We Got",             "yt_query": "Chance the Rapper All We Got Kanye West audio",        "genius_query": "All We Got Chance the Rapper"},
            "CH_09": {"title": "Cocoa Butter Kisses",    "yt_query": "Chance the Rapper Cocoa Butter Kisses audio",          "genius_query": "Cocoa Butter Kisses Chance the Rapper"},
            "CH_10": {"title": "Smoke Again",            "yt_query": "Chance the Rapper Smoke Again audio",                  "genius_query": "Smoke Again Chance the Rapper"},
            "CH_11": {"title": "Hot Shower",             "yt_query": "Chance the Rapper Hot Shower DaBaby audio",            "genius_query": "Hot Shower Chance the Rapper"},
            "CH_12": {"title": "I Might Need Security",  "yt_query": "Chance the Rapper I Might Need Security audio",        "genius_query": "I Might Need Security Chance the Rapper"},
        },
    },

    "NS": {
        "name":        "Nas",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "NS",
        "songs": {
            "NS_01": {"title": "N.Y. State of Mind",   "yt_query": "Nas NY State of Mind audio",            "genius_query": "NY State of Mind Nas"},
            "NS_02": {"title": "The World Is Yours",   "yt_query": "Nas The World Is Yours audio",          "genius_query": "The World Is Yours Nas"},
            "NS_03": {"title": "One Love",             "yt_query": "Nas One Love audio",                    "genius_query": "One Love Nas"},
            "NS_04": {"title": "If I Ruled the World", "yt_query": "Nas If I Ruled the World Lauryn Hill audio","genius_query": "If I Ruled the World Nas"},
            "NS_05": {"title": "Hate Me Now",          "yt_query": "Nas Hate Me Now audio",                 "genius_query": "Hate Me Now Nas"},
            "NS_06": {"title": "One Mic",              "yt_query": "Nas One Mic audio",                     "genius_query": "One Mic Nas"},
            "NS_07": {"title": "Made You Look",        "yt_query": "Nas Made You Look audio",               "genius_query": "Made You Look Nas"},
            "NS_08": {"title": "Daughters",            "yt_query": "Nas Daughters audio",                   "genius_query": "Daughters Nas"},
            "NS_09": {"title": "Illmatic",             "yt_query": "Nas Illmatic audio",                   "genius_query": "Illmatic Nas"},
            "NS_10": {"title": "Represent",            "yt_query": "Nas Represent audio",                   "genius_query": "Represent Nas"},
            "NS_11": {"title": "Life's a Bitch",       "yt_query": "Nas Lifes a Bitch AZ audio",            "genius_query": "Lifes a Bitch Nas"},
            "NS_12": {"title": "Cherry Wine",          "yt_query": "Nas Cherry Wine Amy Winehouse audio",   "genius_query": "Cherry Wine Nas"},
        },
    },

    "CG": {
        "name":        "Childish Gambino",
        "genre":       "hip-hop",
        "orientation": "narrative",
        "folder":      "CG",
        "songs": {
            "CG_01": {"title": "Redbone",              "yt_query": "Childish Gambino Redbone audio",              "genius_query": "Redbone Childish Gambino"},
            "CG_02": {"title": "This Is America",      "yt_query": "Childish Gambino This Is America audio",      "genius_query": "This Is America Childish Gambino"},
            "CG_03": {"title": "3005",                 "yt_query": "Childish Gambino 3005 audio",                 "genius_query": "3005 Childish Gambino"},
            "CG_04": {"title": "Sober",                "yt_query": "Childish Gambino Sober audio",               "genius_query": "Sober Childish Gambino"},
            "CG_05": {"title": "Sweatpants",           "yt_query": "Childish Gambino Sweatpants audio",           "genius_query": "Sweatpants Childish Gambino"},
            "CG_06": {"title": "V. 3005 (Beach Picnic Version)", "yt_query": "Childish Gambino Camp audio",      "genius_query": "Camp Childish Gambino"},
            "CG_07": {"title": "Me and Your Mama",     "yt_query": "Childish Gambino Me and Your Mama audio",    "genius_query": "Me and Your Mama Childish Gambino"},
            "CG_08": {"title": "Terrified",            "yt_query": "Childish Gambino Terrified audio",           "genius_query": "Terrified Childish Gambino"},
            "CG_09": {"title": "Heartbeat",            "yt_query": "Childish Gambino Heartbeat audio",           "genius_query": "Heartbeat Childish Gambino"},
            "CG_10": {"title": "Outside",              "yt_query": "Childish Gambino Outside audio",             "genius_query": "Outside Childish Gambino"},
            "CG_11": {"title": "telegraphs",           "yt_query": "Childish Gambino telegraphs audio",          "genius_query": "telegraphs Childish Gambino"},
            "CG_12": {"title": "California",           "yt_query": "Childish Gambino California audio",          "genius_query": "California Childish Gambino"},
        },
    },

    # ══════════════════════════════════════════════
    #  NARRATIVE-FORWARD — FOLK / COUNTRY / AMERICANA
    # ══════════════════════════════════════════════

    "GD": {
        "name":        "Grateful Dead",
        "genre":       "folk-rock",
        "orientation": "narrative",
        "folder":      "GD",
        "songs": {
            "GD_01": {"title": "Minglewood Blues",      "yt_query": "Grateful Dead Minglewood Blues studio",      "genius_query": "Minglewood Blues Grateful Dead"},
            "GD_02": {"title": "They Love Each Other",  "yt_query": "Grateful Dead They Love Each Other audio",   "genius_query": "They Love Each Other Grateful Dead"},
            "GD_03": {"title": "Cassidy",               "yt_query": "Grateful Dead Cassidy studio audio",         "genius_query": "Cassidy Grateful Dead"},
            "GD_04": {"title": "Loser",                 "yt_query": "Grateful Dead Loser studio audio",           "genius_query": "Loser Grateful Dead"},
            "GD_05": {"title": "Jack Straw",            "yt_query": "Grateful Dead Jack Straw studio audio",      "genius_query": "Jack Straw Grateful Dead"},
            "GD_06": {"title": "Tennessee Jed",         "yt_query": "Grateful Dead Tennessee Jed audio",          "genius_query": "Tennessee Jed Grateful Dead"},
            "GD_07": {"title": "Passenger",             "yt_query": "Grateful Dead Passenger audio",              "genius_query": "Passenger Grateful Dead"},
            "GD_08": {"title": "Peggy-O",               "yt_query": "Grateful Dead Peggy-O audio",                "genius_query": "Peggy-O Grateful Dead"},
            "GD_09": {"title": "Me & My Uncle",         "yt_query": "Grateful Dead Me and My Uncle audio",        "genius_query": "Me and My Uncle Grateful Dead"},
            "GD_10": {"title": "Friend of the Devil",   "yt_query": "Grateful Dead Friend of the Devil studio",   "genius_query": "Friend of the Devil Grateful Dead"},
            "GD_11": {"title": "Truckin'",              "yt_query": "Grateful Dead Truckin audio",                "genius_query": "Truckin Grateful Dead"},
            "GD_12": {"title": "Casey Jones",           "yt_query": "Grateful Dead Casey Jones audio",            "genius_query": "Casey Jones Grateful Dead"},
            "GD_13": {"title": "Touch of Grey",         "yt_query": "Grateful Dead Touch of Grey audio",          "genius_query": "Touch of Grey Grateful Dead"},
            "GD_14": {"title": "Ripple",                "yt_query": "Grateful Dead Ripple studio audio",          "genius_query": "Ripple Grateful Dead"},
            "GD_15": {"title": "Scarlet Begonias",      "yt_query": "Grateful Dead Scarlet Begonias studio",      "genius_query": "Scarlet Begonias Grateful Dead"},
        },
    },

    "BD": {
        "name":        "Bob Dylan",
        "genre":       "folk",
        "orientation": "narrative",
        "folder":      "BD",
        "songs": {
            "BD_01": {"title": "Blowin' in the Wind",               "yt_query": "Bob Dylan Blowin in the Wind audio",              "genius_query": "Blowin in the Wind Bob Dylan"},
            "BD_02": {"title": "The Times They Are A-Changin'",     "yt_query": "Bob Dylan Times They Are Changing audio",         "genius_query": "The Times They Are Changin Bob Dylan"},
            "BD_03": {"title": "Like a Rolling Stone",              "yt_query": "Bob Dylan Like a Rolling Stone audio",            "genius_query": "Like a Rolling Stone Bob Dylan"},
            "BD_04": {"title": "Mr. Tambourine Man",                "yt_query": "Bob Dylan Mr Tambourine Man audio",               "genius_query": "Mr Tambourine Man Bob Dylan"},
            "BD_05": {"title": "Knockin' on Heaven's Door",         "yt_query": "Bob Dylan Knockin on Heavens Door audio",         "genius_query": "Knockin on Heavens Door Bob Dylan"},
            "BD_06": {"title": "Tangled Up in Blue",                "yt_query": "Bob Dylan Tangled Up in Blue audio",              "genius_query": "Tangled Up in Blue Bob Dylan"},
            "BD_07": {"title": "Hurricane",                         "yt_query": "Bob Dylan Hurricane audio",                      "genius_query": "Hurricane Bob Dylan"},
            "BD_08": {"title": "It Ain't Me, Babe",                 "yt_query": "Bob Dylan It Aint Me Babe audio",                 "genius_query": "It Aint Me Babe Bob Dylan"},
            "BD_09": {"title": "Don't Think Twice, It's All Right", "yt_query": "Bob Dylan Dont Think Twice Its All Right audio",  "genius_query": "Dont Think Twice Its All Right Bob Dylan"},
            "BD_10": {"title": "Lay Lady Lay",                      "yt_query": "Bob Dylan Lay Lady Lay audio",                   "genius_query": "Lay Lady Lay Bob Dylan"},
            "BD_11": {"title": "Simple Twist of Fate",              "yt_query": "Bob Dylan Simple Twist of Fate audio",           "genius_query": "Simple Twist of Fate Bob Dylan"},
            "BD_12": {"title": "Shelter from the Storm",            "yt_query": "Bob Dylan Shelter from the Storm audio",         "genius_query": "Shelter from the Storm Bob Dylan"},
            "BD_13": {"title": "Forever Young",                     "yt_query": "Bob Dylan Forever Young audio",                  "genius_query": "Forever Young Bob Dylan"},
            "BD_14": {"title": "Just Like a Woman",                 "yt_query": "Bob Dylan Just Like a Woman audio",              "genius_query": "Just Like a Woman Bob Dylan"},
            "BD_15": {"title": "Girl from the North Country",       "yt_query": "Bob Dylan Girl from the North Country audio",    "genius_query": "Girl from the North Country Bob Dylan"},
        },
    },

    "TC": {
        "name":        "Tyler Childers",
        "genre":       "country",
        "orientation": "narrative",
        "folder":      "TC",
        "songs": {
            "TC_01": {"title": "Whitehouse Road",       "yt_query": "Tyler Childers Whitehouse Road audio",        "genius_query": "Whitehouse Road Tyler Childers"},
            "TC_02": {"title": "Lady May",              "yt_query": "Tyler Childers Lady May audio",               "genius_query": "Lady May Tyler Childers"},
            "TC_03": {"title": "Feathered Indians",     "yt_query": "Tyler Childers Feathered Indians audio",      "genius_query": "Feathered Indians Tyler Childers"},
            "TC_04": {"title": "All Your'n",            "yt_query": "Tyler Childers All Yourn audio",              "genius_query": "All Yourn Tyler Childers"},
            "TC_05": {"title": "Hard Way",              "yt_query": "Tyler Childers Hard Way audio",               "genius_query": "Hard Way Tyler Childers"},
            "TC_06": {"title": "Universal Sound",       "yt_query": "Tyler Childers Universal Sound audio",        "genius_query": "Universal Sound Tyler Childers"},
            "TC_07": {"title": "House Fire",            "yt_query": "Tyler Childers House Fire audio",             "genius_query": "House Fire Tyler Childers"},
            "TC_08": {"title": "Creeker",               "yt_query": "Tyler Childers Creeker audio",                "genius_query": "Creeker Tyler Childers"},
            "TC_09": {"title": "Shake the Frost",       "yt_query": "Tyler Childers Shake the Frost audio",        "genius_query": "Shake the Frost Tyler Childers"},
            "TC_10": {"title": "Follow You to Virgie",  "yt_query": "Tyler Childers Follow You to Virgie audio",   "genius_query": "Follow You to Virgie Tyler Childers"},
            "TC_11": {"title": "Banded Clovis",         "yt_query": "Tyler Childers Banded Clovis audio",          "genius_query": "Banded Clovis Tyler Childers"},
            "TC_12": {"title": "Nose on the Grindstone","yt_query": "Tyler Childers Nose on the Grindstone audio", "genius_query": "Nose on the Grindstone Tyler Childers"},
            "TC_13": {"title": "Country Squire",        "yt_query": "Tyler Childers Country Squire audio",         "genius_query": "Country Squire Tyler Childers"},
            "TC_14": {"title": "Ever Lovin' Hand",      "yt_query": "Tyler Childers Ever Lovin Hand audio",        "genius_query": "Ever Lovin Hand Tyler Childers"},
            "TC_15": {"title": "Matthew",               "yt_query": "Tyler Childers Matthew audio",                "genius_query": "Matthew Tyler Childers"},
        },
    },

    "JI": {
        "name":        "Jason Isbell",
        "genre":       "country",
        "orientation": "narrative",
        "folder":      "JI",
        "songs": {
            "JI_01": {"title": "Cover Me Up",            "yt_query": "Jason Isbell Cover Me Up audio",               "genius_query": "Cover Me Up Jason Isbell"},
            "JI_02": {"title": "If We Were Vampires",    "yt_query": "Jason Isbell If We Were Vampires audio",        "genius_query": "If We Were Vampires Jason Isbell"},
            "JI_03": {"title": "Death Wish",             "yt_query": "Jason Isbell Death Wish audio",                "genius_query": "Death Wish Jason Isbell"},
            "JI_04": {"title": "White Man's World",      "yt_query": "Jason Isbell White Mans World audio",           "genius_query": "White Mans World Jason Isbell"},
            "JI_05": {"title": "Last of My Kind",        "yt_query": "Jason Isbell Last of My Kind audio",            "genius_query": "Last of My Kind Jason Isbell"},
            "JI_06": {"title": "Something More Than Free","yt_query": "Jason Isbell Something More Than Free audio",  "genius_query": "Something More Than Free Jason Isbell"},
            "JI_07": {"title": "Outfit",                 "yt_query": "Jason Isbell Outfit audio",                    "genius_query": "Outfit Jason Isbell"},
            "JI_08": {"title": "Elephant",               "yt_query": "Jason Isbell Elephant audio",                  "genius_query": "Elephant Jason Isbell"},
            "JI_09": {"title": "Speed Trap Town",        "yt_query": "Jason Isbell Speed Trap Town audio",            "genius_query": "Speed Trap Town Jason Isbell"},
            "JI_10": {"title": "Relatively Easy",        "yt_query": "Jason Isbell Relatively Easy audio",           "genius_query": "Relatively Easy Jason Isbell"},
            "JI_11": {"title": "Flying Over Water",      "yt_query": "Jason Isbell Flying Over Water audio",         "genius_query": "Flying Over Water Jason Isbell"},
            "JI_12": {"title": "Children of Children",   "yt_query": "Jason Isbell Children of Children audio",      "genius_query": "Children of Children Jason Isbell"},
        },
    },

    "SS": {
        "name":        "Sturgill Simpson",
        "genre":       "country",
        "orientation": "narrative",
        "folder":      "SS",
        "songs": {
            "SS_01": {"title": "Turtles All the Way Down","yt_query": "Sturgill Simpson Turtles All the Way Down audio","genius_query": "Turtles All the Way Down Sturgill Simpson"},
            "SS_02": {"title": "Life of Sin",             "yt_query": "Sturgill Simpson Life of Sin audio",            "genius_query": "Life of Sin Sturgill Simpson"},
            "SS_03": {"title": "Long White Line",         "yt_query": "Sturgill Simpson Long White Line audio",        "genius_query": "Long White Line Sturgill Simpson"},
            "SS_04": {"title": "Brace for Impact",        "yt_query": "Sturgill Simpson Brace for Impact audio",       "genius_query": "Brace for Impact Sturgill Simpson"},
            "SS_05": {"title": "In Bloom",                "yt_query": "Sturgill Simpson In Bloom audio",               "genius_query": "In Bloom Sturgill Simpson"},
            "SS_06": {"title": "Welcome to Earth",        "yt_query": "Sturgill Simpson Welcome to Earth audio",       "genius_query": "Welcome to Earth Sturgill Simpson"},
            "SS_07": {"title": "Keep It Between the Lines","yt_query": "Sturgill Simpson Keep It Between the Lines audio","genius_query": "Keep It Between the Lines Sturgill Simpson"},
            "SS_08": {"title": "Water in a Well",         "yt_query": "Sturgill Simpson Water in a Well audio",        "genius_query": "Water in a Well Sturgill Simpson"},
            "SS_09": {"title": "Call to Arms",            "yt_query": "Sturgill Simpson Call to Arms audio",           "genius_query": "Call to Arms Sturgill Simpson"},
            "SS_10": {"title": "Oh Sarah",                "yt_query": "Sturgill Simpson Oh Sarah audio",               "genius_query": "Oh Sarah Sturgill Simpson"},
            "SS_11": {"title": "You Can Have the Crown",  "yt_query": "Sturgill Simpson You Can Have the Crown audio", "genius_query": "You Can Have the Crown Sturgill Simpson"},
            "SS_12": {"title": "All Around You",          "yt_query": "Sturgill Simpson All Around You audio",         "genius_query": "All Around You Sturgill Simpson"},
        },
    },

    "KM": {
        "name":        "Kacey Musgraves",
        "genre":       "country",
        "orientation": "narrative",
        "folder":      "KM",
        "songs": {
            "KM_01": {"title": "Golden Hour",          "yt_query": "Kacey Musgraves Golden Hour audio",           "genius_query": "Golden Hour Kacey Musgraves"},
            "KM_02": {"title": "Butterflies",          "yt_query": "Kacey Musgraves Butterflies audio",           "genius_query": "Butterflies Kacey Musgraves"},
            "KM_03": {"title": "Happy & Sad",          "yt_query": "Kacey Musgraves Happy and Sad audio",         "genius_query": "Happy and Sad Kacey Musgraves"},
            "KM_04": {"title": "Space Cowboy",         "yt_query": "Kacey Musgraves Space Cowboy audio",          "genius_query": "Space Cowboy Kacey Musgraves"},
            "KM_05": {"title": "Rainbow",              "yt_query": "Kacey Musgraves Rainbow audio",               "genius_query": "Rainbow Kacey Musgraves"},
            "KM_06": {"title": "Slow Burn",            "yt_query": "Kacey Musgraves Slow Burn audio",             "genius_query": "Slow Burn Kacey Musgraves"},
            "KM_07": {"title": "Lonely Weekend",       "yt_query": "Kacey Musgraves Lonely Weekend audio",        "genius_query": "Lonely Weekend Kacey Musgraves"},
            "KM_08": {"title": "Follow Your Arrow",    "yt_query": "Kacey Musgraves Follow Your Arrow audio",     "genius_query": "Follow Your Arrow Kacey Musgraves"},
            "KM_09": {"title": "Merry Go Round",       "yt_query": "Kacey Musgraves Merry Go Round audio",        "genius_query": "Merry Go Round Kacey Musgraves"},
            "KM_10": {"title": "Velvet Elvis",         "yt_query": "Kacey Musgraves Velvet Elvis audio",          "genius_query": "Velvet Elvis Kacey Musgraves"},
            "KM_11": {"title": "Justified",            "yt_query": "Kacey Musgraves Justified audio",             "genius_query": "Justified Kacey Musgraves"},
            "KM_12": {"title": "Keep Lookin Up",       "yt_query": "Kacey Musgraves Keep Lookin Up audio",        "genius_query": "Keep Lookin Up Kacey Musgraves"},
        },
    },

    "JP": {
        "name":        "John Prine",
        "genre":       "folk",
        "orientation": "narrative",
        "folder":      "JP",
        "songs": {
            "JP_01": {"title": "Angel from Montgomery",  "yt_query": "John Prine Angel from Montgomery audio",       "genius_query": "Angel from Montgomery John Prine"},
            "JP_02": {"title": "Sam Stone",              "yt_query": "John Prine Sam Stone audio",                   "genius_query": "Sam Stone John Prine"},
            "JP_03": {"title": "Hello in There",         "yt_query": "John Prine Hello in There audio",              "genius_query": "Hello in There John Prine"},
            "JP_04": {"title": "Paradise",               "yt_query": "John Prine Paradise audio",                    "genius_query": "Paradise John Prine"},
            "JP_05": {"title": "Spanish Pipedream",      "yt_query": "John Prine Spanish Pipedream audio",           "genius_query": "Spanish Pipedream John Prine"},
            "JP_06": {"title": "Donald and Lydia",       "yt_query": "John Prine Donald and Lydia audio",            "genius_query": "Donald and Lydia John Prine"},
            "JP_07": {"title": "Souvenirs",              "yt_query": "John Prine Souvenirs audio",                   "genius_query": "Souvenirs John Prine"},
            "JP_08": {"title": "Lake Marie",             "yt_query": "John Prine Lake Marie audio",                  "genius_query": "Lake Marie John Prine"},
            "JP_09": {"title": "In Spite of Ourselves",  "yt_query": "John Prine In Spite of Ourselves audio",       "genius_query": "In Spite of Ourselves John Prine"},
            "JP_10": {"title": "Summer's End",           "yt_query": "John Prine Summers End audio",                 "genius_query": "Summers End John Prine"},
            "JP_11": {"title": "When I Get to Heaven",   "yt_query": "John Prine When I Get to Heaven audio",        "genius_query": "When I Get to Heaven John Prine"},
            "JP_12": {"title": "Far From Me",            "yt_query": "John Prine Far From Me audio",                 "genius_query": "Far From Me John Prine"},
        },
    },

    "GW": {
        "name":        "Gillian Welch",
        "genre":       "folk",
        "orientation": "narrative",
        "folder":      "GW",
        "songs": {
            "GW_01": {"title": "Everything Is Free",     "yt_query": "Gillian Welch Everything Is Free audio",      "genius_query": "Everything Is Free Gillian Welch"},
            "GW_02": {"title": "Time (The Revelator)",   "yt_query": "Gillian Welch Time The Revelator audio",      "genius_query": "Time The Revelator Gillian Welch"},
            "GW_03": {"title": "Elvis Presley Blues",    "yt_query": "Gillian Welch Elvis Presley Blues audio",      "genius_query": "Elvis Presley Blues Gillian Welch"},
            "GW_04": {"title": "Orphan Girl",            "yt_query": "Gillian Welch Orphan Girl audio",             "genius_query": "Orphan Girl Gillian Welch"},
            "GW_05": {"title": "Acony Bell",             "yt_query": "Gillian Welch Acony Bell audio",              "genius_query": "Acony Bell Gillian Welch"},
            "GW_06": {"title": "Look at Miss Ohio",      "yt_query": "Gillian Welch Look at Miss Ohio audio",       "genius_query": "Look at Miss Ohio Gillian Welch"},
            "GW_07": {"title": "One More Dollar",        "yt_query": "Gillian Welch One More Dollar audio",         "genius_query": "One More Dollar Gillian Welch"},
            "GW_08": {"title": "By the Mark",            "yt_query": "Gillian Welch By the Mark audio",             "genius_query": "By the Mark Gillian Welch"},
            "GW_09": {"title": "Revelator",              "yt_query": "Gillian Welch Revelator audio",               "genius_query": "Revelator Gillian Welch"},
            "GW_10": {"title": "I Dream a Highway",      "yt_query": "Gillian Welch I Dream a Highway audio",       "genius_query": "I Dream a Highway Gillian Welch"},
            "GW_11": {"title": "Dear Someone",           "yt_query": "Gillian Welch Dear Someone audio",            "genius_query": "Dear Someone Gillian Welch"},
            "GW_12": {"title": "Rock of Ages",           "yt_query": "Gillian Welch Rock of Ages audio",            "genius_query": "Rock of Ages Gillian Welch"},
        },
    },

    "CW": {
        "name":        "Colter Wall",
        "genre":       "country",
        "orientation": "narrative",
        "folder":      "CW",
        "songs": {
            "CW_01": {"title": "Sleeping on the Blacktop",  "yt_query": "Colter Wall Sleeping on the Blacktop audio",   "genius_query": "Sleeping on the Blacktop Colter Wall"},
            "CW_02": {"title": "Motorcycle",                "yt_query": "Colter Wall Motorcycle audio",                  "genius_query": "Motorcycle Colter Wall"},
            "CW_03": {"title": "Imaginary Appalachia",      "yt_query": "Colter Wall Imaginary Appalachia audio",        "genius_query": "Imaginary Appalachia Colter Wall"},
            "CW_04": {"title": "The Devil Wears a Suit and Tie","yt_query": "Colter Wall Devil Wears a Suit and Tie audio","genius_query": "The Devil Wears a Suit and Tie Colter Wall"},
            "CW_05": {"title": "Brands",                    "yt_query": "Colter Wall Brands audio",                     "genius_query": "Brands Colter Wall"},
            "CW_06": {"title": "Wild Dogs",                 "yt_query": "Colter Wall Wild Dogs audio",                  "genius_query": "Wild Dogs Colter Wall"},
            "CW_07": {"title": "John Beyers",               "yt_query": "Colter Wall John Beyers audio",                "genius_query": "John Beyers Colter Wall"},
            "CW_08": {"title": "Western Swing & Slow Country","yt_query": "Colter Wall Western Swing Slow Country audio","genius_query": "Western Swing Slow Country Colter Wall"},
            "CW_09": {"title": "Plain to See Plainsman",    "yt_query": "Colter Wall Plain to See Plainsman audio",     "genius_query": "Plain to See Plainsman Colter Wall"},
            "CW_10": {"title": "Tying Knots",               "yt_query": "Colter Wall Tying Knots audio",                "genius_query": "Tying Knots Colter Wall"},
            "CW_11": {"title": "I'm So Lonesome I Could Cry","yt_query": "Colter Wall Im So Lonesome I Could Cry audio","genius_query": "Im So Lonesome I Could Cry Colter Wall"},
            "CW_12": {"title": "Song of the Plains",        "yt_query": "Colter Wall Song of the Plains audio",         "genius_query": "Song of the Plains Colter Wall"},
        },
    },

    # ══════════════════════════════════════════════
    #  PRODUCTION-FORWARD — POP
    # ══════════════════════════════════════════════

    "TS": {
        "name":        "Taylor Swift",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "TS",
        "songs": {
            "TS_01": {"title": "Anti-Hero",                         "yt_query": "Taylor Swift Anti-Hero official audio",                  "genius_query": "Anti-Hero Taylor Swift"},
            "TS_02": {"title": "Shake It Off",                      "yt_query": "Taylor Swift Shake It Off official audio",               "genius_query": "Shake It Off Taylor Swift"},
            "TS_03": {"title": "Love Story (Taylor's Version)",     "yt_query": "Taylor Swift Love Story Taylors Version audio",          "genius_query": "Love Story Taylor Swift"},
            "TS_04": {"title": "Blank Space",                       "yt_query": "Taylor Swift Blank Space official audio",                "genius_query": "Blank Space Taylor Swift"},
            "TS_05": {"title": "Style",                             "yt_query": "Taylor Swift Style official audio",                     "genius_query": "Style Taylor Swift"},
            "TS_06": {"title": "Delicate",                          "yt_query": "Taylor Swift Delicate official audio",                  "genius_query": "Delicate Taylor Swift"},
            "TS_07": {"title": "cardigan",                          "yt_query": "Taylor Swift cardigan official audio",                  "genius_query": "cardigan Taylor Swift"},
            "TS_08": {"title": "august",                            "yt_query": "Taylor Swift august official audio",                    "genius_query": "august Taylor Swift"},
            "TS_09": {"title": "Lover",                             "yt_query": "Taylor Swift Lover official audio",                     "genius_query": "Lover Taylor Swift"},
            "TS_10": {"title": "All Too Well (10 Minute Version)",  "yt_query": "Taylor Swift All Too Well 10 Minute Version audio",     "genius_query": "All Too Well Taylor Swift"},
            "TS_11": {"title": "22",                                "yt_query": "Taylor Swift 22 official audio",                       "genius_query": "22 Taylor Swift"},
            "TS_12": {"title": "Wildest Dreams",                    "yt_query": "Taylor Swift Wildest Dreams official audio",            "genius_query": "Wildest Dreams Taylor Swift"},
            "TS_13": {"title": "Cruel Summer",                      "yt_query": "Taylor Swift Cruel Summer official audio",             "genius_query": "Cruel Summer Taylor Swift"},
            "TS_14": {"title": "The 1",                             "yt_query": "Taylor Swift The 1 folklore audio",                    "genius_query": "The 1 Taylor Swift"},
            "TS_15": {"title": "Don't Blame Me",                    "yt_query": "Taylor Swift Dont Blame Me official audio",            "genius_query": "Dont Blame Me Taylor Swift"},
        },
    },

    "TW": {
        "name":        "The Weeknd",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "TW",
        "songs": {
            "TW_01": {"title": "Blinding Lights",   "yt_query": "The Weeknd Blinding Lights official audio",    "genius_query": "Blinding Lights The Weeknd"},
            "TW_02": {"title": "Save Your Tears",   "yt_query": "The Weeknd Save Your Tears official audio",    "genius_query": "Save Your Tears The Weeknd"},
            "TW_03": {"title": "Starboy",           "yt_query": "The Weeknd Starboy Daft Punk official audio",  "genius_query": "Starboy The Weeknd"},
            "TW_04": {"title": "The Hills",         "yt_query": "The Weeknd The Hills official audio",          "genius_query": "The Hills The Weeknd"},
            "TW_05": {"title": "Can't Feel My Face","yt_query": "The Weeknd Cant Feel My Face official audio",  "genius_query": "Cant Feel My Face The Weeknd"},
            "TW_06": {"title": "Often",             "yt_query": "The Weeknd Often official audio",              "genius_query": "Often The Weeknd"},
            "TW_07": {"title": "Earned It",         "yt_query": "The Weeknd Earned It official audio",          "genius_query": "Earned It The Weeknd"},
            "TW_08": {"title": "Die For You",       "yt_query": "The Weeknd Die For You official audio",        "genius_query": "Die For You The Weeknd"},
            "TW_09": {"title": "Heartless",         "yt_query": "The Weeknd Heartless official audio",          "genius_query": "Heartless The Weeknd"},
            "TW_10": {"title": "Take My Breath",    "yt_query": "The Weeknd Take My Breath official audio",     "genius_query": "Take My Breath The Weeknd"},
            "TW_11": {"title": "Call Out My Name",  "yt_query": "The Weeknd Call Out My Name official audio",   "genius_query": "Call Out My Name The Weeknd"},
            "TW_12": {"title": "After Hours",       "yt_query": "The Weeknd After Hours official audio",        "genius_query": "After Hours The Weeknd"},
            "TW_13": {"title": "In Your Eyes",      "yt_query": "The Weeknd In Your Eyes official audio",       "genius_query": "In Your Eyes The Weeknd"},
            "TW_14": {"title": "Sacrifice",         "yt_query": "The Weeknd Sacrifice official audio",          "genius_query": "Sacrifice The Weeknd"},
            "TW_15": {"title": "Moth to a Flame",   "yt_query": "Swedish House Mafia Weeknd Moth to a Flame audio", "genius_query": "Moth to a Flame The Weeknd"},
        },
    },

    "DL": {
        "name":        "Dua Lipa",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "DL",
        "songs": {
            "DL_01": {"title": "Levitating",         "yt_query": "Dua Lipa Levitating official audio",            "genius_query": "Levitating Dua Lipa"},
            "DL_02": {"title": "Don't Start Now",    "yt_query": "Dua Lipa Dont Start Now official audio",        "genius_query": "Dont Start Now Dua Lipa"},
            "DL_03": {"title": "Physical",           "yt_query": "Dua Lipa Physical official audio",              "genius_query": "Physical Dua Lipa"},
            "DL_04": {"title": "Break My Heart",     "yt_query": "Dua Lipa Break My Heart official audio",        "genius_query": "Break My Heart Dua Lipa"},
            "DL_05": {"title": "One Kiss",           "yt_query": "Calvin Harris Dua Lipa One Kiss audio",         "genius_query": "One Kiss Dua Lipa"},
            "DL_06": {"title": "New Rules",          "yt_query": "Dua Lipa New Rules official audio",             "genius_query": "New Rules Dua Lipa"},
            "DL_07": {"title": "IDGAF",              "yt_query": "Dua Lipa IDGAF official audio",                 "genius_query": "IDGAF Dua Lipa"},
            "DL_08": {"title": "Hallucinate",        "yt_query": "Dua Lipa Hallucinate official audio",           "genius_query": "Hallucinate Dua Lipa"},
            "DL_09": {"title": "Future Nostalgia",   "yt_query": "Dua Lipa Future Nostalgia official audio",      "genius_query": "Future Nostalgia Dua Lipa"},
            "DL_10": {"title": "Hotter than Hell",   "yt_query": "Dua Lipa Hotter than Hell official audio",      "genius_query": "Hotter than Hell Dua Lipa"},
            "DL_11": {"title": "Be the One",         "yt_query": "Dua Lipa Be the One official audio",            "genius_query": "Be the One Dua Lipa"},
            "DL_12": {"title": "Electricity",        "yt_query": "Silk City Dua Lipa Electricity audio",          "genius_query": "Electricity Dua Lipa"},
            "DL_13": {"title": "Cold Heart",         "yt_query": "Elton John Dua Lipa Cold Heart official audio", "genius_query": "Cold Heart Dua Lipa Elton John"},
            "DL_14": {"title": "Love Again",         "yt_query": "Dua Lipa Love Again official audio",            "genius_query": "Love Again Dua Lipa"},
            "DL_15": {"title": "Prisoner",           "yt_query": "Dua Lipa Miley Cyrus Prisoner official audio",  "genius_query": "Prisoner Dua Lipa Miley Cyrus"},
        },
    },

    "AG": {
        "name":        "Ariana Grande",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "AG",
        "songs": {
            "AG_01": {"title": "thank u, next",         "yt_query": "Ariana Grande thank u next official audio",      "genius_query": "thank u next Ariana Grande"},
            "AG_02": {"title": "7 rings",               "yt_query": "Ariana Grande 7 rings official audio",           "genius_query": "7 rings Ariana Grande"},
            "AG_03": {"title": "positions",             "yt_query": "Ariana Grande positions official audio",         "genius_query": "positions Ariana Grande"},
            "AG_04": {"title": "God is a woman",        "yt_query": "Ariana Grande God is a woman official audio",    "genius_query": "God is a woman Ariana Grande"},
            "AG_05": {"title": "no tears left to cry",  "yt_query": "Ariana Grande no tears left to cry official audio","genius_query": "no tears left to cry Ariana Grande"},
            "AG_06": {"title": "Problem",               "yt_query": "Ariana Grande Problem Iggy Azalea audio",        "genius_query": "Problem Ariana Grande"},
            "AG_07": {"title": "Into You",              "yt_query": "Ariana Grande Into You official audio",          "genius_query": "Into You Ariana Grande"},
            "AG_08": {"title": "Side to Side",          "yt_query": "Ariana Grande Side to Side Nicki Minaj audio",   "genius_query": "Side to Side Ariana Grande"},
            "AG_09": {"title": "Break Free",            "yt_query": "Ariana Grande Break Free Zedd audio",            "genius_query": "Break Free Ariana Grande"},
            "AG_10": {"title": "breathin",              "yt_query": "Ariana Grande breathin official audio",          "genius_query": "breathin Ariana Grande"},
            "AG_11": {"title": "Rain on Me",            "yt_query": "Lady Gaga Ariana Grande Rain on Me audio",       "genius_query": "Rain on Me Lady Gaga Ariana Grande"},
            "AG_12": {"title": "stuck with u",          "yt_query": "Ariana Grande Justin Bieber stuck with u audio", "genius_query": "stuck with u Ariana Grande"},
        },
    },

    "ES": {
        "name":        "Ed Sheeran",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "ES",
        "songs": {
            "ES_01": {"title": "Shape of You",          "yt_query": "Ed Sheeran Shape of You official audio",         "genius_query": "Shape of You Ed Sheeran"},
            "ES_02": {"title": "Perfect",               "yt_query": "Ed Sheeran Perfect official audio",              "genius_query": "Perfect Ed Sheeran"},
            "ES_03": {"title": "Thinking Out Loud",     "yt_query": "Ed Sheeran Thinking Out Loud official audio",    "genius_query": "Thinking Out Loud Ed Sheeran"},
            "ES_04": {"title": "Photograph",            "yt_query": "Ed Sheeran Photograph official audio",           "genius_query": "Photograph Ed Sheeran"},
            "ES_05": {"title": "Castle on the Hill",    "yt_query": "Ed Sheeran Castle on the Hill official audio",   "genius_query": "Castle on the Hill Ed Sheeran"},
            "ES_06": {"title": "Galway Girl",           "yt_query": "Ed Sheeran Galway Girl official audio",          "genius_query": "Galway Girl Ed Sheeran"},
            "ES_07": {"title": "Bad Habits",            "yt_query": "Ed Sheeran Bad Habits official audio",           "genius_query": "Bad Habits Ed Sheeran"},
            "ES_08": {"title": "Shivers",               "yt_query": "Ed Sheeran Shivers official audio",              "genius_query": "Shivers Ed Sheeran"},
            "ES_09": {"title": "Overpass Graffiti",     "yt_query": "Ed Sheeran Overpass Graffiti official audio",    "genius_query": "Overpass Graffiti Ed Sheeran"},
            "ES_10": {"title": "The A Team",            "yt_query": "Ed Sheeran The A Team official audio",           "genius_query": "The A Team Ed Sheeran"},
            "ES_11": {"title": "Lego House",            "yt_query": "Ed Sheeran Lego House official audio",           "genius_query": "Lego House Ed Sheeran"},
            "ES_12": {"title": "Happier",               "yt_query": "Ed Sheeran Happier official audio",              "genius_query": "Happier Ed Sheeran"},
        },
    },

    "BE": {
        "name":        "Billie Eilish",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "BE",
        "songs": {
            "BE_01": {"title": "bad guy",               "yt_query": "Billie Eilish bad guy official audio",           "genius_query": "bad guy Billie Eilish"},
            "BE_02": {"title": "Happier Than Ever",     "yt_query": "Billie Eilish Happier Than Ever official audio",  "genius_query": "Happier Than Ever Billie Eilish"},
            "BE_03": {"title": "lovely",                "yt_query": "Billie Eilish lovely Khalid official audio",     "genius_query": "lovely Billie Eilish Khalid"},
            "BE_04": {"title": "when the party's over", "yt_query": "Billie Eilish when the partys over official audio","genius_query": "when the partys over Billie Eilish"},
            "BE_05": {"title": "ocean eyes",            "yt_query": "Billie Eilish ocean eyes official audio",        "genius_query": "ocean eyes Billie Eilish"},
            "BE_06": {"title": "everything i wanted",   "yt_query": "Billie Eilish everything i wanted official audio","genius_query": "everything i wanted Billie Eilish"},
            "BE_07": {"title": "Therefore I Am",        "yt_query": "Billie Eilish Therefore I Am official audio",    "genius_query": "Therefore I Am Billie Eilish"},
            "BE_08": {"title": "your power",            "yt_query": "Billie Eilish your power official audio",        "genius_query": "your power Billie Eilish"},
            "BE_09": {"title": "NDA",                   "yt_query": "Billie Eilish NDA official audio",               "genius_query": "NDA Billie Eilish"},
            "BE_10": {"title": "idontwannabeyouanymore","yt_query": "Billie Eilish idontwannabeyouanymore audio",     "genius_query": "idontwannabeyouanymore Billie Eilish"},
            "BE_11": {"title": "Getting Older",         "yt_query": "Billie Eilish Getting Older official audio",     "genius_query": "Getting Older Billie Eilish"},
            "BE_12": {"title": "Male Fantasy",          "yt_query": "Billie Eilish Male Fantasy official audio",      "genius_query": "Male Fantasy Billie Eilish"},
        },
    },

    "HS": {
        "name":        "Harry Styles",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "HS",
        "songs": {
            "HS_01": {"title": "As It Was",             "yt_query": "Harry Styles As It Was official audio",          "genius_query": "As It Was Harry Styles"},
            "HS_02": {"title": "Watermelon Sugar",      "yt_query": "Harry Styles Watermelon Sugar official audio",   "genius_query": "Watermelon Sugar Harry Styles"},
            "HS_03": {"title": "Adore You",             "yt_query": "Harry Styles Adore You official audio",          "genius_query": "Adore You Harry Styles"},
            "HS_04": {"title": "Golden",                "yt_query": "Harry Styles Golden official audio",             "genius_query": "Golden Harry Styles"},
            "HS_05": {"title": "Lights Up",             "yt_query": "Harry Styles Lights Up official audio",          "genius_query": "Lights Up Harry Styles"},
            "HS_06": {"title": "Sign of the Times",     "yt_query": "Harry Styles Sign of the Times official audio",  "genius_query": "Sign of the Times Harry Styles"},
            "HS_07": {"title": "Falling",               "yt_query": "Harry Styles Falling official audio",            "genius_query": "Falling Harry Styles"},
            "HS_08": {"title": "Treat People With Kindness","yt_query": "Harry Styles Treat People With Kindness audio","genius_query": "Treat People With Kindness Harry Styles"},
            "HS_09": {"title": "Late Night Talking",    "yt_query": "Harry Styles Late Night Talking official audio",  "genius_query": "Late Night Talking Harry Styles"},
            "HS_10": {"title": "Music for a Sushi Restaurant","yt_query": "Harry Styles Music for a Sushi Restaurant audio","genius_query": "Music for a Sushi Restaurant Harry Styles"},
            "HS_11": {"title": "Matilda",               "yt_query": "Harry Styles Matilda official audio",            "genius_query": "Matilda Harry Styles"},
            "HS_12": {"title": "Cinema",                "yt_query": "Harry Styles Cinema official audio",             "genius_query": "Cinema Harry Styles"},
        },
    },

    "OR": {
        "name":        "Olivia Rodrigo",
        "genre":       "pop",
        "orientation": "production",
        "folder":      "OR",
        "songs": {
            "OR_01": {"title": "drivers license",       "yt_query": "Olivia Rodrigo drivers license official audio",   "genius_query": "drivers license Olivia Rodrigo"},
            "OR_02": {"title": "good 4 u",              "yt_query": "Olivia Rodrigo good 4 u official audio",          "genius_query": "good 4 u Olivia Rodrigo"},
            "OR_03": {"title": "deja vu",               "yt_query": "Olivia Rodrigo deja vu official audio",           "genius_query": "deja vu Olivia Rodrigo"},
            "OR_04": {"title": "brutal",                "yt_query": "Olivia Rodrigo brutal official audio",            "genius_query": "brutal Olivia Rodrigo"},
            "OR_05": {"title": "traitor",               "yt_query": "Olivia Rodrigo traitor official audio",           "genius_query": "traitor Olivia Rodrigo"},
            "OR_06": {"title": "happier",               "yt_query": "Olivia Rodrigo happier official audio",           "genius_query": "happier Olivia Rodrigo"},
            "OR_07": {"title": "vampire",               "yt_query": "Olivia Rodrigo vampire official audio",           "genius_query": "vampire Olivia Rodrigo"},
            "OR_08": {"title": "bad idea right?",       "yt_query": "Olivia Rodrigo bad idea right official audio",    "genius_query": "bad idea right Olivia Rodrigo"},
            "OR_09": {"title": "get him back!",         "yt_query": "Olivia Rodrigo get him back official audio",      "genius_query": "get him back Olivia Rodrigo"},
            "OR_10": {"title": "lacy",                  "yt_query": "Olivia Rodrigo lacy official audio",              "genius_query": "lacy Olivia Rodrigo"},
            "OR_11": {"title": "enough for you",        "yt_query": "Olivia Rodrigo enough for you official audio",    "genius_query": "enough for you Olivia Rodrigo"},
            "OR_12": {"title": "1 step forward 3 steps back","yt_query": "Olivia Rodrigo 1 step forward 3 steps back audio","genius_query": "1 step forward 3 steps back Olivia Rodrigo"},
        },
    },

    # ══════════════════════════════════════════════
    #  PRODUCTION-FORWARD — ELECTRONIC
    # ══════════════════════════════════════════════

    "DP": {
        "name":        "Daft Punk",
        "genre":       "electronic",
        "orientation": "production",
        "folder":      "DP",
        "songs": {
            "DP_01": {"title": "Get Lucky",                    "yt_query": "Daft Punk Get Lucky Pharrell Williams official audio",    "genius_query": "Get Lucky Daft Punk"},
            "DP_02": {"title": "Harder, Better, Faster, Stronger", "yt_query": "Daft Punk Harder Better Faster Stronger audio",     "genius_query": "Harder Better Faster Stronger Daft Punk"},
            "DP_03": {"title": "Around the World",             "yt_query": "Daft Punk Around the World official audio",             "genius_query": "Around the World Daft Punk"},
            "DP_04": {"title": "One More Time",                "yt_query": "Daft Punk One More Time official audio",                "genius_query": "One More Time Daft Punk"},
            "DP_05": {"title": "Instant Crush",               "yt_query": "Daft Punk Instant Crush Julian Casablancas audio",      "genius_query": "Instant Crush Daft Punk"},
            "DP_06": {"title": "Lose Yourself to Dance",       "yt_query": "Daft Punk Lose Yourself to Dance official audio",       "genius_query": "Lose Yourself to Dance Daft Punk"},
            "DP_07": {"title": "Something About Us",           "yt_query": "Daft Punk Something About Us audio",                   "genius_query": "Something About Us Daft Punk"},
            "DP_08": {"title": "Digital Love",                 "yt_query": "Daft Punk Digital Love audio",                         "genius_query": "Digital Love Daft Punk"},
            "DP_09": {"title": "Within",                       "yt_query": "Daft Punk Within audio",                               "genius_query": "Within Daft Punk"},
            "DP_10": {"title": "Give Life Back to Music",      "yt_query": "Daft Punk Give Life Back to Music official audio",     "genius_query": "Give Life Back to Music Daft Punk"},
            "DP_11": {"title": "Fragments of Time",            "yt_query": "Daft Punk Fragments of Time Todd Edwards audio",       "genius_query": "Fragments of Time Daft Punk"},
            "DP_12": {"title": "Human After All",              "yt_query": "Daft Punk Human After All audio",                      "genius_query": "Human After All Daft Punk"},
            "DP_13": {"title": "Technologic",                  "yt_query": "Daft Punk Technologic audio",                          "genius_query": "Technologic Daft Punk"},
            "DP_14": {"title": "Touch",                        "yt_query": "Daft Punk Touch Paul Williams audio",                  "genius_query": "Touch Daft Punk"},
            "DP_15": {"title": "Voyager",                      "yt_query": "Daft Punk Voyager audio",                              "genius_query": "Voyager Daft Punk"},
        },
    },

    "DS": {
        "name":        "Disclosure",
        "genre":       "electronic",
        "orientation": "production",
        "folder":      "DS",
        "songs": {
            "DS_01": {"title": "Latch",                 "yt_query": "Disclosure Latch Sam Smith official audio",              "genius_query": "Latch Disclosure Sam Smith"},
            "DS_02": {"title": "F for You",             "yt_query": "Disclosure F for You official audio",                   "genius_query": "F for You Disclosure"},
            "DS_03": {"title": "You & Me",              "yt_query": "Disclosure You and Me audio",                           "genius_query": "You and Me Disclosure"},
            "DS_04": {"title": "White Noise",           "yt_query": "Disclosure White Noise AlunaGeorge audio",              "genius_query": "White Noise Disclosure"},
            "DS_05": {"title": "Magnets",               "yt_query": "Disclosure Magnets Lorde official audio",               "genius_query": "Magnets Disclosure Lorde"},
            "DS_06": {"title": "Holding On",            "yt_query": "Disclosure Holding On Gregory Porter audio",            "genius_query": "Holding On Disclosure"},
            "DS_07": {"title": "Voices",                "yt_query": "Disclosure Voices Sasha Keable audio",                  "genius_query": "Voices Disclosure"},
            "DS_08": {"title": "Help Me Lose My Mind",  "yt_query": "Disclosure Help Me Lose My Mind London Grammar audio",  "genius_query": "Help Me Lose My Mind Disclosure"},
            "DS_09": {"title": "Omen",                  "yt_query": "Disclosure Omen Sam Smith official audio",              "genius_query": "Omen Disclosure Sam Smith"},
            "DS_10": {"title": "Jaded",                 "yt_query": "Disclosure Jaded official audio",                       "genius_query": "Jaded Disclosure"},
            "DS_11": {"title": "Good Intentions",       "yt_query": "Disclosure Good Intentions official audio",             "genius_query": "Good Intentions Disclosure"},
            "DS_12": {"title": "Ultimatum",             "yt_query": "Disclosure Ultimatum official audio",                   "genius_query": "Ultimatum Disclosure"},
            "DS_13": {"title": "Watch Your Step",       "yt_query": "Disclosure Watch Your Step Kelis audio",                "genius_query": "Watch Your Step Disclosure"},
            "DS_14": {"title": "Moving Mountains",      "yt_query": "Disclosure Moving Mountains official audio",            "genius_query": "Moving Mountains Disclosure"},
            "DS_15": {"title": "Where Angels Fear to Tread", "yt_query": "Disclosure Where Angels Fear to Tread audio",     "genius_query": "Where Angels Fear to Tread Disclosure"},
        },
    },

    "CB": {
        "name":        "Caribou",
        "genre":       "psychedelic-electronic",
        "orientation": "production",
        "folder":      "CB",
        "songs": {
            "CB_01": {"title": "Can't Do Without You",      "yt_query": "Caribou Cant Do Without You audio",           "genius_query": "Cant Do Without You Caribou"},
            "CB_02": {"title": "Sun",                        "yt_query": "Caribou Sun audio",                           "genius_query": "Sun Caribou"},
            "CB_03": {"title": "Odessa",                     "yt_query": "Caribou Odessa audio",                        "genius_query": "Odessa Caribou"},
            "CB_04": {"title": "Leave House",                "yt_query": "Caribou Leave House audio",                   "genius_query": "Leave House Caribou"},
            "CB_05": {"title": "Found Out",                  "yt_query": "Caribou Found Out audio",                     "genius_query": "Found Out Caribou"},
            "CB_06": {"title": "Bowls",                      "yt_query": "Caribou Bowls audio",                         "genius_query": "Bowls Caribou"},
            "CB_07": {"title": "Mars",                       "yt_query": "Caribou Mars audio",                          "genius_query": "Mars Caribou"},
            "CB_08": {"title": "Your Love Will Set You Free","yt_query": "Caribou Your Love Will Set You Free audio",   "genius_query": "Your Love Will Set You Free Caribou"},
            "CB_09": {"title": "Back Home",                  "yt_query": "Caribou Back Home audio",                     "genius_query": "Back Home Caribou"},
            "CB_10": {"title": "Never Come Back",            "yt_query": "Caribou Never Come Back audio",               "genius_query": "Never Come Back Caribou"},
            "CB_11": {"title": "All I Ever Need",            "yt_query": "Caribou All I Ever Need audio",               "genius_query": "All I Ever Need Caribou"},
            "CB_12": {"title": "Silver",                     "yt_query": "Caribou Silver audio",                        "genius_query": "Silver Caribou"},
        },
    },

    "JB": {
        "name":        "James Blake",
        "genre":       "electronic",
        "orientation": "production",
        "folder":      "JB",
        "songs": {
            "JB_01": {"title": "Limit to Your Love",    "yt_query": "James Blake Limit to Your Love audio",           "genius_query": "Limit to Your Love James Blake"},
            "JB_02": {"title": "Retrograde",            "yt_query": "James Blake Retrograde audio",                   "genius_query": "Retrograde James Blake"},
            "JB_03": {"title": "The Wilhelm Scream",    "yt_query": "James Blake The Wilhelm Scream audio",           "genius_query": "The Wilhelm Scream James Blake"},
            "JB_04": {"title": "Overgrown",             "yt_query": "James Blake Overgrown audio",                   "genius_query": "Overgrown James Blake"},
            "JB_05": {"title": "Life Round Here",       "yt_query": "James Blake Life Round Here audio",              "genius_query": "Life Round Here James Blake"},
            "JB_06": {"title": "I Need a Forest Fire",  "yt_query": "James Blake I Need a Forest Fire Bon Iver audio","genius_query": "I Need a Forest Fire James Blake"},
            "JB_07": {"title": "Timeless",              "yt_query": "James Blake Timeless audio",                    "genius_query": "Timeless James Blake"},
            "JB_08": {"title": "Are You Even Real?",    "yt_query": "James Blake Are You Even Real audio",           "genius_query": "Are You Even Real James Blake"},
            "JB_09": {"title": "Famous Last Words",     "yt_query": "James Blake Famous Last Words audio",           "genius_query": "Famous Last Words James Blake"},
            "JB_10": {"title": "Assume Form",           "yt_query": "James Blake Assume Form audio",                 "genius_query": "Assume Form James Blake"},
            "JB_11": {"title": "Can't Believe the Way We Flow","yt_query": "James Blake Cant Believe the Way We Flow audio","genius_query": "Cant Believe the Way We Flow James Blake"},
            "JB_12": {"title": "Mile High",             "yt_query": "James Blake Mile High Travis Scott Metro Boomin audio","genius_query": "Mile High James Blake"},
        },
    },

    "LCD": {
        "name":        "LCD Soundsystem",
        "genre":       "electronic",
        "orientation": "production",
        "folder":      "LCD",
        "songs": {
            "LCD_01": {"title": "All My Friends",        "yt_query": "LCD Soundsystem All My Friends audio",           "genius_query": "All My Friends LCD Soundsystem"},
            "LCD_02": {"title": "Daft Punk Is Playing at My House","yt_query": "LCD Soundsystem Daft Punk Is Playing at My House audio","genius_query": "Daft Punk Is Playing at My House LCD Soundsystem"},
            "LCD_03": {"title": "I Can Change",          "yt_query": "LCD Soundsystem I Can Change audio",             "genius_query": "I Can Change LCD Soundsystem"},
            "LCD_04": {"title": "Drunk Girls",           "yt_query": "LCD Soundsystem Drunk Girls audio",              "genius_query": "Drunk Girls LCD Soundsystem"},
            "LCD_05": {"title": "Someone Great",         "yt_query": "LCD Soundsystem Someone Great audio",            "genius_query": "Someone Great LCD Soundsystem"},
            "LCD_06": {"title": "North American Scum",   "yt_query": "LCD Soundsystem North American Scum audio",      "genius_query": "North American Scum LCD Soundsystem"},
            "LCD_07": {"title": "Call the Police",       "yt_query": "LCD Soundsystem Call the Police audio",          "genius_query": "Call the Police LCD Soundsystem"},
            "LCD_08": {"title": "American Dream",        "yt_query": "LCD Soundsystem American Dream audio",           "genius_query": "American Dream LCD Soundsystem"},
            "LCD_09": {"title": "Oh Baby",               "yt_query": "LCD Soundsystem Oh Baby audio",                  "genius_query": "Oh Baby LCD Soundsystem"},
            "LCD_10": {"title": "Tonite",                "yt_query": "LCD Soundsystem Tonite audio",                   "genius_query": "Tonite LCD Soundsystem"},
            "LCD_11": {"title": "Change Yr Mind",        "yt_query": "LCD Soundsystem Change Yr Mind audio",           "genius_query": "Change Yr Mind LCD Soundsystem"},
            "LCD_12": {"title": "How Do You Sleep?",     "yt_query": "LCD Soundsystem How Do You Sleep audio",         "genius_query": "How Do You Sleep LCD Soundsystem"},
        },
    },

    "BN": {
        "name":        "Bonobo",
        "genre":       "electronic",
        "orientation": "production",
        "folder":      "BN",
        "songs": {
            "BN_01": {"title": "Kiara",                 "yt_query": "Bonobo Kiara audio",                             "genius_query": "Kiara Bonobo"},
            "BN_02": {"title": "No Reason",             "yt_query": "Bonobo No Reason audio",                        "genius_query": "No Reason Bonobo"},
            "BN_03": {"title": "Outlier",               "yt_query": "Bonobo Outlier audio",                          "genius_query": "Outlier Bonobo"},
            "BN_04": {"title": "Break Apart",           "yt_query": "Bonobo Break Apart Rhye audio",                 "genius_query": "Break Apart Bonobo"},
            "BN_05": {"title": "Bambro Koyo Ganda",     "yt_query": "Bonobo Bambro Koyo Ganda audio",               "genius_query": "Bambro Koyo Ganda Bonobo"},
            "BN_06": {"title": "Surface",               "yt_query": "Bonobo Surface audio",                          "genius_query": "Surface Bonobo"},
            "BN_07": {"title": "Linked",                "yt_query": "Bonobo Linked audio",                           "genius_query": "Linked Bonobo"},
            "BN_08": {"title": "It Came From the Sea",  "yt_query": "Bonobo It Came From the Sea audio",             "genius_query": "It Came From the Sea Bonobo"},
            "BN_09": {"title": "Kerala",                "yt_query": "Bonobo Kerala audio",                           "genius_query": "Kerala Bonobo"},
            "BN_10": {"title": "Towers",                "yt_query": "Bonobo Towers audio",                           "genius_query": "Towers Bonobo"},
            "BN_11": {"title": "Prelude",               "yt_query": "Bonobo Prelude audio",                          "genius_query": "Prelude Bonobo"},
            "BN_12": {"title": "Sapphire",              "yt_query": "Bonobo Sapphire audio",                         "genius_query": "Sapphire Bonobo"},
        },
    },

    "GZ": {
        "name":        "Gorillaz",
        "genre":       "psychedelic-electronic",
        "orientation": "production",
        "folder":      "GZ",
        "songs": {
            "GZ_01": {"title": "Feel Good Inc.",        "yt_query": "Gorillaz Feel Good Inc official audio",          "genius_query": "Feel Good Inc Gorillaz"},
            "GZ_02": {"title": "DARE",                  "yt_query": "Gorillaz DARE official audio",                  "genius_query": "DARE Gorillaz"},
            "GZ_03": {"title": "Clint Eastwood",        "yt_query": "Gorillaz Clint Eastwood official audio",        "genius_query": "Clint Eastwood Gorillaz"},
            "GZ_04": {"title": "On Melancholy Hill",    "yt_query": "Gorillaz On Melancholy Hill official audio",    "genius_query": "On Melancholy Hill Gorillaz"},
            "GZ_05": {"title": "Rhinestone Eyes",       "yt_query": "Gorillaz Rhinestone Eyes official audio",       "genius_query": "Rhinestone Eyes Gorillaz"},
            "GZ_06": {"title": "Stylo",                 "yt_query": "Gorillaz Stylo Bobby Womack Mos Def audio",     "genius_query": "Stylo Gorillaz"},
            "GZ_07": {"title": "Empire Ants",           "yt_query": "Gorillaz Empire Ants Little Dragon audio",      "genius_query": "Empire Ants Gorillaz"},
            "GZ_08": {"title": "Momentz",               "yt_query": "Gorillaz Momentz De La Soul audio",             "genius_query": "Momentz Gorillaz"},
            "GZ_09": {"title": "Andromeda",             "yt_query": "Gorillaz Andromeda D.R.A.M audio",              "genius_query": "Andromeda Gorillaz"},
            "GZ_10": {"title": "Tranz",                 "yt_query": "Gorillaz Tranz official audio",                 "genius_query": "Tranz Gorillaz"},
            "GZ_11": {"title": "Hollywood",             "yt_query": "Gorillaz Hollywood Snoop Dogg Jamie Principle audio","genius_query": "Hollywood Gorillaz"},
            "GZ_12": {"title": "New Gold",              "yt_query": "Gorillaz New Gold Tame Impala Bootie Brown audio","genius_query": "New Gold Gorillaz"},
        },
    },
}

# ── Convenience helpers ───────────────────────────────────────────────────────

def get_all_song_ids():
    """Return list of all (artist_code, song_id) tuples across the full catalog."""
    return [
        (artist_code, song_id)
        for artist_code, artist_data in CATALOG.items()
        for song_id in artist_data["songs"]
    ]


def get_song_metadata(song_id: str) -> dict:
    """Return flat metadata dict for a given song_id."""
    artist_code = song_id.split("_")[0]
    artist_data = CATALOG[artist_code]
    song_data   = artist_data["songs"][song_id]
    return {
        "song_id":       song_id,
        "artist_code":   artist_code,
        "artist_name":   artist_data["name"],
        "genre":         artist_data["genre"],
        "orientation":   artist_data["orientation"],
        "folder":        artist_data["folder"],
        "title":         song_data["title"],
        "yt_query":      song_data["yt_query"],
        "genius_query":  song_data["genius_query"],
    }


def total_songs() -> int:
    return sum(len(a["songs"]) for a in CATALOG.values())


if __name__ == "__main__":
    print(f"Catalog: {len(CATALOG)} artists, {total_songs()} songs")
    for code, artist in CATALOG.items():
        print(f"  {code:4s} | {artist['name']:<25} | {artist['genre']:<25} | {artist['orientation']}")
