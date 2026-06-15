#!/usr/bin/env bash
# run_pipeline.sh — Full Musical Congruence pipeline runner
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                   # run all steps
#   ./run_pipeline.sh --from 4          # resume from step 4
#   ./run_pipeline.sh --steps 1 2       # run only steps 1 and 2
#   ./run_pipeline.sh --artists KL TS   # restrict scraping to specific artists
#   ./run_pipeline.sh --device cpu      # force CPU for all embedding steps
#
# Steps:
#   1  scrape_lyrics      Genius API → lyrics JSON
#   2  scrape_audio       yt-dlp → MP3 files
#   3  build_schema       Auto-generate schema .py files
#   4  embed_mulan        MuQ-MuLan embeddings
#   5  embed_clap         LAION-CLAP-Music embeddings
#   6  embed_mert_sbert   MERT + Sentence-BERT late fusion
#   7  segment_analysis   Segment-level LMC (nice-to-have)
#   8  get_popularity     Spotify popularity + audio features
#   9  combine            Merge everything → master_results.csv

set -euo pipefail

# ─── Parse arguments ─────────────────────────────────────────────────────────
FROM_STEP=1
STEPS=()
ARTISTS=()
DEVICE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)     FROM_STEP=$2;  shift 2 ;;
        --steps)    shift
                    while [[ $# -gt 0 && $1 != --* ]]; do
                        STEPS+=("$1"); shift
                    done ;;
        --artists)  shift
                    while [[ $# -gt 0 && $1 != --* ]]; do
                        ARTISTS+=("$1"); shift
                    done ;;
        --device)   DEVICE=$2;    shift 2 ;;
        --dry-run)  DRY_RUN=true; shift   ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Build artist flag string
ARTIST_FLAG=""
if [[ ${#ARTISTS[@]} -gt 0 ]]; then
    ARTIST_FLAG="--artists ${ARTISTS[*]}"
fi

DEVICE_FLAG=""
if [[ -n "$DEVICE" ]]; then
    DEVICE_FLAG="--device $DEVICE"
fi

DRY_FLAG=""
if $DRY_RUN; then
    DRY_FLAG="--dry-run"
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

log()     { echo -e "${BLUE}[$(date +%H:%M:%S)]${RESET} $*"; }
success() { echo -e "${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}⚠${RESET}  $*"; }
error()   { echo -e "${RED}✗${RESET}  $*"; }

should_run() {
    local step=$1
    if [[ ${#STEPS[@]} -gt 0 ]]; then
        for s in "${STEPS[@]}"; do [[ "$s" == "$step" ]] && return 0; done
        return 1
    fi
    [[ $step -ge $FROM_STEP ]]
}

check_env_var() {
    local var=$1
    if [[ -z "${!var:-}" || "${!var}" == *"YOUR_"* ]]; then
        warn "$var is not set — step may fail."
    else
        success "$var is set"
    fi
}

# ─── Pre-flight checks ────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Musical Congruence Pipeline                 ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""

log "Checking environment…"
check_env_var GENIUS_API_TOKEN
check_env_var SPOTIFY_CLIENT_ID
check_env_var SPOTIFY_CLIENT_SECRET

# Check Python dependencies
python3 -c "import lyricsgenius" 2>/dev/null && success "lyricsgenius" || warn "lyricsgenius not installed (pip install lyricsgenius)"
python3 -c "import yt_dlp"        2>/dev/null && success "yt-dlp"        || warn "yt-dlp not installed (pip install yt-dlp)"
python3 -c "import transformers"   2>/dev/null && success "transformers"  || warn "transformers not installed"
python3 -c "import sentence_transformers" 2>/dev/null && success "sentence-transformers" || warn "sentence-transformers not installed"
python3 -c "import spotipy"        2>/dev/null && success "spotipy"       || warn "spotipy not installed (pip install spotipy)"
python3 -c "import muq"            2>/dev/null && success "muq"           || warn "MuQ not installed"

command -v ffmpeg &>/dev/null && success "ffmpeg" || warn "ffmpeg not installed (brew install ffmpeg)"

echo ""
log "Catalog summary:"
python3 -c "
import sys; sys.path.insert(0,'.')
from config import CATALOG, total_songs
print(f'  {len(CATALOG)} artists  |  {total_songs()} songs')
"

echo ""
log "Starting pipeline  (from step $FROM_STEP)…"
echo ""

# ─── Step 1: Scrape lyrics ────────────────────────────────────────────────────
if should_run 1; then
    echo -e "\n${BOLD}── Step 1: Scrape lyrics ────────────────────────────${RESET}"
    python3 01_scrape_lyrics.py $ARTIST_FLAG $DRY_FLAG
    success "Step 1 done"
fi

# ─── Step 2: Scrape audio ────────────────────────────────────────────────────
if should_run 2; then
    echo -e "\n${BOLD}── Step 2: Scrape audio ─────────────────────────────${RESET}"
    warn "This step downloads audio from YouTube. Expect ~3–6 min per artist."
    python3 02_scrape_audio.py $ARTIST_FLAG $DRY_FLAG
    success "Step 2 done"
fi

# ─── Step 3: Build schemas ───────────────────────────────────────────────────
if should_run 3; then
    echo -e "\n${BOLD}── Step 3: Build schemas ────────────────────────────${RESET}"
    python3 03_build_schema.py $ARTIST_FLAG
    success "Step 3 done"
fi

# ─── Step 4: MuQ-MuLan embeddings ───────────────────────────────────────────
if should_run 4; then
    echo -e "\n${BOLD}── Step 4: MuQ-MuLan embeddings ─────────────────────${RESET}"
    warn "First run will download the MuQ-MuLan model (~1.5 GB)."
    python3 04_embed_mulan.py $DEVICE_FLAG
    success "Step 4 done"
fi

# ─── Step 5: CLAP embeddings ─────────────────────────────────────────────────
if should_run 5; then
    echo -e "\n${BOLD}── Step 5: LAION-CLAP-Music embeddings ──────────────${RESET}"
    warn "First run will download CLAP model (~600 MB)."
    warn "If CLAP is slow on MPS, add --device cpu"
    python3 05_embed_clap.py $DEVICE_FLAG
    success "Step 5 done"
fi

# ─── Step 6: MERT + SBERT embeddings ────────────────────────────────────────
if should_run 6; then
    echo -e "\n${BOLD}── Step 6: MERT + Sentence-BERT embeddings ──────────${RESET}"
    warn "First run will download MERT (~380 MB) and SBERT (~420 MB)."
    python3 06_embed_mert_sbert.py $DEVICE_FLAG
    success "Step 6 done"
fi

# ─── Step 7: Segment analysis ────────────────────────────────────────────────
if should_run 7; then
    echo -e "\n${BOLD}── Step 7: Segment-level analysis (optional) ────────${RESET}"
    warn "This step is slow — 2–4 min per song. Skip with --from 8 if not needed."
    python3 07_segment_analysis.py $DEVICE_FLAG $ARTIST_FLAG
    success "Step 7 done"
fi

# ─── Step 8: Spotify popularity ──────────────────────────────────────────────
if should_run 8; then
    echo -e "\n${BOLD}── Step 8: Spotify popularity ───────────────────────${RESET}"
    python3 08_get_popularity.py
    success "Step 8 done"
fi

# ─── Step 9: Combine results ─────────────────────────────────────────────────
if should_run 9; then
    echo -e "\n${BOLD}── Step 9: Combine results → master_results.csv ─────${RESET}"
    python3 09_combine_results.py
    success "Step 9 done"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Pipeline complete!                          ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""
echo "Next step: open analysis/analysis.R in RStudio"
echo "Primary output: results/master_results.csv"
echo ""
