# Measuring Lyric–Music Congruence Beyond MuLan
### A critical literature review and a ranked menu of alternatives

*Research memo, 2026-07-14. Companion to the observational arm (`src/lmc`) and the
generative arm (`src/lmcgen`). Purpose: identify academically defensible, fully
automatable ways to quantify lyric–music congruence that do **not** depend on
MuLan's (or CLAP's) caption-trained text tower.*

> **Bibliographic note.** arXiv IDs / venues below were checked during the
> literature pass, but a handful of canonical citations (MuLan, CLAP, MERT,
> CKA, RSA, Palmer & Kelly) are cited from established knowledge and should be
> confirmed against the primary source before they go into the thesis. This is
> normal due diligence, not a flag of doubt.

---

## 0. TL;DR

Your instinct is correct, and it is *stronger* than "MuLan is imperfect." The
problem is a **construct–instrument mismatch**, not a tuning issue:

- MuLan and CLAP are **CLIP-for-audio**: their text encoders were trained on
  *music captions / tags* ("upbeat acoustic folk, male vocal"), so **raw lyrics
  are out-of-distribution** for the text tower. You are asking a caption model to
  place *"the door was open, I walked through / your name still hanging in the
  room"* in music-description space — a task it was never trained for.
- Reference-free cosine metrics of this exact form (CLIPScore) are **known to be
  insensitive to word order, negation, and compositional structure** (Hessel et
  al., 2021) — precisely the linguistic content lyrics carry.
- A raw cosine also **entangles the congruence signal with the modality gap** —
  audio and text embeddings sit in separate cones (Liang et al., 2022), so part
  of your number is geometry, not agreement.

So a single cosine in a caption space is **under-identified** for your construct.
Below are **four families** of alternatives, each ranked internally by strength of
theory, all automatable over the whole corpus, all validatable against your
generative-arm human survey. The recommended stack (Part 5):

1. **Flagship** — train a small **bespoke lyric↔audio contrastive model on your
   own LRCLIB matched pairs** ("LyricLMC"). This is the single highest-leverage
   move you can make and it is *feasible on your hardware*, because your audio
   tower (MERT) is already built and you have ~19.5 M matched pairs for free.
2. **Interpretable backbone** — decompose each modality into valence/arousal (+ a
   semantic axis), compare there. Directly grounded in music psychology.
3. **Holistic-geometry test** — second-order representational-similarity
   (RSA/CKA/Gromov–Wasserstein) between the audio and lyric embedding *geometries*.
   This is the approach most faithful to your "congruence lives in the embedding
   manifold" thesis.
4. **Corroboration on a subset** — generative-likelihood (Jukebox/PMI) and an
   audio-LLM judge.

Validate the battery with **matched-vs-mismatched retrieval** (automatic, whole
corpus) as the objective anchor and the **human survey** as the gold standard;
do **not** validate the *metric* on popularity (that is your downstream question,
and using it to pick the metric is circular / collider-prone).

---

## 1. Diagnosis — why MuLan/CLAP is the wrong instrument for *lyrics*

### 1.1 The captioning-CLIP paradigm
MuLan (Huang et al., ISMIR 2022), LAION-CLAP (Wu et al., ICASSP 2023) and CLAP
(Elizalde et al., ICASSP 2023) are dual-encoder contrastive models trained to pull
*audio* and its *natural-language description* together. The text distribution is
captions, tags, and metadata. Lyrics are a **different register** (narrative,
figurative, first-person, repetitive) that the text tower never saw at scale.
Feeding lyrics in is domain shift on the exact side of the model you depend on.

Your own README already records this as the project-wide KNOWN ISSUE; the
literature makes it a *nameable, citable* failure mode rather than a hunch.

### 1.2 CLIPScore-style metrics are structurally blind to what lyrics encode
Your LMC is `cos(audio_emb, lyric_text_emb)` — this is **CLIPScore** (Hessel et
al., EMNLP 2021, arXiv:2104.08718) with audio swapped for image. CLIPScore is a
strong caption metric, but its documented weaknesses are:
- **insensitive to word order and negation**,
- **compositionally insensitive** (right concepts, wrong configuration still
  scores high).

Lyrics live in exactly those dimensions. So the metric's known blind spots line up
one-to-one with your signal of interest. This is a citable argument, not just an
observed limitation.

### 1.3 The modality gap contaminates the cosine
In contrastive dual encoders, the two modalities occupy **separate cones** with a
persistent gap (Liang et al., "Mind the Gap," NeurIPS 2022, arXiv:2203.02053; see
also "It's Not a Modality Gap," arXiv:2405.18570; COMET for audio-text). A raw
audio↔text cosine therefore mixes *congruence* with *where the gap sits*. Minimal
mitigation if you keep a joint space at all: **mean-center audio and text
embeddings separately (per corpus) before cosine**, and/or whiten — this removes
the shared offset and typically improves human-correlation. Cheap, do it
regardless of which family you adopt.

### 1.4 The "semantic–perceived emotion gap" in lyrics
A 2026 critical review (Sci. of Comp. Music / *ScienceDirect*
S156625352600182X, "Revisiting the role of lyrics in MER") formalizes a gap
between the *semantic* emotion of lyric text and the *perceived* emotion of the
sung result. Any lyric-text-only representation (including MuLan's text tower)
inherits this gap. It is another reason to prefer methods that either (a) model the
*sung* audio jointly, or (b) triangulate multiple views.

**Bottom line:** the fix belongs at the construct level. Keep going.

---

## 2. Reframing the construct so it is measurable

Three findings from music psychology should shape *any* metric you build:

1. **Melody dominates, and the lyric effect is asymmetric.** Ali & Peynircioğlu
   (2006, *Psychology of Music*): lyrics *detract* from emotion in happy/calm
   (positive) songs but *enhance* it in sad/angry (negative) songs; melody is the
   stronger partner throughout. Replicated with qualifications by Ma, Baker,
   Vukovics, Davis & Elliott (2024, *Musicae Scientiae*,
   doi:10.1177/10298649221149109). **Implication:** congruence is not a symmetric
   distance; a good metric may need to be *sign-aware* (agreement matters more, or
   differently, on the negative-valence side).

2. **Modality–dimension asymmetry.** Across the lyrics-MER literature (Hu &
   Downie, 2010; Malheiro et al.; Delbouys et al., 2018) **lyrics carry *valence*
   well but *arousal* poorly**, while **audio carries *arousal* strongly**.
   **Implication:** a naïve "compare full VA of lyrics vs full VA of audio" is
   confounded — you would be comparing two channels on an axis one of them can't
   measure. Compare *valence* primarily from lyrics, *arousal* primarily from
   audio, and treat their *joint* configuration as the object.

3. **Congruence is multi-level but bound.** It is at once affective, semantic, and
   structural — your thesis is that these are bound into one high-dimensional
   representation. No single scalar captures that; a **battery** whose components
   *agree* is the empirical way to defend the "bound holistic representation"
   claim (convergent validity; Campbell & Fiske, 1959, multitrait–multimethod).

---

## 3. The four families of alternatives (ranked within each by strength of theory)

Notation: for a song, `A` = audio, `L` = lyric text. Corpus of `N` songs.

---

### FAMILY A — Better / purpose-built joint embeddings (drop-in replacement for the cosine)

The same shape as now (`cos(f_A(A), f_L(L))`) but with encoders that actually
handle lyrics. Fully automatic; slots into `alignment.py` with minimal change.

**A1. Bespoke contrastive lyric↔audio model trained on YOUR pairs ("LyricLMC") — STRONGEST.**
Train a two-tower model whose text tower ingests *lyrics* and whose audio tower is
MERT (you already extract MERT 1024-d vectors in `mert.py`). Align with InfoNCE on
**matched (audio, lyrics) pairs**, using **mismatched lyrics as negatives**.

- *Why it's the strongest theory–feasibility point:* it removes the exact defect
  (lyrics become **in-distribution** for the text side) while **preserving the
  full high-dimensional joint geometry** you believe congruence lives in. It is
  the MuLan/CLAP/MuSCALL recipe (Manco et al., "Contrastive Audio-Language
  Learning for Music," ISMIR 2022, arXiv:2208.12208) but *pointed at lyrics
  instead of captions*.
- *Data:* you have ~19.5 M synced-lyric tracks; even 20–100 k pairs is plenty for
  projection-head training. You can build unlimited hard negatives (same
  artist/genre, different song) — which directly teaches the model *congruence*
  rather than *topic*.
- *Feasible on an M4 16 GB:* freeze MERT (audio) and a strong text encoder
  (sentence-transformer, or an emotion-tuned RoBERTa), and **train only two small
  projection heads** with InfoNCE. That is minutes-to-hours on CPU/MPS, no GPU
  cluster. A fuller LoRA fine-tune is the cloud-optional upgrade.
- *Prior art proving the pattern works for audio↔lyrics specifically:* Yu et al.,
  "Deep Cross-Modal Correlation Learning for Audio and Lyrics in Music Retrieval,"
  ACM TOMM 2019 (arXiv:1711.08976); "Scalable Music Cover Retrieval Using
  Lyrics-Aligned Audio Embeddings" (arXiv:2601.11262).
- *Validation is built in:* the training objective **is** matched-vs-mismatched, so
  retrieval AUC on held-out songs is a direct, automatic quality check.
- **Theory strength: very high.** This is the recommended flagship.

**A2. CLaMP 3 (off-the-shelf, modern).** Wu (Wood) et al., "CLaMP 3," ACL Findings
2025 (arXiv:2502.10362; code: github.com/sanderwood/clamp3). Aligns audio +
symbolic + sheet music with **multilingual** text in one space, trained on 2.31 M
music-text pairs (M4-RAG). Better and multilingual vs MuLan/CLAP.
- *Caveat:* text is still *descriptions*, so the lyric-OOD problem is reduced (more
  data, multilingual) but **not solved**. Best used as a **stronger baseline** and
  a second joint-space opinion, not as the fix.
- **Theory strength: moderate** (same paradigm, better executed).

**A3. Re-purposed CLAP with corpus centering + prompt-ensembled anchors.** Keep
CLAP but (i) mean-center per modality (§1.3), (ii) for the *affective* target,
don't embed raw lyrics — embed **short affect descriptions** and score by *margin*
between corner anchors rather than raw cosine (this is what your generative arm
already gestures at). Cheapest, weakest.
- **Theory strength: low-moderate** (patches the instrument, doesn't replace it).

*Family A ranking: **A1 ≫ A2 > A3**.*

---

### FAMILY B — Decompose-then-compare in an interpretable shared space

Map each modality *independently* into a small, human-meaningful space (VA, or VA +
categorical emotion, + a semantic-topic vector), then measure agreement there.
Sidesteps the joint-embedding problem entirely; each half is independently
validated by its own literature.

**B1. Valence–Arousal agreement with modality-appropriate channels — STRONGEST in this family.**
- *Audio → VA:* a modern MER model. **Music2Emo** (amaai-lab, HF, 2025: MERT +
  key/chords + knowledge distillation, VA *and* categorical, SOTA on
  MTG-Jamendo/DEAM/PMEmo/EmoMusic); or "Towards Unified Music Emotion Recognition
  across Dimensional and Categorical Models" (2025). Meta-analysis (ACM Comp.
  Surveys 2025) reports best-case audio VA at r≈0.67 valence / r≈0.81 arousal.
- *Lyrics → VA:* transformer emotion regressor (e.g., fine-tuned RoBERTa on EmoBank
  VAD; GoEmotions, Demszky et al. 2020) **and/or** a lexicon backbone — NRC-VAD
  (Mohammad, ACL 2018; 55 k words scored on V/A/D), ANEW, LIWC. Lexicons are
  transparent and reviewer-proof; transformers are stronger. Use both as an
  internal convergence check.
- *Compare, respecting §2.2:* congruence = agreement of the **joint VA
  configuration**, weighting **valence toward the lyric estimate** and **arousal
  toward the audio estimate**, e.g. negative Euclidean distance in VA (as in the
  Deezer Mood Detection Dataset line, Delbouys et al. 2018) or a signed,
  valence-emphasized agreement à la Ali & Peynircioğlu.
- *Prior art:* Hu & Downie (2010); Malheiro et al., "Classification and Regression
  of Music Lyrics: Emotionally-Significant Features" (IEEE TAC); "When Lyrics
  Outperform Audio for Music Mood Classification."
- **Theory strength: high, best interpretability.** Its cost is philosophical: it
  *reduces* the holistic representation to a few axes — against your thesis. So use
  it as the **interpretable backbone/validator**, not the sole metric.

**B2. Categorical-emotion distribution agreement.** Predict an emotion *distribution*
(e.g., GoEmotions 27-way, or a mood taxonomy) per modality; congruence = 1 −
Jensen–Shannon divergence, or cosine of the distributions. Richer than 2-D VA,
still interpretable.
- **Theory strength: moderate-high.**

**B3. Topic/semantic agreement.** Represent lyric *content* with a strong sentence
encoder (SBERT/E5) and audio *content* with auto-tags mapped to text; compare.
Captures the "topical" level. Weakest of the three (audio→topic is lossy).
- **Theory strength: moderate.**

*Family B ranking: **B1 > B2 > B3**. Adopt B1 as your interpretable backbone.*

---

### FAMILY C — Second-order / representational-geometry congruence (the "holistic" route)

Instead of putting `A` and `L` in one space and comparing points, compare the
**geometry** of the audio embedding space with the **geometry** of the lyric
embedding space *across the corpus*. Two songs that are close in audio-space
should be close in lyric-space **iff** lyrics and music co-vary — i.e. congruence
becomes a *relational* property of the manifold. **This is the family most faithful
to your embedding-space thesis**, and it is the most novel contribution potential.

**C1. Representational Similarity Analysis / CKA — STRONGEST theory here.**
Build the `N×N` audio-similarity matrix `S_A` (from MERT or any audio encoder) and
the `N×N` lyric-similarity matrix `S_L` (from a lyric text encoder). Their
alignment **is** corpus-level congruence:
- **RSA** (Kriegeskorte et al., 2008): correlate the off-diagonals of `S_A` and
  `S_L`.
- **CKA** (Kornblith et al., ICML 2019): centered-kernel alignment — **invariant to
  orthogonal transforms and isotropic scaling**, so it compares heterogeneous
  modalities *without needing a shared space and without the modality gap*
  (exactly the contamination in §1.3). Recent work proves RSA ≡ CKA ≡ CCA under
  conditions (bioRxiv 2024.10.23.619871).
- *Global vs per-song:* CKA/RSA are naturally **corpus-level** (one number: "how
  much does music geometry track lyric geometry in this corpus / genre / cell").
  That is itself a *publishable* congruence statistic (e.g., "congruence is higher
  in folk than EDM"). For a **per-song** score, use a **local** variant: for song
  `i`, compare its `k`-NN neighborhood ranking in audio-space vs lyric-space (rank
  correlation / local RSA). Honest caveat: the per-song operationalization is a
  design choice you'd need to define and defend — but that is exactly the kind of
  methodological novelty a thesis wants.
- **Theory strength: very high; best fit to your worldview.**

**C2. Gromov–Wasserstein (GW) alignment.** Alvarez-Melis & Jaakkola,
"Gromov-Wasserstein Alignment of Word Embedding Spaces," EMNLP 2018
(arXiv:1809.00013); OT view of audio-text retrieval (arXiv:2405.10084). GW finds a
soft coupling between two metric spaces using **only intra-space distances** — no
shared coordinates. The GW distance is a global (in)congruence score; the coupling
matrix's diagonal mass gives a **per-song** alignment signal (how much a song maps
to *itself* under the optimal cross-modal transport).
- *Why it's beautiful for you:* it operationalizes "these two geometries say the
  same thing" without ever embedding them together — the cleanest possible answer
  to the modality-gap objection.
- **Theory strength: very high; slightly more exotic / harder to communicate than
  CKA.**

**C3. (Deep) CCA.** Classic CCA / Deep CCA (Andrew et al. 2013; Yu et al. audio-
lyrics DCCA, arXiv:1711.08976): learn maximally-correlated projections; the
**canonical correlations** are a direct, well-understood congruence measure, and
DCCA gives a per-song projected-space cosine. Middle ground between Family A
(learned joint space) and C1 (fixed-geometry comparison).
- **Theory strength: high, and the most "standard-statistics" defensible.**

*Family C ranking: **C1 ≈ C2 > C3** on novelty/fit; **C3 > C1 > C2** on ease of
communication. Include CKA as your holistic-geometry test.*

---

### FAMILY D — Generative / likelihood-based scoring (model the joint distribution directly)

If a model was trained to **generate audio conditioned on lyrics**, then a
likelihood-style score of "how well does this audio go with these lyrics" is a
*proper*, theory-grounded congruence measure — it reads the joint distribution the
model learned, not a geometric proxy.

**D1. Jukebox representations / conditional scoring — STRONGEST theory.**
Jukebox (Dhariwal et al., 2020) is a lyrics-conditioned autoregressive audio model;
its **encoder–decoder lyric attention learns exactly the lyric↔music alignment you
care about** (attention marches through lyric tokens as the music progresses).
- *JukeMIR* (Castellon, Donahue & Liang, ISMIR 2021, arXiv:2107.05677) showed
  Jukebox features beat tagging-pretrained models by ~30% on MIR **including
  emotion** — evidence these representations encode lyric-aware musical meaning.
- *Two uses:* (a) **Jukebox embeddings** as the audio (or joint) tower for Family A
  or C — richer than MERT; (b) a **PMI-style conditional score**
  `log P(A | L) − log P(A)` (raw likelihood over-rewards generic pairings; PMI is
  the fix — this exact correction appears in music-caption alignment work).
- *Feasibility caveat (important given "fully automatic, whole corpus"):* Jukebox
  is **large and slow** — impractical on your M4 for thousands of songs. It is a
  **cloud-batch, corroborate-on-a-subset** tool, not the primary whole-corpus
  metric. Pre-extracting JukeMIR features once and caching is the realistic path.
- **Theory strength: very high; feasibility: heavy.**

**D2. Lyric-conditioned song models as scorers (ACE-Step / DiffRhythm / YuE).**
You already have hands-on experience with these (ACE-Step, DiffRhythm smoke tests).
A flow/diffusion model's reconstruction or denoising loss for *this audio* given
*these lyrics* vs *shuffled lyrics* is a mismatch-sensitive congruence proxy.
Cheaper than Jukebox, but noisier and less established.
- **Theory strength: moderate; exploratory.**

*Family D ranking: **D1 > D2**. Use D1 on a subset to corroborate, not as the
whole-corpus workhorse.*

---

### Cross-cutting: STRUCTURAL / PROSODIC sub-battery (the "temporal" level)

Your line-level ±window design already exploits LRCLIB timestamps; the structural
level has its own clean, cheap, fully-automatic metrics that no embedding captures:

- **Linguistic-stress ↔ metrical-strength alignment** (the "textsetting" / Stress
  Rule literature: Palmer & Kelly 1992; "Linguistic prosody and musical meter in
  song"): get syllable stress from a pronunciation dictionary (CMUdict) /
  g2p, get beat strength from a beat tracker (librosa/`madmom`), measure how often
  stressed syllables land on strong beats.
- **Multimodal Lyrics-Rhythm Matching** (arXiv:2301.02732) and "Relationships
  between Keywords and Strong Beats in Lyrical Music" (arXiv:2412.04202): direct,
  reusable formulations of lyric-to-rhythm alignment; the latter's *keyword ↔
  strong-beat* idea ties structure to semantics.
- **Neuro-grounding** for the construct's reality: Gordon et al., "EEG Correlates of
  Song Prosody," *Frontiers in Psychology* 2011 — linguistic and musical rhythm
  interact in the brain (construct validity, good for the intro).

These are model-independent, run in seconds/song, and give a third **convergent**
view. Include 1–2 of them.

---

### Cross-cutting: AUDIO-LLM-AS-JUDGE (explanatory, not a clean scalar)

Instruction-tuned audio-language models — SALMONN, Qwen-Audio (arXiv:2311.07919),
MU-LLaMA (MERT-based), Music Flamingo (arXiv:2511.10289) — can be *prompted* to
rate and **explain** lyric-music congruence. Benchmarks: MuChoMusic
(arXiv:2408.01337).
- *Use:* qualitative validation, error analysis, and generating human-readable
  rationales for a subset (great for the thesis narrative and for designing survey
  items) — **not** as the primary metric (outputs aren't calibrated scalars, and
  the biggest/most-capable ones are closed or heavy).
- **Theory strength: auxiliary.**

---

## 4. Overall ranking (synthesis across families)

Ranked by **strength of theory × fit to your embedding thesis × automatability**:

| # | Method | Family | Theory | Fits "holistic-embedding" thesis | Whole-corpus auto | Role |
|---|--------|--------|:--:|:--:|:--:|------|
| 1 | **Bespoke lyric↔audio contrastive (LyricLMC)** | A1 | ★★★★★ | ★★★★★ | ✅ (train once) | **Flagship metric** |
| 2 | **RSA / CKA geometry congruence** | C1 | ★★★★★ | ★★★★★ | ✅ | **Holistic test** |
| 3 | **VA decompose-then-compare (asymmetry-aware)** | B1 | ★★★★☆ | ★★☆☆☆ | ✅ | **Interpretable backbone** |
| 4 | **Gromov–Wasserstein alignment** | C2 | ★★★★★ | ★★★★★ | ✅ | Holistic alt / robustness |
| 5 | **(Deep) CCA canonical correlation** | C3 | ★★★★☆ | ★★★★☆ | ✅ | Standard-stats anchor |
| 6 | **Jukebox / JukeMIR + PMI scoring** | D1 | ★★★★★ | ★★★★☆ | ⚠️ cloud/subset | Corroboration |
| 7 | **Categorical-emotion JSD agreement** | B2 | ★★★★☆ | ★★★☆☆ | ✅ | Backbone add-on |
| 8 | **Structural / prosodic stress-beat** | — | ★★★★☆ | ★★☆☆☆ | ✅ | Convergent 3rd view |
| 9 | **CLaMP 3 off-the-shelf** | A2 | ★★★☆☆ | ★★★★☆ | ✅ | Stronger baseline |
| 10 | **Audio-LLM-as-judge** | — | ★★★☆☆ | ★★★☆☆ | ⚠️ | Explanatory only |
| — | *MuLan/CLAP raw cosine (current)* | — | ★★☆☆☆ | ★★★★☆ | ✅ | *baseline to beat* |

---

## 5. Recommended workflow — a convergent battery

Don't pick one number; **triangulate**, and let cross-method agreement be the
evidence (this is also what turns "MuLan is bad" into a positive methodological
contribution). Concretely:

1. **Keep MuLan + CLAP as declared baselines** (the thing you're improving on).
2. **Build LyricLMC (A1)** as the flagship — frozen MERT audio tower + frozen lyric
   text encoder + two trained projection heads, InfoNCE on matched pairs with
   hard negatives. This is the new primary `lmc_*` column.
3. **Add the VA backbone (B1)**, respecting the valence(lyric)/arousal(audio)
   asymmetry — interpretable, reviewer-friendly, and it explains *why* a pair is
   (in)congruent.
4. **Add the geometry test (C1: CKA + local per-song RSA)** — your holistic claim,
   made quantitative and modality-gap-free.
5. **Add one structural metric** (stress↔beat) for a genuinely independent third
   axis.
6. **Corroborate on a random subset** with Jukebox/PMI (D1) and an audio-LLM judge
   for rationale.
7. Report the **inter-method correlation matrix**. High convergence among 2/3/4/5
   *is* your empirical argument that lyric-music congruence is a real, bound,
   high-dimensional construct — and that MuLan under-measures it.

Every step is a cached column in `master_results.csv`, computed by the same
resumable machinery you already have.

---

## 6. Validation design (how you *prove* a metric beats MuLan)

You chose the human survey as ground truth. Make it a formal **construct-validity**
argument (Campbell & Fiske MTMM):

- **Primary (human) criterion.** On the generative-arm stimuli, collect per-item
  congruence-relevant ratings (VA, liking, lyric-music "fit," comprehension). A
  better metric **correlates more strongly with the human congruence judgment** and
  **discriminates the 4×4 congruent-diagonal vs incongruent-antidiagonal cells**
  better than MuLan. Report Spearman ρ and cell-separation effect sizes per metric.
- **Secondary (automatic, whole-corpus) criterion — matched-vs-mismatched.** For
  every real song, score its **true** lyrics vs `k` **random/impostor** lyrics.
  A valid metric ranks the true pair higher → report **retrieval AUC / mean rank /
  recall@k** over the full corpus. Needs no humans, scales infinitely, and is a
  clean objective proxy. (This is also LyricLMC's training objective, so evaluate
  it on **held-out** songs to avoid leakage.)
- **Convergent/discriminant.** Metrics targeting the same level (e.g. LyricLMC vs
  VA-congruence) should correlate (convergent); metrics for different levels
  (affective vs structural) should correlate less (discriminant). That pattern is
  the evidence for a multi-level-but-bound construct.
- **Do NOT** select the metric by how well it predicts **popularity**. Popularity
  is the *downstream* outcome of your thesis; choosing the congruence metric to
  maximize that correlation is circular and, given `LMC = cos(audio, text)` sits on
  the audio→popularity path, collider-/overfit-prone (cf. your own Grinde-2024
  note). Validate the *metric* against humans + matched/mismatched; *then*,
  separately, regress popularity on the validated metric.

---

## 7. Concrete next steps (smallest → largest)

1. **Free 1-hour win:** add **per-modality mean-centering** before every MuLan/CLAP
   cosine (§1.3) and re-run the battery; see if human-correlation and
   cell-separation improve. Establishes the modality-gap contribution cheaply.
2. **Half-day win:** implement the **matched-vs-mismatched AUC harness** on the
   existing corpus. Instantly gives you an *automatic* yardstick to rank *any*
   metric, including the ones you already have.
3. **1–3 day flagship:** implement **LyricLMC (A1)** — projection heads on frozen
   MERT + a lyric text encoder, InfoNCE with hard negatives, evaluate by held-out
   matched/mismatched AUC, then by survey correlation. New `lyriclmc_*` columns.
4. **1–2 day holistic:** add **CKA (corpus) + local RSA (per song)** columns (C1).
5. **Ongoing:** VA backbone (B1), one structural metric, and a Jukebox/PMI subset
   pass (D1) as corroboration.

---

## 8. Key references (verify bibliographic details before citing)

**Problem framing / metrics**
- Hessel et al. (2021) *CLIPScore*. EMNLP. arXiv:2104.08718.
- Liang et al. (2022) *Mind the Gap: the Modality Gap in Multi-modal Contrastive
  Learning*. NeurIPS. arXiv:2203.02053. ("It's Not a Modality Gap,"
  arXiv:2405.18570; COMET audio-text, arXiv:2605.29628.)
- "Revisiting the role of lyrics in MER: the semantic–perceived emotion gap"
  (2026). *ScienceDirect* S156625352600182X.

**Music psychology of the construct**
- Ali & Peynircioğlu (2006) *Songs and emotions: are lyrics and melodies equal
  partners?* Psychology of Music.
- Ma, Baker, Vukovics, Davis & Elliott (2024) replication. *Musicae Scientiae*.
  doi:10.1177/10298649221149109.

**Family A — joint embeddings**
- Huang et al. (2022) *MuLan*. ISMIR.  · Wu et al. (2023) *LAION-CLAP*. ICASSP. ·
  Elizalde et al. (2023) *CLAP*. ICASSP.
- Manco et al. (2022) *Contrastive Audio-Language Learning for Music (MuSCALL)*.
  ISMIR. arXiv:2208.12208.
- Wu (Wood) et al. (2025) *CLaMP 3*. ACL Findings. arXiv:2502.10362.
- Yu et al. (2019) *Deep Cross-Modal Correlation Learning for Audio and Lyrics*.
  ACM TOMM. arXiv:1711.08976.
- Li et al. (2023) *MERT*. arXiv:2306.00107.

**Family B — decompose-then-compare**
- Music2Emo (amaai-lab, 2025, HuggingFace). · "Towards Unified MER across
  Dimensional and Categorical Models" (2025). · MER meta-analysis, *ACM Computing
  Surveys* (2025).
- Mohammad (2018) *NRC-VAD Lexicon*. ACL. · Demszky et al. (2020) *GoEmotions*. ACL.
- Hu & Downie (2010); Malheiro et al. *Emotionally-relevant features for
  classification/regression of music lyrics* (IEEE TAC); Delbouys et al. (2018)
  *Music Mood Detection Based on Audio and Lyrics* (Deezer). ISMIR.

**Family C — representational geometry**
- Kriegeskorte et al. (2008) *Representational Similarity Analysis*. Front. Syst.
  Neurosci. · Kornblith et al. (2019) *Similarity of NN Representations Revisited
  (CKA)*. ICML. · (RSA≡CKA≡CCA, bioRxiv 2024.10.23.619871.)
- Alvarez-Melis & Jaakkola (2018) *Gromov-Wasserstein Alignment of Word Embedding
  Spaces*. EMNLP. arXiv:1809.00013. · OT audio-text retrieval, arXiv:2405.10084.
- Andrew et al. (2013) *Deep CCA*. ICML.

**Family D — generative scoring**
- Dhariwal et al. (2020) *Jukebox*. · Castellon, Donahue & Liang (2021) *JukeMIR*.
  ISMIR. arXiv:2107.05677.

**Structural / prosodic**
- Palmer & Kelly (1992) *Linguistic prosody and musical meter*. · "Multimodal
  Lyrics-Rhythm Matching" arXiv:2301.02732. · "Keywords and Strong Beats in Lyrical
  Music" arXiv:2412.04202. · Gordon et al. (2011) *EEG Correlates of Song Prosody*.
  Front. Psychol.

**Audio-LLM judges**
- Qwen-Audio (arXiv:2311.07919) · SALMONN · MU-LLaMA · Music Flamingo
  (arXiv:2511.10289) · MuChoMusic (arXiv:2408.01337).
</content>
</invoke>
