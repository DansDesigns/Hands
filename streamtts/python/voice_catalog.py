#!/usr/bin/env python3
"""Built-in catalog of Piper voices converted for the sherpa-onnx release.

Only metadata is bundled here. Model archives are downloaded on first use so
StreamTTS itself remains small; model files stay in the local models/ folder.
Catalog entries are based on the official sherpa-onnx TTS model pages:
https://k2-fsa.github.io/sherpa/onnx/tts/all/
"""
from __future__ import annotations

# (model ID, friendly label). Keep model IDs identical to official sherpa-onnx
# tts-models release archive names (without the .tar.bz2 suffix).
VOICE_ITEMS: tuple[tuple[str, str], ...] = (
    # English (United Kingdom)
    ("vits-piper-en_GB-alan-low", "Piper · English (UK) · Alan (low)"),
    ("vits-piper-en_GB-alan-medium", "Piper · English (UK) · Alan (medium)"),
    ("vits-piper-en_GB-alba-medium", "Piper · English (UK) · Alba (medium)"),
    ("vits-piper-en_GB-aru-medium", "Piper · English (UK) · Aru (medium)"),
    ("vits-piper-en_GB-cori-high", "Piper · English (UK) · Cori (high)"),
    ("vits-piper-en_GB-cori-medium", "Piper · English (UK) · Cori (medium)"),
    ("vits-piper-en_GB-dii-high", "Piper · English (UK) · Dii (high)"),
    ("vits-piper-en_GB-jenny_dioco-medium", "Piper · English (UK) · Jenny Dioco (medium)"),
    ("vits-piper-en_GB-miro-high", "Piper · English (UK) · Miro (high)"),
    ("vits-piper-en_GB-northern_english_male-medium", "Piper · English (UK) · Northern English male (medium)"),
    ("vits-piper-en_GB-semaine-medium", "Piper · English (UK) · Semaine (medium)"),
    ("vits-piper-en_GB-southern_english_female-low", "Piper · English (UK) · Southern English female (low)"),
    ("vits-piper-en_GB-southern_english_female-medium", "Piper · English (UK) · Southern English female (medium)"),
    ("vits-piper-en_GB-southern_english_male-medium", "Piper · English (UK) · Southern English male (medium)"),
    ("vits-piper-en_GB-vctk-medium", "Piper · English (UK) · VCTK (medium; default speaker)"),

    # English (United States)
    ("vits-piper-en_US-amy-low", "Piper · English (US) · Amy (low)"),
    ("vits-piper-en_US-amy-medium", "Piper · English (US) · Amy (medium)"),
    ("vits-piper-en_US-arctic-medium", "Piper · English (US) · Arctic (medium)"),
    ("vits-piper-en_US-bryce-medium", "Piper · English (US) · Bryce (medium)"),
    ("vits-piper-en_US-danny-low", "Piper · English (US) · Danny (low)"),
    ("vits-piper-en_US-glados-high", "Piper · English (US) · GLaDOS (high)"),
    ("vits-piper-en_US-hfc_female-medium", "Piper · English (US) · HFC female (medium)"),
    ("vits-piper-en_US-hfc_male-medium", "Piper · English (US) · HFC male (medium)"),
    ("vits-piper-en_US-joe-medium", "Piper · English (US) · Joe (medium)"),
    ("vits-piper-en_US-john-medium", "Piper · English (US) · John (medium)"),
    ("vits-piper-en_US-kathleen-low", "Piper · English (US) · Kathleen (low)"),
    ("vits-piper-en_US-kristin-medium", "Piper · English (US) · Kristin (medium)"),
    ("vits-piper-en_US-kusal-medium", "Piper · English (US) · Kusal (medium)"),
    ("vits-piper-en_US-l2arctic-medium", "Piper · English (US) · L2 Arctic (medium)"),
    ("vits-piper-en_US-lessac-high", "Piper · English (US) · Lessac (high)"),
    ("vits-piper-en_US-lessac-low", "Piper · English (US) · Lessac (low)"),
    ("vits-piper-en_US-lessac-medium", "Piper · English (US) · Lessac (medium)"),
    ("vits-piper-en_US-ljspeech-high", "Piper · English (US) · LJSpeech (high)"),
    ("vits-piper-en_US-ljspeech-medium", "Piper · English (US) · LJSpeech (medium)"),
    ("vits-piper-en_US-miro-high", "Piper · English (US) · Miro (high)"),
    ("vits-piper-en_US-norman-medium", "Piper · English (US) · Norman (medium)"),
    ("vits-piper-en_US-reza_ibrahim-medium", "Piper · English (US) · Reza Ibrahim (medium)"),
    ("vits-piper-en_US-ryan-high", "Piper · English (US) · Ryan (high)"),
    ("vits-piper-en_US-ryan-low", "Piper · English (US) · Ryan (low)"),
    ("vits-piper-en_US-ryan-medium", "Piper · English (US) · Ryan (medium)"),
    ("vits-piper-en_US-sam-medium", "Piper · English (US) · Sam (medium)"),

    # French
    ("vits-piper-fr_FR-gilles-low", "Piper · Français · Gilles (low)"),
    ("vits-piper-fr_FR-siwis-low", "Piper · Français · Siwis (low)"),
    ("vits-piper-fr_FR-siwis-medium", "Piper · Français · Siwis (medium)"),
    ("vits-piper-fr_FR-tom-medium", "Piper · Français · Tom (medium)"),
    ("vits-piper-fr_FR-upmc-medium", "Piper · Français · UPMC (medium)"),

    # German
    ("vits-piper-de_DE-eva_k-x_low", "Piper · Deutsch · Eva K (x-low)"),
    ("vits-piper-de_DE-kerstin-low", "Piper · Deutsch · Kerstin (low)"),
    ("vits-piper-de_DE-pavoque-low", "Piper · Deutsch · Pavoque (low)"),
    ("vits-piper-de_DE-ramona-low", "Piper · Deutsch · Ramona (low)"),
    ("vits-piper-de_DE-thorsten-low", "Piper · Deutsch · Thorsten (low)"),
    ("vits-piper-de_DE-thorsten-medium", "Piper · Deutsch · Thorsten (medium)"),
    ("vits-piper-de_DE-thorsten_emotional-medium", "Piper · Deutsch · Thorsten emotional (medium)"),

    # Dutch
    ("vits-piper-nl_BE-nathalie-medium", "Piper · Nederlands (Belgium) · Nathalie (medium)"),
    ("vits-piper-nl_NL-alex-medium", "Piper · Nederlands · Alex (medium)"),
    ("vits-piper-nl_NL-pim-medium", "Piper · Nederlands · Pim (medium)"),
    ("vits-piper-nl_NL-ronnie-medium", "Piper · Nederlands · Ronnie (medium)"),

    # Spanish
    ("vits-piper-es_AR-daniela-high", "Piper · Español (Argentina) · Daniela (high)"),
    ("vits-piper-es_ES-carlfm-x_low", "Piper · Español (Spain) · CarlFM (x-low)"),
    ("vits-piper-es_ES-davefx-medium", "Piper · Español (Spain) · DaveFX (medium)"),
    ("vits-piper-es_ES-sharvard-medium", "Piper · Español (Spain) · Sharvard (medium)"),
    ("vits-piper-es_MX-ald-medium", "Piper · Español (Mexico) · Ald (medium)"),
    ("vits-piper-es_MX-claude-high", "Piper · Español (Mexico) · Claude (high)"),

    # Portuguese
    ("vits-piper-pt_BR-edresson-low", "Piper · Português (Brazil) · Edresson (low)"),
    ("vits-piper-pt_BR-faber-medium", "Piper · Português (Brazil) · Faber (medium)"),
    ("vits-piper-pt_BR-jeff-medium", "Piper · Português (Brazil) · Jeff (medium)"),
    ("vits-piper-pt_PT-tugao-medium", "Piper · Português (Portugal) · Tugão (medium)"),

    # Italian
    ("vits-piper-it_IT-paola-medium", "Piper · Italiano · Paola (medium)"),
    ("vits-piper-it_IT-riccardo-x_low", "Piper · Italiano · Riccardo (x-low)"),

    # Polish
    ("vits-piper-pl_PL-gosia-medium", "Piper · Polski · Gosia (medium)"),
    ("vits-piper-pl_PL-justyna_wg_glos-medium", "Piper · Polski · Justyna (medium)"),
)

VOICE_CATALOG: dict[str, dict[str, str]] = {
    model_id: {"label": label, "model_name": model_id}
    for model_id, label in VOICE_ITEMS
}

if len(VOICE_CATALOG) != len(VOICE_ITEMS):
    raise RuntimeError("Duplicate model ID in StreamTTS voice catalog")
