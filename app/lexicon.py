"""Hand-crafted multilingual cues used as six extra features.

The classifier is trained on English labels only. These phrases map every
language onto the same six feature slots, so a weight learned from English
insults still fires when the same slot is triggered by another language.
That is the prototype stand-in for cross-lingual transfer (XLM-R / mDistilBERT
in the problem statement).
"""

from __future__ import annotations

LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

# Each entry contributes to one or more labels. Matching is casefold substring.
PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    # English insults
    ("idiot", ("toxic", "insult")),
    ("idiots", ("toxic", "insult")),
    ("moron", ("toxic", "insult")),
    ("morons", ("toxic", "insult")),
    ("stupid", ("toxic", "insult")),
    ("dumb", ("toxic", "insult")),
    ("loser", ("toxic", "insult")),
    ("pathetic", ("toxic", "insult")),
    ("trash", ("toxic", "insult")),
    ("clown", ("toxic", "insult")),
    ("imbecile", ("toxic", "insult")),
    ("shut up", ("toxic", "insult")),
    ("shut your mouth", ("toxic", "insult")),
    # English obscenity
    ("fuck", ("toxic", "obscene")),
    ("shit", ("toxic", "obscene")),
    ("asshole", ("toxic", "obscene")),
    ("bitch", ("toxic", "obscene")),
    ("bastard", ("toxic", "obscene")),
    # English threats
    ("i will kill", ("toxic", "severe_toxic", "threat")),
    ("i'll kill", ("toxic", "severe_toxic", "threat")),
    ("going to hurt you", ("toxic", "severe_toxic", "threat")),
    ("i will find you", ("toxic", "severe_toxic", "threat")),
    ("watch your back", ("toxic", "threat")),
    ("you should die", ("toxic", "severe_toxic", "threat")),
    ("kill yourself", ("toxic", "severe_toxic", "threat")),
    ("i'll hurt", ("toxic", "severe_toxic", "threat")),
    # English identity hate
    ("go back to your country", ("toxic", "insult", "identity_hate")),
    ("your kind", ("toxic", "identity_hate")),
    ("people like you don't belong", ("toxic", "identity_hate")),
    ("don't belong here", ("toxic", "identity_hate")),
    ("your race", ("toxic", "identity_hate")),
    ("your religion", ("toxic", "identity_hate")),
    ("typical immigrant", ("toxic", "insult", "identity_hate")),
    # Spanish
    ("idiota", ("toxic", "insult")),
    ("idiotas", ("toxic", "insult")),
    ("estúpido", ("toxic", "insult")),
    ("estupido", ("toxic", "insult")),
    ("imbécil", ("toxic", "insult")),
    ("imbecil", ("toxic", "insult")),
    ("cállate", ("toxic", "insult")),
    ("callate", ("toxic", "insult")),
    ("mierda", ("toxic", "obscene")),
    ("vete a tu país", ("toxic", "insult", "identity_hate")),
    ("vete a tu pais", ("toxic", "insult", "identity_hate")),
    ("te voy a matar", ("toxic", "severe_toxic", "threat")),
    ("odio a tu gente", ("toxic", "identity_hate")),
    # French
    ("imbécile", ("toxic", "insult")),
    ("imbecile", ("toxic", "insult")),
    ("idiot", ("toxic", "insult")),
    ("taisez-vous", ("toxic", "insult")),
    ("ferme ta gueule", ("toxic", "insult", "obscene")),
    ("connard", ("toxic", "obscene", "insult")),
    ("je vais te tuer", ("toxic", "severe_toxic", "threat")),
    ("rentrez dans votre pays", ("toxic", "identity_hate", "insult")),
    ("votre race", ("toxic", "identity_hate")),
    # German
    ("idiot", ("toxic", "insult")),
    ("halt die fresse", ("toxic", "insult", "obscene")),
    ("arschloch", ("toxic", "obscene", "insult")),
    ("ich bringe dich um", ("toxic", "severe_toxic", "threat")),
    ("geh zurück in dein land", ("toxic", "identity_hate", "insult")),
    ("geh zuruck in dein land", ("toxic", "identity_hate", "insult")),
    ("leute wie dich", ("toxic", "identity_hate")),
    # Hindi (Devanagari and a common transliteration)
    ("बेवकूफ", ("toxic", "insult")),
    ("bevakoof", ("toxic", "insult")),
    ("चुप रहो", ("toxic", "insult")),
    ("chup raho", ("toxic", "insult")),
    ("मर जा", ("toxic", "severe_toxic", "threat")),
    ("mar ja", ("toxic", "severe_toxic", "threat")),
    ("तेरे जैसे", ("toxic", "identity_hate")),
    ("apne desh wapas ja", ("toxic", "identity_hate", "insult")),
    ("अपने देश वापस जा", ("toxic", "identity_hate", "insult")),
    # Portuguese
    ("idiota", ("toxic", "insult")),
    ("cala a boca", ("toxic", "insult")),
    ("vou te matar", ("toxic", "severe_toxic", "threat")),
    ("volta para o teu país", ("toxic", "identity_hate", "insult")),
    ("volta para o teu pais", ("toxic", "identity_hate", "insult")),
]


def lexicon_hits(text: str) -> dict[str, int]:
    folded = text.casefold()
    counts = {label: 0 for label in LABELS}
    for phrase, labels in PATTERNS:
        if phrase.casefold() in folded:
            for label in labels:
                counts[label] += 1
    return counts
