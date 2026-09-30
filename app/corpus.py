"""English-only training rows and hand-written evaluation rows.

Training text is English, matching the Jigsaw / Civil Comments setup in the
problem statement. The multilingual challenge set is never used for fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.lexicon import LABELS

_INSULTS = [
    "idiot",
    "moron",
    "stupid",
    "dumb",
    "loser",
    "pathetic",
    "clown",
    "imbecile",
]
_OBSCENE = ["fuck you", "you shithead", "asshole", "stupid bitch", "bastard"]
_THREATS = [
    "I will kill you",
    "I'll kill you",
    "I am going to hurt you",
    "I will find you",
    "you should die",
    "kill yourself",
]
_IDENTITY = [
    "go back to your country",
    "people like you don't belong here",
    "your kind is not welcome",
    "your race does not belong in this thread",
    "your religion makes you unfit for this group",
    "typical immigrant, always complaining",
]
_CIVIL = [
    "The 2019 snapshot shows two separate discussion clusters.",
    "Thanks for posting the co-authorship list from the archive.",
    "I disagree with the campaign slogan, but the turnout numbers are useful.",
    "Can someone check this citation against the library copy?",
    "The bridge between the north and south forums appeared in March.",
    "Our lab added two joint papers after the workshop.",
    "Please share the primary source before we amplify this claim.",
    "The rumour reached the outer ring of accounts within a day.",
    "Fact-checking cut three reshare edges in the last snapshot.",
    "I think the communities merged once the hashtags overlapped.",
    "Bob and Carol started a joint project on diffusion networks.",
    "This thread should stay focused on the election timetable.",
    "The archive deposit for this year is complete.",
    "Eve's public-health notes help explain the cascade.",
    "Let's compare modularity before and after the intervention.",
    "Neighbourhood reports are more reliable than the anonymous post.",
    "The student union minutes are in the repository.",
    "I appreciate the careful summary of the debate.",
    "Cross-posting increased, yet the two groups are still distinct.",
    "We should label uncertain claims instead of deleting the whole thread.",
]


def _row(text: str, **flags: int) -> dict:
    item = {"text": text}
    for label in LABELS:
        item[label] = int(flags.get(label, 0))
    if any(item[label] for label in LABELS if label != "toxic"):
        item["toxic"] = 1
    return item


def build_training_frame(n_toxic: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    toxic_rows: list[dict] = []
    wrappers = [
        "{}",
        "Honestly, {}.",
        "Listen, {}.",
        "On this thread: {}.",
        "I mean it, {}.",
        "Everyone can see that {}.",
    ]
    targets = ["you", "this group", "the organisers", "that account", "all of you"]
    per_kind = n_toxic // 5

    for _ in range(per_kind):
        word = _INSULTS[int(rng.integers(0, len(_INSULTS)))]
        target = targets[int(rng.integers(0, len(targets)))]
        text = wrappers[int(rng.integers(0, len(wrappers)))].format(f"{target} are a {word}")
        toxic_rows.append(_row(text, toxic=1, insult=1))

    for _ in range(per_kind):
        phrase = _OBSCENE[int(rng.integers(0, len(_OBSCENE)))]
        text = wrappers[int(rng.integers(0, len(wrappers)))].format(phrase)
        toxic_rows.append(_row(text, toxic=1, obscene=1, insult=1))

    for _ in range(per_kind):
        phrase = _THREATS[int(rng.integers(0, len(_THREATS)))]
        text = wrappers[int(rng.integers(0, len(wrappers)))].format(phrase)
        severe = 1 if "kill" in phrase or "die" in phrase or "hurt" in phrase else 0
        toxic_rows.append(_row(text, toxic=1, threat=1, severe_toxic=severe))

    for _ in range(per_kind):
        phrase = _IDENTITY[int(rng.integers(0, len(_IDENTITY)))]
        text = wrappers[int(rng.integers(0, len(wrappers)))].format(phrase)
        toxic_rows.append(_row(text, toxic=1, identity_hate=1, insult=1))

    for _ in range(per_kind):
        phrase = _THREATS[int(rng.integers(0, len(_THREATS)))]
        dirty = _OBSCENE[int(rng.integers(0, len(_OBSCENE)))]
        text = f"{dirty}. {phrase}."
        toxic_rows.append(_row(text, toxic=1, obscene=1, threat=1, severe_toxic=1, insult=1))

    civil_rows: list[dict] = []
    tails = [
        "",
        " I would like to hear the other forum too.",
        " The snapshot is worth keeping.",
        " Please add the date of the deposit.",
        " This belongs in the methods note.",
    ]
    while len(civil_rows) < len(toxic_rows):
        text = _CIVIL[int(rng.integers(0, len(_CIVIL)))]
        tail = tails[int(rng.integers(0, len(tails)))]
        civil_rows.append(_row(text + tail))

    frame = pd.DataFrame(toxic_rows + civil_rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


ENGLISH_HOLDOUT = [
    _row("What a pathetic excuse for a moderator.", toxic=1, insult=1),
    _row("You are a clown and everyone knows it.", toxic=1, insult=1),
    _row("The archived graph from May still has two components."),
    _row("I'll kill you if you keep posting those minutes.", toxic=1, threat=1, severe_toxic=1),
    _row("Please attach the DOI before the next lab meeting."),
    _row("Fuck you, this campaign is trash.", toxic=1, obscene=1, insult=1),
    _row("Go back to your country and stop voting in ours.", toxic=1, insult=1, identity_hate=1),
    _row("Your kind never contributes anything useful.", toxic=1, identity_hate=1),
    _row("I will find you after the rally.", toxic=1, threat=1, severe_toxic=1),
    _row("Thanks, the split between the two wards is clearer now."),
    _row("Shut up, you dumb loser.", toxic=1, insult=1),
    _row("You should die for sharing that pamphlet.", toxic=1, threat=1, severe_toxic=1),
    _row("Carol's note on link rot belongs in the methods section."),
    _row("Asshole, delete this before I report the archive.", toxic=1, obscene=1, insult=1),
    _row("People like you don't belong in a public forum.", toxic=1, identity_hate=1),
    _row("The intervention removed the edges out of the source account."),
    _row("You moron, you ruined the whole discussion.", toxic=1, insult=1),
    _row("Can we avoid insults and look at the membership counts?"),
    _row("Kill yourself, nobody wants your opinion.", toxic=1, threat=1, severe_toxic=1),
    _row("The co-authors list grew after the 2021 workshop."),
    _row("Stupid bitch, stay out of this ward thread.", toxic=1, obscene=1, insult=1),
    _row("Your race does not belong in this movement.", toxic=1, identity_hate=1),
    _row("I am going to hurt you for fact-checking me.", toxic=1, threat=1, severe_toxic=1),
    _row("Both communities were still separate in the mid snapshot."),
    _row("This is a useful record of how the hashtag spread."),
    _row("You are an imbecile and a bastard.", toxic=1, insult=1, obscene=1),
    _row("Watch your back when you leave the forum tonight.", toxic=1, threat=1),
    _row("Typical immigrant, stealing the thread as usual.", toxic=1, insult=1, identity_hate=1),
    _row("Modularity dropped once the cross-links became dense."),
    _row("I disagree with the slogan. Here is a calmer wording."),
]


MULTILINGUAL_CHALLENGE = [
    _row("Todos ustedes son unos idiotas.", toxic=1, insult=1),
    _row("Gracias por el enlace al archivo de marzo."),
    _row("Cállate, imbécil.", toxic=1, insult=1),
    _row("Te voy a matar si sigues publicando.", toxic=1, threat=1, severe_toxic=1),
    _row("Vete a tu país, aquí no pintas nada.", toxic=1, insult=1, identity_hate=1),
    _row("Esto es una mierda de campaña.", toxic=1, obscene=1),
    _row("Vous êtes tous des imbéciles.", toxic=1, insult=1),
    _row("Le graphe de co-auteurs a grandi après l'atelier."),
    _row("Ferme ta gueule, connard.", toxic=1, insult=1, obscene=1),
    _row("Je vais te tuer.", toxic=1, threat=1, severe_toxic=1),
    _row("Rentrez dans votre pays.", toxic=1, insult=1, identity_hate=1),
    _row("Halt die Fresse, du Idiot.", toxic=1, insult=1, obscene=1),
    _row("Die Gemeinschaften sind im Archiv noch getrennt."),
    _row("Ich bringe dich um.", toxic=1, threat=1, severe_toxic=1),
    _row("Geh zurück in dein Land.", toxic=1, insult=1, identity_hate=1),
    _row("तुम सब बेवकूफ हो.", toxic=1, insult=1),
    _row("चुप रहो और यह धागा पढ़ो.", toxic=1, insult=1),
    _row("मर जा, किसी को तुम्हारी बात नहीं चाहिए.", toxic=1, threat=1, severe_toxic=1),
    _row("अपने देश वापस जा.", toxic=1, insult=1, identity_hate=1),
    _row("Cala a boca, idiota.", toxic=1, insult=1),
    _row("O arquivo mostra dois grupos no início."),
    _row("Vou te matar.", toxic=1, threat=1, severe_toxic=1),
    _row("Volta para o teu país.", toxic=1, insult=1, identity_hate=1),
    _row("The snapshot is useful. Gracias por compartirlo."),
]


def holdout_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
