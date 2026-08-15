"""Whisper models emit censored profanity; map it back (kept from the
reference worker). Apply to whisper-based providers only, Deepgram doesn't
censor unless asked to."""

from __future__ import annotations

_DECENSOR_MAP = {
    "f**k": "fuck",
    "f***ing": "fucking",
    "f*****g": "fucking",
    "f******": "fucking",
    "fuck***t": "fucking bullshit",
    "fuck***": "fucking",
    "f**ing": "fucking",
    "f*****": "fucker",
    "f***": "fuck",
    "f**": "fuck",
    "sh**": "shit",
    "s**t": "shit",
    "s***": "shit",
    "a**": "ass",
    "b**ch": "bitch",
    "b***h": "bitch",
    "c***": "cunt",
    "p***y": "pussy",
    "d**n": "damn",
    "****": "fuck",
}


def decensor(text: str) -> str:
    for old, new in _DECENSOR_MAP.items():
        text = text.replace(old, new)
        text = text.replace(old.capitalize(), new.capitalize())
    return text
