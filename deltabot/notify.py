"""Logging setup and optional Telegram alerts."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import requests

from .config import ROOT, Settings

log = logging.getLogger("deltabot")


def setup_logging(level: int = logging.INFO) -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file = RotatingFileHandler(ROOT / "logs" / "bot.log", maxBytes=5_000_000, backupCount=5)
    file.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file)


class Notifier:
    def __init__(self, s: Settings):
        self.token, self.chat = s.telegram_token, s.telegram_chat_id
        self.prefix = "[LIVE]" if s.is_live else "[PAPER]"

    def __call__(self, msg: str) -> None:
        log.info("ALERT %s", msg)
        if not (self.token and self.chat):
            return
        try:
            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          data={"chat_id": self.chat, "text": f"{self.prefix} {msg}"}, timeout=10)
        except requests.RequestException as e:
            log.warning("telegram failed: %s", e)
