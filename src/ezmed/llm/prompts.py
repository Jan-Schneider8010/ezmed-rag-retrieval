"""Loader for prompt artefacts under repo-root `prompts/`.

Each prompt is a directory with `system.md` + `user.md`. The user template is
`.format`-able; placeholders use Python format-string syntax (`{name}`).
QA prompting strategies live under `prompts/qa_generation/strategies/<name>.md`.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ezmed.settings import settings


@dataclass(frozen=True)
class Prompt:
    system: str
    user_template: str

    def render_user(self, **kwargs: object) -> str:
        return self.user_template.format(**kwargs)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def load_prompt(name: str) -> Prompt:
    base = settings.prompts_dir / name
    return Prompt(system=_read(base / "system.md"), user_template=_read(base / "user.md"))


@lru_cache(maxsize=None)
def load_qa_strategy(name: str) -> str:
    return _read(settings.prompts_dir / "qa_generation" / "strategies" / f"{name}.md")
