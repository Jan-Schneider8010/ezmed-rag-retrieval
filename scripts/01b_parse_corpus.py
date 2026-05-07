"""Step 1b: parse data/raw/*.xml to data/processed/{pmcid}.json."""

import logging
from pathlib import Path

from ezmed.ingestion.parser import parse_article

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

logger = logging.getLogger(__name__)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(RAW_DIR.glob("*.xml"))
    ok, fail = 0, 0
    for xml_path in xml_files:
        out_path = OUT_DIR / f"{xml_path.stem}.json"
        try:
            parsed = parse_article(xml_path.read_bytes())
        except Exception as e:
            logger.warning("failed %s: %s", xml_path.name, e)
            fail += 1
            continue
        out_path.write_text(parsed.model_dump_json(indent=2))
        ok += 1
    print(f"parsed: {ok}, failed: {fail}, total: {len(xml_files)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
