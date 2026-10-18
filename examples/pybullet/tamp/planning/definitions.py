from pathlib import Path

HERE = Path(__file__).resolve().parent
DOMAIN_PDDL = (HERE / "domain.pddl").read_text()
STREAM_PDDL = (HERE / "streams.pddl").read_text()
