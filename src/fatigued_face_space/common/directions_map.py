from dataclasses import dataclass

@dataclass(frozen=True)
class directions:
    missing: str = "missing"
    up: str = "up"
    down: str = "down"
    unchanged: str = "unchanged"

@dataclass(frozen=True) 
class dir_match:
    up: str ="up-match"
    down: str ="down-match"
    unchanged: str ="unchanged-match"
    nomatch: str ="nomatch"
    unchanged_mixed: str = "unchanged-mixed"
    missing: str = "missing"
