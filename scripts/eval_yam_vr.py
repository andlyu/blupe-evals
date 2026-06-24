"""Compatibility wrapper for the renamed Mac <-> Quest bridge.

Use scripts/mac_quest_bridge.py for new commands.
"""

from mac_quest_bridge import main

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
