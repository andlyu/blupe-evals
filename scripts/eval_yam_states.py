"""Compatibility wrapper for the renamed eval state machine.

Use scripts/eval_states.py for new commands.
"""

from eval_states import *  # noqa: F401,F403


if __name__ == "__main__":
    import tyro
    from eval_states import main

    tyro.cli(main)
