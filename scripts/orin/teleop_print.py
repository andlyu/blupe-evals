"""Terminal-only teleop smoke test — no MuJoCo, no display.

Reads live Quest controller state from the running PC Service via xrobotoolkit_sdk
and prints it. Proves the pipeline end-to-end: Quest -> PC Service -> SDK -> Python.

Run on the Orin (env is set up by run_teleop.sh, which also ensures the service is up):
    bash ~/blupe-evals/scripts/orin/run_teleop.sh ~/blupe-evals/scripts/orin/teleop_print.py

First point the Quest at the Orin: Network -> 192.168.0.185, Controller + Send ON.
Then move the controllers / squeeze grip + trigger and watch the numbers change.
All-zeros means the Quest isn't sending yet (check the Quest Network screen). Ctrl-C to stop.
"""

import time

import tyro
import xrobotoolkit_sdk as xrt


def _vec(v):
    return "[" + " ".join(f"{x:+.2f}" for x in v) + "]"


def main(rate_hz: float = 10.0):
    xrt.init()
    print(f"connected to PC Service. reading at {rate_hz:.0f} Hz — move the controllers. Ctrl-C to stop.\n")
    dt = 1.0 / rate_hz
    try:
        while True:
            rp = xrt.get_right_controller_pose()  # [x,y,z, qx,qy,qz,qw]
            lp = xrt.get_left_controller_pose()
            line = (
                f"R xyz {_vec(rp[:3])} grip {xrt.get_right_grip():.2f} trig {xrt.get_right_trigger():.2f} "
                f"A {int(xrt.get_A_button())} B {int(xrt.get_B_button())} | "
                f"L xyz {_vec(lp[:3])} grip {xrt.get_left_grip():.2f} trig {xrt.get_left_trigger():.2f}"
            )
            print("\r" + line + "   ", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        xrt.close()


if __name__ == "__main__":
    tyro.cli(main)
