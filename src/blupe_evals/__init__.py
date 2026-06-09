"""blupe-evals: make it easy for a user to run teleop evals on any robot.

Teleop (XRoboToolkit/Quest) to set up & reset; gate the user's own policy loop
on/off; keep every command safe; track success run-over-run. The user brings a
`Robot` adapter + a `run(robot, stop)` loop — their policy never leaves their side.
"""
