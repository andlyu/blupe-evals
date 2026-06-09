# robot-models — URDF / MJCF / asset reference

How this stack models a robot arm: the file formats and what reads each. The end-to-end
"add an arm" guidance now lives with the framework it belongs to:
**[../xrobotoolkit/teleop-integration.md](../xrobotoolkit/teleop-integration.md)**.

| I want to… | Read |
|---|---|
| What files a new arm needs (sim vs hardware) + the config schema | [../xrobotoolkit/teleop-integration.md](../xrobotoolkit/teleop-integration.md) |
| Why YAM was painful (URDF/MJCF inconsistency) | [../xrobotoolkit/teleop-integration.md](../xrobotoolkit/teleop-integration.md) §"How YAM violated this" |

Format conventions:
- **URDF** = `urdfdom` XML; read by **placo** (IK) and loadable by MuJoCo.
- **MJCF** (`.xml`) = MuJoCo-native; read by **MuJoCo** (sim/render) and by **placo** via `Flags.mjcf`.
- **scene.xml** = MJCF wrapper that `include`s the robot and adds floor/lights/skybox.
- Meshes = `.stl` under an `assets/` subdir.
- The two model files (`.xml` + `.urdf`) must be **consistent — same link & joint names**.
