from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import time

import drjit as dr
from polyscope import imgui as psim

# TODO: dataclass to store the animation trajectory + options & eval it
# TODO: GUI pane in the selection window to edit the trajectory: add/remove points, edit their position.


@dataclass(kw_only=True)
class Trajectory:
    # Whether to enable the trajectory
    enabled: bool = True

    points: list[dr.ScalarPoint3f] = field(default_factory=list)
    # Movement velocity [m/s]. Default is set based on a walking speed of 4 km/h.
    velocity: float = 1.11
    # Whether the trajectory should start playing backwards when it reaches the end
    is_mirroring: bool = True


@dataclass(kw_only=True)
class AnimationConfig:
    playing: bool = False
    speed_multiplier: float = 1.0

    # Time at which the animation started playing the first time (Unix timestamp)
    time_started: float | None = None

    trajectories: dict[str, Trajectory] = field(
        default_factory=lambda: defaultdict(Trajectory)
    )


def animation_gui(gui: "SionnaRtGui"):
    """
    GUI for the main animation controls.
    """
    was_playing = gui.animation_config.playing
    toggled = psim.Button("Pause" if was_playing else "Play")
    if toggled:
        gui.animation_config.playing = not gui.animation_config.playing
        if gui.animation_config.playing:
            gui.animation_config.time_started = time.time()

    if psim.Button("0.5x"):
        gui.animation_config.speed_multiplier = 0.5
    psim.SameLine()
    if psim.Button("1x"):
        gui.animation_config.speed_multiplier = 1.0
    psim.SameLine()
    if psim.Button("2x"):
        gui.animation_config.speed_multiplier = 2.0
    psim.SameLine()
    if psim.Button("10x"):
        gui.animation_config.speed_multiplier = 10.0
    psim.SameLine()
    psim.Text(f"Speed: {gui.animation_config.speed_multiplier:.1f}x")


def trajectory_gui(gui: "SionnaRtGui", object: rt.SceneObject):
    """
    GUI to edit the animation trajectory & velocity of a selected object.
    """
    traj = gui.animation_config.trajectories[object.name]

    # TODO: add/remove/split trajectory points
    # TODO: movement gizmo for the trajectory points

    _, traj.velocity = psim.SliderFloat("Velocity [m/s]", traj.velocity, 0.1, 10.0)

    _, traj.is_mirroring = psim.Checkbox("Mirror", traj.is_mirroring)


def animation_tick(gui: "SionnaRtGui"):
    """
    Tick the animation.
    """
    cfg = gui.animation_config
    if not cfg.playing:
        return

    # for traj in gui.animation_config.trajectories.values():
    #     if traj.is_mirroring:
    #         traj.points.reverse()
    #     traj.points.append(traj.points[-1] + traj.velocity * time.time())
