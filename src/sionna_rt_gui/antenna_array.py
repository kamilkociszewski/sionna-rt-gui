from __future__ import annotations

from dataclasses import dataclass

import polyscope.imgui as psim
from sionna import rt
from sionna.rt.antenna_pattern import (
    antenna_pattern_registry,
    polarization_registry,
)

from .sionna_utils import add_paths_to_polyscope


@dataclass(kw_only=True)
class AntennaArrayConfig:
    """
    Class holding configuration parameters for an antenna array.
    This is needed because once created, antenna array objects don't
    expose those fields.
    """

    num_rows: int = 1
    num_cols: int = 1
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    pattern_i: int = antenna_pattern_registry.list().index("iso")
    polarization_i: int = polarization_registry.list().index("cross")

    @property
    def pattern(self) -> str:
        return antenna_pattern_registry.list()[self.pattern_i]

    @property
    def polarization(self) -> str:
        return polarization_registry.list()[self.polarization_i]

    def create(self) -> rt.AntennaArray:
        return rt.PlanarArray(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            vertical_spacing=self.vertical_spacing,
            horizontal_spacing=self.horizontal_spacing,
            pattern=self.pattern,
            polarization=self.polarization,
        )


def _antenna_array_gui(
    name: str, cfg: AntennaArrayConfig, array: rt.AntennaArray
) -> tuple[rt.AntennaArray, bool]:
    needs_update = False

    if psim.TreeNode(name):
        # psim.Text(f"Array size: {dr.width(array.normalized_positions)}\n")

        changed, cfg.pattern_i = psim.Combo(
            f"Pattern##{name}", cfg.pattern_i, antenna_pattern_registry.list()
        )
        needs_update |= changed
        changed, cfg.polarization_i = psim.Combo(
            f"Polarization##{name}",
            cfg.polarization_i,
            polarization_registry.list(),
        )
        needs_update |= changed

        changed, (cfg.num_rows, cfg.num_cols) = psim.InputInt2(
            f"Rows, cols##{name}", (cfg.num_rows, cfg.num_cols)
        )
        needs_update |= changed
        changed, (cfg.vertical_spacing, cfg.horizontal_spacing) = psim.InputFloat2(
            f"V/H spacing##{name}", (cfg.vertical_spacing, cfg.horizontal_spacing)
        )
        needs_update |= changed

        psim.Spacing()
        psim.TreePop()

    if needs_update:
        array = cfg.create()
    return array, needs_update


def antenna_array_gui(gui: "SionnaRtGui"):
    gui.scene._tx_array, tx_changed = _antenna_array_gui(
        "TX array", gui.tx_array_config, gui.scene._tx_array
    )
    gui.scene._rx_array, rx_changed = _antenna_array_gui(
        "RX array", gui.rx_array_config, gui.scene._rx_array
    )

    if tx_changed:
        # Note: receivers don't affect radio maps.
        gui.reset_radio_map()

    if (tx_changed or rx_changed) and gui.cfg.paths.auto_update:
        # TODO: probably should move this to a little method
        gui.clear_paths()
        gui.paths = gui.compute_paths()
        add_paths_to_polyscope(gui, gui.paths, gui.ps_groups)
