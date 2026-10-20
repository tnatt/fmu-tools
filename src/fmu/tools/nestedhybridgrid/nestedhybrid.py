"""Nested hybrid grid creation.

Create a merged grid where one region is refined (subdivided) and stitched
back into the original grid via Non-Neighbour Connections (NNCs).

Public API
----------
create_nested_hybrid_grid : function
    Build a nested hybrid grid from a coarse grid, a region property,
    and a refinement specification.
nnc_to_gridproperty : function
    Convert NNC transmissibility DataFrames to GridProperty instances.
nnc_to_flowsimulator_input : function
    Write NNC transmissibilities to a flow-simulator input file.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

import numpy as np
import pandas as pd
import xtgeo
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable
    from xtgeo.common.types import Dimensions


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

FaceDirection: TypeAlias = Literal["i+", "i-", "j+", "j-", "k+", "k-"]
IJKIndex: TypeAlias = tuple[int, int, int]
BoundaryCellFace: TypeAlias = tuple[IJKIndex, IJKIndex, FaceDirection]
ORIGINAL_IJK_PROPERTY_NAMES = ("parent_I", "parent_J", "parent_K")


class BoundingBox(BaseModel):
    imin: int = Field(ge=1)
    imax: int = Field(ge=1)
    jmin: int = Field(ge=1)
    jmax: int = Field(ge=1)
    kmin: int = Field(ge=1)
    kmax: int = Field(ge=1)

    @classmethod
    def from_condition(cls, condition: np.ma.MaskedArray) -> Self:
        """Get the ijk bounding box from a 3D boolean mask."""
        region_indices = condition.nonzero()

        return cls(
            imin=int(region_indices[0].min() + 1),
            imax=int(region_indices[0].max() + 1),
            jmin=int(region_indices[1].min() + 1),
            jmax=int(region_indices[1].max() + 1),
            kmin=int(region_indices[2].min() + 1),
            kmax=int(region_indices[2].max() + 1),
        )


class Refinement(BaseModel):
    col: int = Field(ge=1)
    row: int = Field(ge=1)
    lay: int = Field(ge=1)

    @classmethod
    def from_tuple(cls, refinement: tuple[int, int, int]) -> Self:
        """Create a refinement model from a 3-tuple."""
        try:
            col, row, lay = refinement
            return cls(col=col, row=row, lay=lay)
        except ValidationError:
            raise ValueError("Refinement factors must be >= 1")


class FaultFaceDirection(IntEnum):
    """RMS fault face codes from RMSAPI FaceDirection.

    RMSAPI values:
    - west=0, east=1, south=2, north=3, top=4, bottom=5
    """

    WEST = 0
    EAST = 1
    SOUTH = 2
    NORTH = 3
    TOP = 4
    BOTTOM = 5


@dataclass(frozen=True)
class GridFault:
    name: str
    fw_indices: np.ma.MaskedArray
    hw_indices: np.ma.MaskedArray
    fw_face: np.ma.MaskedArray
    hw_face: np.ma.MaskedArray


def _get_nested_fault_cells(
    cells: np.ndarray, face: int, nested_dimensions: Dimensions
) -> np.ndarray:
    """Return nested flat-cell positions on a parent-cell face."""

    ii, jj, kk = np.unravel_index(cells, nested_dimensions)

    face_code = FaultFaceDirection(face)

    if face_code is FaultFaceDirection.WEST:
        selected = ii == ii.min()
    elif face_code is FaultFaceDirection.EAST:
        selected = ii == ii.max()
    elif face_code is FaultFaceDirection.SOUTH:
        selected = jj == jj.max()
    elif face_code is FaultFaceDirection.NORTH:
        selected = jj == jj.min()
    elif face_code is FaultFaceDirection.TOP:
        selected = kk == kk.min()
    elif face_code is FaultFaceDirection.BOTTOM:
        selected = kk == kk.max()
    else:
        raise ValueError(f"Unexpected fault face code: {face_code}")

    return cells[selected]


def _create_flat_to_rms_cell_number_mapping(grid_indexer: Any) -> np.ma.MaskedArray:
    """Map C-order flat grid indices to RMS cell numbers."""
    ncol, nrow, nlay = grid_indexer.dimensions

    cell_numbers = grid_indexer.get_cell_numbers_in_range((0, 0, 0), (ncol, nrow, nlay))
    i, j, k = grid_indexer.get_indices(cell_numbers).T

    flat_indices = _ijk_to_flat_index(i=i, j=j, k=k, jdim=nrow, kdim=nlay)

    flat_to_rms_cell_number = np.ma.masked_all(ncol * nrow * nlay, dtype=np.int64)
    flat_to_rms_cell_number[flat_indices] = cell_numbers
    return flat_to_rms_cell_number


def _crop_for_region(grid: xtgeo.Grid, refinement_bbox: BoundingBox) -> xtgeo.Grid:
    """Crop grid to the bounding box of the refinement region."""

    cropped_grid = grid.copy()

    irange = (refinement_bbox.imin, refinement_bbox.imax)
    jrange = (refinement_bbox.jmin, refinement_bbox.jmax)
    krange = (refinement_bbox.kmin, refinement_bbox.kmax)

    cropped_grid.crop(irange, jrange, krange, props="all")
    _logger.info("Cropped grid dimensions: %s", cropped_grid.dimensions)

    return cropped_grid


def _find_boundary_faces(refined_area: np.ma.MaskedArray) -> list[BoundaryCellFace]:
    """Find cell faces on the boundary of a refinement area.

    Args:
        refined_area: 3D boolean array where ``True`` marks cells to refine.

    Returns:
        A list of ``(outside_ijk, inside_ijk, face_dir)`` where indices are
        0-based and *face_dir* is one of ``'i+', 'i-', 'j+', 'j-', 'k+', 'k-'``.
    """
    ni, nj, nk = refined_area.shape
    active = ~refined_area.mask

    in_target = refined_area & active
    outside_active = ~refined_area & active

    faces: list[BoundaryCellFace] = []

    # i+
    mask = in_target[: ni - 1, :, :] & outside_active[1:ni, :, :]
    for i, j, k in np.argwhere(mask):
        faces.append(((int(i + 1), int(j), int(k)), (int(i), int(j), int(k)), "i+"))

    # i-
    mask = in_target[1:ni, :, :] & outside_active[: ni - 1, :, :]
    for idx, j, k in np.argwhere(mask):
        faces.append(((int(idx), int(j), int(k)), (int(idx + 1), int(j), int(k)), "i-"))

    # j+
    mask = in_target[:, : nj - 1, :] & outside_active[:, 1:nj, :]
    for i, j, k in np.argwhere(mask):
        faces.append(((int(i), int(j + 1), int(k)), (int(i), int(j), int(k)), "j+"))

    # j-
    mask = in_target[:, 1:nj, :] & outside_active[:, : nj - 1, :]
    for i, jdx, k in np.argwhere(mask):
        faces.append(((int(i), int(jdx), int(k)), (int(i), int(jdx + 1), int(k)), "j-"))

    # k+
    mask = in_target[:, :, : nk - 1] & outside_active[:, :, 1:nk]
    for i, j, k in np.argwhere(mask):
        faces.append(((int(i), int(j), int(k + 1)), (int(i), int(j), int(k)), "k+"))

    # k-
    mask = in_target[:, :, 1:nk] & outside_active[:, :, : nk - 1]
    for i, j, kdx in np.argwhere(mask):
        faces.append(((int(i), int(j), int(kdx)), (int(i), int(j), int(kdx + 1)), "k-"))

    _logger.info("Found %d boundary faces for refinement area", len(faces))
    return faces


def _compute_nnc_table(
    refined_area: np.ma.MaskedArray,
    refinement: Refinement,
    coarse_ncol: int,
    lmap1: np.ndarray,
    lmap2: np.ndarray,
) -> pd.DataFrame:
    """Compute NNC cell-pair mapping between mother and refined cells.

    For each boundary face between the target region and the surrounding mother
    cells, this function determines which refined sub-cells in the merged grid
    connect to which mother cell, and through which face direction.

    The mapping is purely topological (index-based) — no geometric computation
    is performed here.  The resulting table is intended to be passed to
    :meth:`xtgeo.Grid.get_transmissibilities` so that it can compute the
    actual transmissibility for each cell pair.

    Convention:
        - ``I1, J1, K1`` is always the **mother** cell (1-based, merged grid).
        - ``I2, J2, K2`` is always the **refined** cell (1-based, merged grid).
        - ``DIRECTION`` is from the mother cell's perspective (e.g. ``"I+"``
          means looking in the positive I-direction from the mother cell
          you reach the refined cell).

    Args:
        refined_area: 3D boolean array where ``True`` marks cells to refine.
        refinement: ``(rcol, rrow, rlay)`` refinement factors.
        coarse_ncol: Number of columns in the coarse grid (grid1 in the merge).
        lmap1: Numpy array with layer_mapping (input k -> output k) for grid1
        lmap2: Numpy array with layer_mapping (input k -> output k) for grid2


    Returns:
        A DataFrame with columns ``I1, J1, K1, I2, J2, K2, DIRECTION``.
    """
    faces = _find_boundary_faces(refined_area)
    bbox = BoundingBox.from_condition(refined_area)

    rcol, rrow, rlay = refinement.col, refinement.row, refinement.lay
    i0 = bbox.imin - 1
    j0 = bbox.jmin - 1
    k0 = bbox.kmin - 1

    # In the merged grid, grid2 (refined) starts after a 1-column gap:
    i_offset = coarse_ncol + 1

    rows: list[dict[str, int | str]] = []

    for outside_ijk, inside_ijk, face_dir in faces:
        # outside_ijk = mother cell (0-based in original/merged grid)
        mi, mj, mk = outside_ijk

        # inside_ijk = target cell (0-based in original grid) → cropped coords
        ci = inside_ijk[0] - i0
        cj = inside_ijk[1] - j0
        ck = inside_ijk[2] - k0

        # Determine direction from mother and which refined cells lie on the face.
        # face_dir is from the *inside* (target) cell's perspective;
        # the mother's perspective is the opposite sign.
        #
        # For I-faces: the varying refined indices are J and K  (rrow × rlay cells)
        # For J-faces: the varying refined indices are I and K  (rcol × rlay cells)
        # For K-faces: the varying refined indices are I and J  (rcol × rrow cells)
        ref_is: Iterable[int]
        ref_js: Iterable[int]
        ref_ks: Iterable[int]

        if face_dir == "i-":
            # Target at higher I than mother → mother's I+ face
            direction = "I+"
            ref_is = [ci * rcol]  # first i-column of refined block (I- face)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "i+":
            # Target at lower I than mother → mother's I- face
            direction = "I-"
            ref_is = [ci * rcol + rcol - 1]  # last i-column (I+ face)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "j-":
            # Target at higher J than mother → mother's J+ face
            direction = "J+"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = [cj * rrow]  # first j-row (J- face)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "j+":
            # Target at lower J than mother → mother's J- face
            direction = "J-"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = [cj * rrow + rrow - 1]  # last j-row (J+ face)
            ref_ks = range(ck * rlay, ck * rlay + rlay)
        elif face_dir == "k-":
            # Target at higher K than mother → mother's K+ face
            direction = "K+"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = [ck * rlay]  # first k-layer (K- face)
        elif face_dir == "k+":
            # Target at lower K than mother → mother's K- face
            direction = "K-"
            ref_is = range(ci * rcol, ci * rcol + rcol)
            ref_js = range(cj * rrow, cj * rrow + rrow)
            ref_ks = [ck * rlay + rlay - 1]  # last k-layer (K+ face)
        else:
            raise ValueError(f"Unexpected face direction: {face_dir!r}")

        for ri in ref_is:
            for rj in ref_js:
                for rk in ref_ks:
                    rows.append(
                        {
                            "I1": mi + 1,
                            "J1": mj + 1,
                            "K1": lmap1[mk] + 1,
                            "I2": ri + i_offset + 1,
                            "J2": rj + 1,
                            "K2": lmap2[rk] + 1,
                            "DIRECTION": direction,
                        }
                    )

    _logger.info(
        "NNC table: %d cell pairs from %d boundary faces", len(rows), len(faces)
    )
    return pd.DataFrame(rows, columns=["I1", "J1", "K1", "I2", "J2", "K2", "DIRECTION"])


def _set_actnum_in_grid(grid: xtgeo.Grid, active_mask: np.ndarray) -> None:
    """Deactivate cells where active_mask is False."""
    actnum = grid.get_actnum()
    actnum.values[~active_mask] = 0
    grid.set_actnum(actnum)


def _ijk_to_flat_index(
    i: np.ndarray, j: np.ndarray, k: np.ndarray, jdim: int, kdim: int
) -> np.ndarray:
    """Convert 0-based (i, j, k) indices to positions in a 1D flattened array.
    This follows NumPy C-order (row-major) indexing.
    """
    return i * jdim * kdim + j * kdim + k


def _group_by_parent_cell(
    parent_indices: np.ndarray, selected_mask: np.ndarray
) -> dict[int, np.ndarray]:
    """
    Group selected cell positions by parent-cell index.

    Returns a mapping from parent-cell index to the positions of selected
    cells belonging to that parent cell.
    """

    selected_parent_indices = parent_indices[selected_mask]
    selected_mask_indices = np.flatnonzero(selected_mask)

    if selected_parent_indices.size == 0:
        return {}

    # Sort so cells belonging to the same parent cell become contiguous.
    sort_order = np.argsort(selected_parent_indices, kind="stable")

    sorted_parent_indices = selected_parent_indices[sort_order]
    sorted_selected_mask_indices = selected_mask_indices[sort_order]

    # Split whenever the parent-cell index changes.
    cuts = np.flatnonzero(np.diff(sorted_parent_indices)) + 1
    selected_indices_by_parent_cell = np.split(sorted_selected_mask_indices, cuts)

    # Get the parent-cell index for each group.
    parent_idx_keys = sorted_parent_indices[np.r_[0, cuts]]

    return dict(zip(parent_idx_keys, selected_indices_by_parent_cell))


def _cell_numbers_to_flat_indices(
    cell_numbers: np.ma.MaskedArray, grid_indexer: Any
) -> np.ma.MaskedArray:
    """Convert RMS active-cell numbers to parent-grid flat indices."""

    mask = np.ma.getmaskarray(cell_numbers)

    active_cell_numbers = cell_numbers[~mask].astype(np.uint64)
    i, j, k = grid_indexer.get_indices(active_cell_numbers).T  # 0-based
    _, nrow, nlay = grid_indexer.dimensions

    flat_indices = np.ma.masked_all(cell_numbers.shape, dtype=np.int64)
    flat_indices[~mask] = _ijk_to_flat_index(i=i, j=j, k=k, jdim=nrow, kdim=nlay)
    return flat_indices


def _read_grid_faults(
    project: Any,
    grid_name: str,
) -> list[GridFault]:
    """Read RMS faults and keep FW/HW parent-cell indices as masked IJK arrays."""
    rms_grid = project.grid_models[grid_name].get_grid()
    indexer = rms_grid.grid_indexer

    faults = []

    for rms_fault in rms_grid.grid_faults:
        values = rms_fault.get_values()

        fw_cells, hw_cells, fw_face, hw_face = values.T

        faults.append(
            GridFault(
                name=rms_fault.name,
                fw_indices=_cell_numbers_to_flat_indices(fw_cells, indexer),
                hw_indices=_cell_numbers_to_flat_indices(hw_cells, indexer),
                fw_face=fw_face,
                hw_face=hw_face,
            )
        )

    return faults


def _write_grid_faults(
    project: Any,
    grid_name: str,
    grid: xtgeo.Grid,
    faults: list[GridFault],
    nested_cells_by_parent: dict[int, np.ndarray],
) -> None:
    """Write nested-grid faults to RMS using parent-cell mapping properties."""
    rms_grid = project.grid_models[grid_name].get_grid()
    indexer = rms_grid.grid_indexer

    # RMS fault cell numbers refer to active-cell IDs.
    flat_to_rms_cell_number = _create_flat_to_rms_cell_number_mapping(indexer)

    for fault in faults:
        values = _map_fault_to_nested_grid(
            fault=fault,
            nested_cells_by_parent=nested_cells_by_parent,
            flat_to_rms_cell_number=flat_to_rms_cell_number,
            nested_dimensions=grid.dimensions,
        )

        rms_fault = rms_grid.grid_faults.create(fault.name)
        rms_fault.set_values(values)


def _map_fault_to_nested_grid(
    fault: GridFault,
    nested_cells_by_parent: dict[int, np.ndarray],
    flat_to_rms_cell_number: np.ma.MaskedArray,
    nested_dimensions: Dimensions,
) -> np.ma.MaskedArray:
    """Map a parent-grid fault onto nested-grid fault faces."""

    rows = []

    for fw_idx, hw_idx, fw_face, hw_face in zip(
        fault.fw_indices,
        fault.hw_indices,
        fault.fw_face,
        fault.hw_face,
    ):
        for parent_idx, face, cell_column, face_column in (
            (fw_idx, fw_face, 0, 2),
            (hw_idx, hw_face, 1, 3),
        ):
            if np.ma.is_masked(parent_idx) or np.ma.is_masked(face):
                continue

            cells = nested_cells_by_parent.get(parent_idx)

            if cells is None:
                raise ValueError(
                    f"No nested cells found for parent cell index {parent_idx}."
                )

            nested_cells = _get_nested_fault_cells(
                cells=cells,
                face=face,
                nested_dimensions=nested_dimensions,
            )

            rms_cells = flat_to_rms_cell_number[nested_cells]

            side_rows = np.ma.masked_all((len(rms_cells), 4), dtype=np.uint64)
            side_rows[:, cell_column] = rms_cells
            side_rows[:, face_column] = face

            rows.append(side_rows)

    return np.ma.vstack(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class NestedHybridGrid:
    """Class to build a nested hybrid grid with a refined target region.

    The target region is refined and merged back into the coarse grid.
    A **NNC mapping table** is computed that lists every coarse ↔ refined cell
    pair that should be connected by a Non-Neighbour Connection (NNC).
    """

    def __init__(
        self,
        coarse_grid: xtgeo.Grid,
        region: xtgeo.GridProperty,
        refinement: tuple[int, int, int],
        target_region_id: int = 1,
    ) -> None:
        """Create a NestedHybridGrid instance."""

        self._validate_inputs(coarse_grid, region, refinement, target_region_id)

        self._nnc_table: pd.DataFrame | None = None
        self._parent_faults: dict[str, GridFault] | None = None

        self._parent_grid = coarse_grid
        self._parent_dimensions = coarse_grid.dimensions
        self._parent_subgrids = coarse_grid.subgrids
        self._parent_region = region

        self._target_region_id = target_region_id
        self._refinement = Refinement.from_tuple(refinement)

        self._refined_area = region.values == target_region_id
        self._refined_bbox = BoundingBox.from_condition(self._refined_area)
        self._refined_nlay = self._get_num_refined_layers()

        self._layer_map_coarse = self._generate_layer_map_coarse()
        self._layer_map_refined = self._generate_layer_map_refined()

        self._grid = self._build_nested_hybrid_grid()

        self._nested_cells_by_parent = self._create_nested_cells_by_parent()

    @classmethod
    def from_rms(
        cls,
        project: Any,
        grid_name: str,
        region_name: str,
        refinement: tuple[int, int, int],
        properties: list[str] | None = None,
        *,
        target_region_id: int = 1,
    ) -> Self:
        """Create a NestedHybridGrid instance from an RMS project.

        Args:
            project: RMS project instance.
            grid_name: Name of the grid in the RMS project.
            region_name: Name of the region property in the RMS project.
            refinement: Refinement factors as (ncol, nrow, nlay).
            properties: Optional list of property names to load from the project.
            target_region_id: Region value to refine (default: 1).
        """

        parent_grid = xtgeo.grid_from_roxar(project, grid_name)
        region = xtgeo.gridproperty_from_roxar(project, grid_name, region_name)

        for propname in properties or []:
            prop = xtgeo.gridproperty_from_roxar(project, grid_name, propname)
            parent_grid.append_prop(prop)

        instance = cls(parent_grid, region, refinement, target_region_id)

        instance._parent_faults = _read_grid_faults(
            project=project, grid_name=grid_name
        )
        return instance

    def to_rms(self, project: Any, grid_name: str) -> None:
        """Write the nested hybrid grid and its properties to an RMS project."""
        self.grid.to_roxar(project, grid_name)

        for prop in self.properties:
            if prop.name not in ORIGINAL_IJK_PROPERTY_NAMES:
                prop.to_roxar(project, grid_name, prop.name)

        if self._parent_faults is not None:
            _write_grid_faults(
                project=project,
                grid_name=grid_name,
                grid=self.grid,
                faults=self._parent_faults,
                nested_cells_by_parent=self._nested_cells_by_parent,
            )

    @staticmethod
    def _validate_inputs(
        parent_grid: xtgeo.Grid,
        region: xtgeo.GridProperty,
        refinement: tuple[int, int, int],
        target_region_id: int,
    ) -> None:
        """Validate input arguments."""
        if region.dimensions != parent_grid.dimensions:
            raise ValueError(
                f"Region property dimensions {region.dimensions} do not match "
                f"input grid dimensions {parent_grid.dimensions}"
            )

        if not (region.values == target_region_id).any():
            raise ValueError(
                f"No cells found for target_region_id={target_region_id} "
                f"in region property {region.name}"
            )

        if not isinstance(refinement, tuple) or len(refinement) != 3:
            raise ValueError(
                "Refinement must be a tuple of three integers: (rcol, rrow, rlay)"
            )

    def _build_nested_hybrid_grid(self) -> xtgeo.Grid:
        """Build the nested hybrid grid."""
        coarse_grid = self._parent_grid.copy()

        region_name = self._parent_region.name
        if region_name not in coarse_grid.propnames:
            coarse_grid.append_prop(self._parent_region)
        for prop in coarse_grid.get_ijk(names=ORIGINAL_IJK_PROPERTY_NAMES):
            coarse_grid.append_prop(prop)

        # Create the refined grid, i.e. crop and refine.
        refined_grid = _crop_for_region(coarse_grid, self._refined_bbox)
        refined_grid.refine(
            self._refinement.col, self._refinement.row, self._refinement.lay
        )

        # Deactivate outside the target region in the refined grid
        refined_region = refined_grid.get_prop_by_name(region_name)
        active_area = refined_region.values == self._target_region_id
        _set_actnum_in_grid(refined_grid, active_area)

        # Deactivate the target region in the coarse grid
        coarse_region = coarse_grid.get_prop_by_name(region_name)
        active_area = coarse_region.values != self._target_region_id
        _set_actnum_in_grid(coarse_grid, active_area)

        grid = xtgeo.grid_merge(
            grid1=coarse_grid,
            grid2=refined_grid,
            layer_map1=self._layer_map_coarse,
            layer_map2=self._layer_map_refined,
        )
        _logger.info("Merged grid dimensions: %s", grid.dimensions)

        grid.subgrids = self._set_zonation(grid.nlay)
        return grid

    @property
    def grid(self) -> xtgeo.Grid:
        """The final nested hybrid grid."""
        return self._grid

    @property
    def properties(self) -> list[xtgeo.GridProperty]:
        """The final nested hybrid grid properties."""
        return self.grid.props

    @property
    def nnc_table(self) -> pd.DataFrame:
        """Non-Neighbour Connection (NNC) mapping table."""
        if self._nnc_table is None:
            self._nnc_table = self._compute_nnc_table()
        return self._nnc_table

    def write_nnc_table(self, filename: str | Path) -> None:
        """Write the NNC mapping table to CSV."""
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.nnc_table.to_csv(output_path, index=False)

    def _compute_nnc_table(self) -> pd.DataFrame:
        """Compute the NNC mapping table."""
        return _compute_nnc_table(
            refined_area=self._refined_area,
            refinement=self._refinement,
            coarse_ncol=self._parent_dimensions.ncol,
            lmap1=self._layer_map_coarse,
            lmap2=self._layer_map_refined,
        )

    def _generate_layer_map_coarse(self) -> np.ndarray:
        """Map original coarse-layer indices to indices in the final merged grid.

        Layers below the refined K-window are unchanged; layers at/above the window are
        shifted to account for extra layers introduced by K-refinement.
        """
        rlay = self._refinement.lay
        k0 = self._refined_bbox.kmin - 1
        parent_nlay = self._parent_dimensions.nlay

        lmap = np.arange(parent_nlay, dtype=np.int32)
        return lmap + np.where(
            lmap < k0,
            0,
            (rlay - 1) * np.minimum(int(self._refined_nlay / rlay), lmap - k0),
        )

    def _generate_layer_map_refined(self) -> np.ndarray:
        """Map original refined grid layer indices to indices in the final merged grid.

        Refined layers are offset by the first layer index of the refined
        K-window in the original coarse grid.
        """
        lmap = np.arange(self._refined_nlay, dtype=np.int32)
        return lmap + self._refined_bbox.kmin - 1

    def _get_num_refined_layers(self) -> int:
        """Get the number of layers of the refined grid."""
        bbox = self._refined_bbox
        return (bbox.kmax - bbox.kmin + 1) * self._refinement.lay

    def _set_zonation(self, nlay: int) -> dict | None:
        """Create an updated subgrid dictionary for the merged grid."""
        subgrid = self._parent_subgrids
        if subgrid is None:
            return None

        lmap = self._layer_map_coarse

        updated_subgrid = {}

        # sorted list of zones to add
        zl = sorted(subgrid, key=lambda x: subgrid[x][0])

        for zi in range(len(zl)):
            zn = zl[zi]
            zmin = lmap[subgrid[zn][0] - 1] + 1
            zmax = (
                nlay + 1 if zi == len(zl) - 1 else lmap[subgrid[zl[zi + 1]][0] - 1] + 1
            )
            updated_subgrid[zn] = range(zmin, zmax)

        return updated_subgrid

    def _create_nested_cells_by_parent(self) -> dict[int, np.ndarray]:
        """Group active merged-grid cells by flattened parent-grid index."""
        parent_i = self._grid.get_prop_by_name("parent_I")
        parent_j = self._grid.get_prop_by_name("parent_J")
        parent_k = self._grid.get_prop_by_name("parent_K")

        parent_indices = _ijk_to_flat_index(
            i=parent_i.values - 1,
            j=parent_j.values - 1,
            k=parent_k.values - 1,
            jdim=self._parent_dimensions.nrow,
            kdim=self._parent_dimensions.nlay,
        )

        valid = ~np.ma.getmaskarray(parent_indices)
        return _group_by_parent_cell(parent_indices=parent_indices, selected_mask=valid)


def create_nested_hybrid_grid(
    grid: xtgeo.Grid,
    region: xtgeo.GridProperty,
    target_region_id: int,
    refinement: tuple[int, int, int],
) -> tuple[
    xtgeo.Grid,
    pd.DataFrame,
]:
    """Create a nested hybrid grid by refining one region and merging it back.

    The cells belonging to *target_region_id* are replaced by a refined
    (subdivided) version of the same region.

    A **NNC mapping table** is returned that lists every
    mother ↔ refined cell pair that should be connected by a Non-Neighbour
    Connection (NNC).  The table is derived from the topological knowledge
    available at merge time (which original cell was refined and how its
    sub-cells map into the merged grid).

    The table columns are:

    - ``I1, J1, K1``: mother cell indices (1-based) in the merged grid.
    - ``I2, J2, K2``: refined cell indices (1-based) in the merged grid.
    - ``DIRECTION``: face direction from the mother cell's perspective
      (``I+``, ``I-``, ``J+``, ``J-``, ``K+``, ``K-``).

    This table can be passed to
    :meth:`xtgeo.Grid.get_transmissibilities` to compute NNC
    transmissibilities for the specified cell pairs.

    Args:
        grid: The original coarse grid.
        region: A :class:`xtgeo.GridProperty` whose values identify the
            regions (e.g. an integer region parameter).
        target_region_id: The region value to refine.
        refinement: ``(ncol, nrow, nlay)`` refinement factors.

    Returns:
        A tuple ``(merged_grid, nnc_table)`` where *merged_grid*
        is a new :class:`xtgeo.Grid` with the refined region stitched back into
        the coarse grid and *nnc_table* is a :class:`pandas.DataFrame` mapping
        mother cells to their connected refined cells.
    """
    warnings.warn(
        "create_nested_hybrid_grid is currently experimental. It may undergo "
        "breaking changes in future versions without notice.",
        FutureWarning,
    )

    nhg = NestedHybridGrid(
        coarse_grid=grid,
        region=region,
        refinement=refinement,
        target_region_id=target_region_id,
    )

    return nhg.grid, nhg.nnc_table


def nnc_to_gridproperty(
    grid: xtgeo.Grid,
    nnc_df: pd.DataFrame,
) -> tuple[xtgeo.GridProperty, xtgeo.GridProperty, xtgeo.GridProperty]:
    """Convert NNC transmissibility data to three GridProperty instances.

    Takes the NNC DataFrame produced by :meth:`xtgeo.Grid.get_transmissibilities`
    and maps transmissibility values onto grid cells, producing one property per
    direction (I, J, K).

    For rows where DIRECTION contains ``"+"``, the transmissibility value is
    placed in cell ``(I1, J1, K1)``.  For rows where DIRECTION contains
    ``"-"``, the value is placed in cell ``(I2, J2, K2)``.  Index columns
    (I1, J1, K1, I2, J2, K2) are expected to be **1-based**.

    If multiple rows map to the same cell and direction, the transmissibility
    values are summed (parallel flow paths are additive).

    Args:
        grid: The xtgeo Grid that defines the geometry.
        nnc_df: A DataFrame with at least columns
            ``I1, J1, K1, I2, J2, K2, T, DIRECTION``.

    Returns:
        A tuple ``(tranx_nnc, trany_nnc, tranz_nnc)`` of
        :class:`xtgeo.GridProperty` instances named ``"TRANX_NNC"``,
        ``"TRANY_NNC"``, and ``"TRANZ_NNC"`` respectively.
        Cells without an NNC value are set to ``-1.0``.
    """
    required_cols = {"I1", "J1", "K1", "I2", "J2", "K2", "T", "DIRECTION"}
    missing = required_cols - set(nnc_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in nnc_df: {missing}")

    ncol, nrow, nlay = grid.ncol, grid.nrow, grid.nlay
    fill = -1.0

    arrays = {
        "I": np.zeros((ncol, nrow, nlay), dtype=np.float64),
        "J": np.zeros((ncol, nrow, nlay), dtype=np.float64),
        "K": np.zeros((ncol, nrow, nlay), dtype=np.float64),
    }
    touched = {
        "I": np.zeros((ncol, nrow, nlay), dtype=bool),
        "J": np.zeros((ncol, nrow, nlay), dtype=bool),
        "K": np.zeros((ncol, nrow, nlay), dtype=bool),
    }

    direction_col = nnc_df["DIRECTION"].astype(str)
    is_plus = direction_col.str.contains(r"\+", regex=True)
    is_minus = direction_col.str.contains("-")
    prefix_col = direction_col.str[0].str.upper()

    for prefix in ("I", "J", "K"):
        arr = arrays[prefix]
        tch = touched[prefix]
        mask_prefix = prefix_col == prefix

        # "+" rows → use (I1, J1, K1)
        sel_plus = nnc_df.loc[mask_prefix & is_plus]
        if not sel_plus.empty:
            ii = sel_plus["I1"].values.astype(int) - 1
            jj = sel_plus["J1"].values.astype(int) - 1
            kk = sel_plus["K1"].values.astype(int) - 1
            tt = sel_plus["T"].values.astype(float)
            valid = (
                (ii >= 0)
                & (ii < ncol)
                & (jj >= 0)
                & (jj < nrow)
                & (kk >= 0)
                & (kk < nlay)
            )
            np.add.at(arr, (ii[valid], jj[valid], kk[valid]), tt[valid])
            tch[ii[valid], jj[valid], kk[valid]] = True

        # "-" rows → use (I2, J2, K2)
        sel_minus = nnc_df.loc[mask_prefix & is_minus]
        if not sel_minus.empty:
            ii = sel_minus["I2"].values.astype(int) - 1
            jj = sel_minus["J2"].values.astype(int) - 1
            kk = sel_minus["K2"].values.astype(int) - 1
            tt = sel_minus["T"].values.astype(float)
            valid = (
                (ii >= 0)
                & (ii < ncol)
                & (jj >= 0)
                & (jj < nrow)
                & (kk >= 0)
                & (kk < nlay)
            )
            np.add.at(arr, (ii[valid], jj[valid], kk[valid]), tt[valid])
            tch[ii[valid], jj[valid], kk[valid]] = True

        # Set untouched cells to fill value
        arr[~tch] = fill

    prop_names = {"I": "TRANX_NNC", "J": "TRANY_NNC", "K": "TRANZ_NNC"}
    props = {}
    for prefix in ("I", "J", "K"):
        props[prefix] = xtgeo.GridProperty(
            grid,
            name=prop_names[prefix],
            values=np.ma.array(arrays[prefix]),
            discrete=False,
        )

    return props["I"], props["J"], props["K"]


def nnc_to_flowsimulator_input(
    nnc_df: pd.DataFrame,
    filepath: str | os.PathLike[str],
) -> None:
    """Write NNC transmissibilities to a flow-simulator input file.

    Produces a file with the ``NNC`` keyword suitable for reservoir
    simulators that use Eclipse-style input decks, such as Eclipse and
    OPM Flow.  The file can be included in the deck via ``INCLUDE``.
    Each row of *nnc_df* becomes one NNC record with the six cell
    indices and the transmissibility value.

    Args:
        nnc_df: A DataFrame with at least columns
            ``I1, J1, K1, I2, J2, K2, T``.  Optional columns ``TYPE``
            and ``DIRECTION`` are written as end-of-line comments.
        filepath: Path to the output file.
    """
    required_cols = {"I1", "J1", "K1", "I2", "J2", "K2", "T"}
    missing = required_cols - set(nnc_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in nnc_df: {missing}")

    has_type = "TYPE" in nnc_df.columns
    has_dir = "DIRECTION" in nnc_df.columns

    with open(filepath, "w") as f:
        f.write("NNC\n")
        for _, row in nnc_df.iterrows():
            line = (
                f"    {int(row['I1']):>4} {int(row['J1']):>4} {int(row['K1']):>4}"
                f"    {int(row['I2']):>4} {int(row['J2']):>4} {int(row['K2']):>4}"
                f"   {row['T']:.6f}  /"
            )
            comment_parts = []
            if has_type:
                comment_parts.append(str(row["TYPE"]))
            if has_dir:
                comment_parts.append(str(row["DIRECTION"]))
            if comment_parts:
                line += "  -- " + " ".join(comment_parts)
            f.write(line + "\n")
        f.write("/\n")

    _logger.info("NNC keyword written to %s", filepath)
