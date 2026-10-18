"""
Unstructured FIT meshes: topology, circumcentric dual and mesh generators.

The discrete identities C G = 0 and D C = 0 are exact (integer incidence
matrices); the signed circumcentric dual measures tile the domain exactly
(sum_e A_e L_e / 3 = volume in 3D, sum_n A_n = area in 2D) for any mesh,
and are non-negative on interior objects of a Delaunay mesh.
"""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from wakis.unstructured import (
    SimplexMesh,
    ball_mesh,
    disk_mesh,
    gmsh_ball_mesh,
)


@pytest.fixture(scope="module")
def random_meshes():
    rng = np.random.default_rng(1)
    p2 = rng.random((150, 2))
    p3 = rng.random((200, 3))
    return (
        SimplexMesh(p2, Delaunay(p2).simplices),
        SimplexMesh(p3, Delaunay(p3).simplices),
    )


class TestTopology:
    def test_exact_sequence(self, random_meshes):
        """curl grad = 0 and div curl = 0 hold exactly on the incidences."""
        m2, m3 = random_meshes
        assert abs(m2.C @ m2.G).max() == 0
        assert abs(m3.C @ m3.G).max() == 0
        assert abs(m3.D @ m3.C).max() == 0
        # Euler characteristic of a ball: V - E + F - T = 1
        assert m3.n_nodes - m3.n_edges + m3.n_faces - m3.n_cells == 1
        assert m2.n_nodes - m2.n_edges + m2.n_cells == 1

    def test_dual_tiles_domain(self, random_meshes):
        """Signed dual measures sum to the mesh volume (any triangulation)."""
        m2, m3 = random_meshes
        area = m2.cell_volume.sum()
        assert m2.dual_measure().sum() == pytest.approx(area, rel=1e-12)
        assert (m2.dual_length() * m2.edge_length).sum() / 2 == pytest.approx(
            area, rel=1e-12
        )
        vol = m3.cell_volume.sum()
        assert (m3.dual_measure() * m3.edge_length).sum() / 3 == pytest.approx(
            vol, rel=1e-12
        )
        assert (m3.dual_length() * m3.face_area).sum() / 3 == pytest.approx(
            vol, rel=1e-12
        )


class TestDiagnostics:
    def test_delaunay_is_nonnegative(self, random_meshes):
        """Interior dual measures of a Delaunay mesh are >= 0."""
        for m in random_meshes:
            rep = m.diagnostics()
            assert rep["negative_dual_length"] == 0
            if m.dim == 3:
                assert rep["negative_dual_area"] == 0

    def test_non_delaunay_reported(self):
        """Two flat triangles on a long diagonal violate the Delaunay
        (empty circle) property: the diagonal gets a negative dual edge."""
        nodes = [[-1, 0], [1, 0], [0, 0.2], [0, -0.2]]
        m = SimplexMesh(nodes, [[0, 1, 2], [0, 1, 3]])
        rep = m.diagnostics()
        assert rep["negative_dual_length"] == 1
        assert rep["not_well_centered"] == 2


class TestMeshers:
    def test_disk_mesh_conforming(self):
        """Ring nodes on every interface; area-preserving polygons."""
        m = disk_mesh([0.5, 1.0], 0.1)
        inner = m.cell_volume[m.cell_tags == 0].sum()
        assert inner == pytest.approx(np.pi * 0.25, rel=1e-12)
        assert m.cell_volume.sum() == pytest.approx(np.pi, rel=1e-12)
        r = np.hypot(*m.nodes[m.cells].transpose(2, 0, 1))
        crossing = (r.min(1) < 0.49) & (r.max(1) > 0.51)
        assert not crossing.any()
        assert m.diagnostics()["negative_dual_area"] == 0

    def test_ball_mesh_conforming(self):
        """Shells conform (the material interface is made of mesh faces
        with all nodes on one sphere, enlarged so that the polyhedron keeps
        the volume of the r = b sphere) and the interior dual measures of
        the Delaunay mesh are non-negative."""
        b = 0.5
        m = ball_mesh([b, 1.0], 0.3)
        inner_nodes = np.unique(m.cells[m.cell_tags == 0])
        outer_nodes = np.unique(m.cells[m.cell_tags == 1])
        shared = np.intersect1d(inner_nodes, outer_nodes)
        rs = np.linalg.norm(m.nodes[shared], axis=1)
        assert np.ptp(rs) < 1e-12  # interface nodes all on one sphere
        inner = m.cell_volume[m.cell_tags == 0].sum()
        assert inner == pytest.approx(4 / 3 * np.pi * b**3, rel=1e-2)
        # without the correction: inscribed polyhedron, O(h^2) deficit
        m0 = ball_mesh([b, 1.0], 0.3, preserve_volume=False)
        inner0 = m0.cell_volume[m0.cell_tags == 0].sum()
        assert 0.8 < inner0 / (4 / 3 * np.pi * b**3) < 0.95
        rep = m.diagnostics()
        assert rep["negative_dual_area"] == 0
        assert rep["negative_dual_length"] == 0

    def test_meshio_roundtrip(self, tmp_path):
        pytest.importorskip("meshio")
        m = disk_mesh([0.5, 1.0], 0.2)
        fn = str(tmp_path / "disk.vtu")
        m.write(fn)
        m2 = SimplexMesh.read(fn)
        assert np.array_equal(m2.cell_tags, m.cell_tags)
        assert m2.n_edges == m.n_edges

    def test_gmsh_is_optional(self, tmp_path):
        """gmsh is an optional external tool: a clear ImportError if absent."""
        try:
            import gmsh  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="optional external"):
                gmsh_ball_mesh([0.5, 1.0], 0.3, str(tmp_path / "b.msh"))
            return
        m = gmsh_ball_mesh([0.5, 1.0], 0.3, str(tmp_path / "b.msh"))
        assert set(np.unique(m.cell_tags)) == {0, 1}
