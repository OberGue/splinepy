import numpy as np

try:
    from . import common as c
except BaseException:
    import common as c


class GetPropertyTest(c.SplineBasedTestCase):
    def test_unique_knots(self):
        for spline in (
            self.bspline_2p2d(),
            self.bspline_3p3d(),
            self.nurbs_2p2d(),
            self.nurbs_3p3d(),
        ):
            copy_knot_vectors = spline.knot_vectors[:]
            unique_knots = [np.unique(ckvs) for ckvs in copy_knot_vectors]

            for uk, uk_fct in zip(unique_knots, spline.unique_knots):
                assert c.np.allclose(uk, uk_fct)

    def test_knot_multiplicities(self):
        for spline in (
            self.bspline_2p2d(),
            self.bspline_3p3d(),
            self.nurbs_2p2d(),
            self.nurbs_3p3d(),
        ):
            copy_knot_vectors = spline.knot_vectors[:]
            multiplicity = [
                np.unique(ckvs, return_counts=True)
                for ckvs in copy_knot_vectors
            ]

            for m_u, m_fct in zip(multiplicity, spline.knot_multiplicities):
                assert c.np.allclose(m_u[1], m_fct)

            for u_kv, kn_m, spl_kv in zip(
                spline.unique_knots,
                spline.knot_multiplicities,
                spline.knot_vectors,
            ):
                assert c.np.allclose(np.array(spl_kv), np.repeat(u_kv, kn_m))


if __name__ == "__main__":
    c.unittest.main()
