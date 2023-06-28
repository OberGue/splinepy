try:
    from . import common as c
except BaseException:
    import common as c


class PermuteParametricAxesTest(c.SplineBasedTestCase):
    @c.pytest.mark.integration_test
    def test_permute_parametric_axes(self):
        """
        test permute
        """
        # Define some splines
        z = self.bezier_3d
        r = self.rational_3d
        b = self.bspline_3d
        n = self.nurbs_3d
        originals = (z, r, b, n)

        # define permutation
        permutation = [2, 0, 1]

        # return permuted
        for orig in originals:
            # make more work
            orig.elevate_degrees(1)
            orig.elevate_degrees(2)
            orig.elevate_degrees(2)
            if "knot_vectors" in orig.required_properties:
                orig.insert_knots(0, [0.4, 0.7, 0.8])
                orig.insert_knots(1, [0.1, 0.2])
                orig.insert_knots(2, [0.3, 0.5, 0.6, 0.9])

            perm = c.splinepy.spline.permute_parametric_axes(
                orig, permutation, inplace=False
            )
            queries = c.np.asarray(c.q3D)

            self.assertTrue(
                c.np.allclose(
                    orig.evaluate(queries),
                    perm.evaluate(queries[:, permutation]),
                ),
                f"{perm.whatami} failed to permute.",
            )

        # ic.nplace
        for orig in originals:
            perm = orig.copy()
            c.splinepy.spline.permute_parametric_axes(
                perm, permutation, inplace=True
            )
            queries = c.np.asarray(c.q3D)

            self.assertTrue(
                c.np.allclose(
                    orig.evaluate(queries),
                    perm.evaluate(queries[:, permutation]),
                ),
                f"{perm.whatami} failed to permute inplace.",
            )


if __name__ == "__main__":
    c.unittest.main()
