try:
    from . import common as c
except BaseException:
    import common as c


class TestSplinepyKnotVectorManipulation(c.unittest.TestCase):
    def setUp(self):
        self.b2P2D = c.b2P2D.copy()
        self.n2P2D = c.n2P2D.copy()
        self.bspline = c.splinepy.BSpline(**self.b2P2D)
        self.nurbs = c.splinepy.NURBS(**self.n2P2D)
        self.ref_bspline = c.splinepy.BSpline(**c.b2P2D)
        self.ref_nurbs = c.splinepy.NURBS(**c.n2P2D)

    def test_insert_knot(self):
        """Test the knot insertion function (.insert_knot())."""

        # reference solutions
        bspline_ref_kv = [
            [0, 0, 0, 0.2, 0.5, 0.7, 1, 1, 1],
            [0, 0, 0, 0.2, 0.5, 0.7, 1, 1, 1],
        ]
        nurbs_ref_kv = [
            [0, 0, 0, 0.2, 0.5, 0.7, 1, 1, 1],
            [0, 0, 0.2, 0.5, 0.7, 1, 1],
        ]

        print(self.nurbs.cps)
        # insert knots
        self.bspline.insert_knots(0, [0.2, 0.7])
        self.bspline.insert_knots(
            1,
            [
                0.2,
                0.5,
                0.7,
            ],
        )

        self.nurbs.insert_knots(
            0,
            [
                0.2,
                0.5,
                0.7,
            ],
        )
        self.nurbs.insert_knots(
            1,
            [
                0.2,
                0.5,
                0.7,
            ],
        )

        # test knot_vectors
        self.assertTrue(
            c.are_items_close(self.bspline.knot_vectors, bspline_ref_kv)
        )
        self.assertTrue(
            c.are_items_close(self.nurbs.knot_vectors, nurbs_ref_kv)
        )

        # use random query points
        q2D = c.np.random.rand(10, 2)

        # test evaluation
        assert c.np.allclose(
            self.bspline.evaluate(q2D), self.ref_bspline.evaluate(q2D)
        )

        print(self.nurbs.cps)

        assert c.np.allclose(
            self.nurbs.evaluate(q2D), self.ref_nurbs.evaluate(q2D)
        )

    def test_insert_knot_with_matrix(self):
        """Test the knot insertion function (.insert_knot())."""

        # BSpline Data
        b_spline_knots_0 = c.np.random.rand(8)
        b_spline_knots_1 = c.np.random.rand(9)
        matrix_bspline = self.bspline.knot_insertion_matrix(
            0, b_spline_knots_0
        )
        self.bspline.insert_knots(
            0,
            b_spline_knots_0,
        )
        matrix_bspline = (
            self.bspline.knot_insertion_matrix(1, b_spline_knots_1)
            @ matrix_bspline
        )
        self.bspline.insert_knots(1, b_spline_knots_1)

        # NURBS Data
        nurbs_knots_0 = c.np.random.rand(10)
        nurbs_knots_1 = c.np.random.rand(11)
        matrix_nurbs = self.nurbs.knot_insertion_matrix(0, nurbs_knots_0)
        self.nurbs.insert_knots(0, nurbs_knots_0)
        matrix_nurbs = (
            self.nurbs.knot_insertion_matrix(1, nurbs_knots_1) @ matrix_nurbs
        )
        self.nurbs.insert_knots(1, nurbs_knots_1)

        # use random query points
        q2D = c.np.random.rand(50, 2)

        # test evaluation
        self.assertTrue(
            c.np.allclose(
                self.bspline.evaluate(q2D), self.ref_bspline.evaluate(q2D)
            )
        )
        self.assertTrue(
            c.np.allclose(
                self.nurbs.evaluate(q2D), self.ref_nurbs.evaluate(q2D)
            )
        )

        # Test control points and weights
        self.assertTrue(
            c.np.allclose(
                self.bspline.control_points,
                matrix_bspline @ self.ref_bspline.control_points,
            )
        )
        self.assertTrue(
            c.np.allclose(
                self.nurbs.weights, matrix_nurbs @ self.ref_nurbs.weights
            )
        )
        self.assertTrue(
            c.np.allclose(
                self.nurbs.control_points,
                matrix_nurbs
                @ (self.ref_nurbs.weights * self.ref_nurbs.control_points)
                / self.nurbs.weights,
            )
        )

    def test_remove_knot(self):
        """Test the function .remove_knots.
        This test also depends on the function .insert_knots!"""

        # insert and remove knots
        self.bspline.insert_knots(
            0,
            [
                0.2,
                0.7,
            ],
        )
        self.bspline.insert_knots(
            1,
            [
                0.2,
                0.5,
                0.7,
            ],
        )
        self.bspline.remove_knots(0, [0.2, 0.7])
        self.bspline.remove_knots(
            1,
            [
                0.2,
                0.5,
                0.7,
            ],
        )

        self.nurbs.insert_knots(
            0,
            [
                0.2,
                0.5,
                0.7,
            ],
        )
        self.nurbs.insert_knots(
            1,
            [
                0.2,
                0.5,
                0.7,
            ],
        )
        self.nurbs.remove_knots(
            0,
            [
                0.2,
                0.5,
                0.7,
            ],
        )
        self.nurbs.remove_knots(
            1,
            [
                0.2,
                0.5,
                0.7,
            ],
        )

        # test knot_vectors
        self.assertTrue(
            c.are_items_close(
                self.bspline.knot_vectors, self.ref_bspline.knot_vectors
            )
        )
        self.assertTrue(
            c.are_items_close(
                self.nurbs.knot_vectors, self.ref_nurbs.knot_vectors
            )
        )

        # use random query points
        q2D = c.np.random.rand(10, 2)

        # test evaluation
        self.assertTrue(
            c.np.allclose(
                self.bspline.evaluate(q2D), self.ref_bspline.evaluate(q2D)
            )
        )
        self.assertTrue(
            c.np.allclose(
                self.nurbs.evaluate(q2D), self.ref_nurbs.evaluate(q2D)
            )
        )


if __name__ == "__main__":
    c.unittest.main()
