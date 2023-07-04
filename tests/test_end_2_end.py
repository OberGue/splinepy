try:
    from . import common as c
except BaseException:
    import common as c

import numpy as np

import splinepy


class SplineTest(c.unittest.TestCase):
    @c.pytest.mark.e2e_test
    def test_something(self):
        # Initialize bspline with any array-like input
        bspline = splinepy.BSpline(
            degrees=[2, 1],
            knot_vectors=[
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            control_points=[
                [0.0, 0.0],  # [0, 0] (control grid index)
                [0.5, 0.0],  # [1, 0]
                [1.0, 0.0],  # [2, 0]
                [0.0, 1.0],  # [0, 1]
                [0.5, 1.0],  # [1, 1]
                [1.0, 1.0],  # [2, 1]
            ],
        )

        multi_index = bspline.multi_index
        grid_cps = np.empty(bspline.control_points.shape)
        grid_cps[multi_index[0, 0]] = [0.0, 0.0]
        grid_cps[multi_index[1, 0]] = [0.5, 0.0]
        grid_cps[multi_index[2, 0], 0] = 1.0
        # which also supports ranges
        grid_cps[multi_index[:, 0], 1] = 0.0
        grid_cps[multi_index[:, 1], 1] = 1.0
        grid_cps[multi_index[:, 1], 0] = [0.0, 0.5, 1.0]

        assert np.allclose(bspline.control_points, grid_cps)

        # Evaluate spline mapping.
        queries = [
            [0.1, 0.2],  # first query
            [0.4, 0.5],  # second query
            [0.1156, 0.9091],  # third query
        ]
        physical_coords = bspline.evaluate(queries)

        # execute this in parallel using multithread
        physical_coords_parallel = bspline.evaluate(queries, nthreads=2)

        # this should hold
        assert np.allclose(physical_coords, physical_coords_parallel)


if __name__ == "__main__":
    c.unittest.main()
