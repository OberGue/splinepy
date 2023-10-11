import pytest

import splinepy

try:
    from tests import common as c
except BaseException:
    import common as c


def foo(x):
    """
    Parametrization Function (determines thickness)
    """
    return (x[:, 0] * 0.05 + x[:, 1] * 0.05 + x[:, 2] * 0.1 + 0.1).reshape(
        -1, 1
    )


def create_microstructure_simple():
    generator = splinepy.microstructure.Microstructure()
    generator.deformation_function = splinepy.Bezier(
        degrees=[2, 1],
        control_points=[[0, 0], [1, 0], [2, -1], [-1, 1], [1, 1], [3, 2]],
    )
    generator.microtile = [
        splinepy.Bezier(
            degrees=[3],
            control_points=[[0, 0.5], [0.5, 1], [0.5, 0], [1, 0.5]],
        ),
        splinepy.Bezier(
            degrees=[4],
            control_points=[
                [0.5, 0],
                [0.75, 0.5],
                [0.8, 0.8],
                [0.25, 0.5],
                [0.5, 1],
            ],
        ),
    ]
    generator.tiling = [10, 10]
    generator.create(macro_sensitivities=True)


def create_microstructure_crosstile3d():
    # Cross tile 3D in a microstructure
    generator = splinepy.microstructure.microstructure.Microstructure()
    generator.deformation_function = splinepy.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
    ).create.extruded(extrusion_vector=[0, 0, 1])
    generator.microtile = splinepy.microstructure.tiles.CrossTile3D()
    generator.tiling = [10, 10, 1]
    generator.create(macro_sensitivities=True)


@pytest.mark.benchmark
def test_microstructure_simple(benchmark):
    benchmark(create_microstructure_simple)
    assert True


@pytest.mark.benchmark
def test_microstructure_cross_tile_3d(benchmark):
    benchmark(create_microstructure_crosstile3d)
    assert True


if __name__ == "__main__":
    c.unittest.main()
