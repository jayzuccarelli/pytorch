# Owner(s): ["module: inductor"]

import torch
from torch._inductor import inductor_prims
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def _reference_padded_blockwise_scatter(
    values,
    padded_rows,
    padded_cols,
    logical_row_chunk,
    physical_row_chunk,
    xdl,
    col_chunk,
    col_inner,
    padding_value,
):
    result = torch.full(
        (padded_rows * padded_cols,),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    rows, cols = values.shape
    for row in range(rows):
        physical_row = (
            row // logical_row_chunk * physical_row_chunk
            + row % logical_row_chunk
        )
        row_outer = physical_row // (2 * xdl)
        row_inner = physical_row % (2 * xdl)
        for col in range(cols):
            col_outer = col // col_chunk
            col_inner_index = col % col_chunk
            offset = (
                (((
                    (row_outer * (padded_cols // col_chunk) + col_outer)
                    * col_inner
                    + col_inner_index % col_inner
                ) * xdl + row_inner % xdl) * 2 + col_inner_index // col_inner)
                * 2
                + row_inner // xdl
            )
            result[offset] = values[row, col]
    return result


@instantiate_parametrized_tests
class PaddedBlockwiseScatterTest(TestCase):
    __unittest_skip__ = not HAS_GPU

    @parametrize(
        "rows,cols,padded_rows,padded_cols,logical_row_chunk,physical_row_chunk",
        [
            (4, 4, 4, 4, 4, 4),
            (4, 3, 4, 4, 4, 4),
            (4, 4, 8, 4, 2, 4),
            (5, 3, 16, 4, 4, 8),
        ],
    )
    def test_exact_layout(
        self,
        rows,
        cols,
        padded_rows,
        padded_cols,
        logical_row_chunk,
        physical_row_chunk,
    ):
        xdl = 2
        col_chunk = 4
        col_inner = 2
        padding_value = 127
        values_cpu = torch.arange(rows * cols).reshape(rows, cols).to(torch.uint8)
        expected = _reference_padded_blockwise_scatter(
            values_cpu,
            padded_rows,
            padded_cols,
            logical_row_chunk,
            physical_row_chunk,
            xdl,
            col_chunk,
            col_inner,
            padding_value,
        ).to(GPU_TYPE)
        values = values_cpu.to(GPU_TYPE)

        def f(x):
            return inductor_prims.padded_blockwise_scatter(
                x,
                padded_rows,
                padded_cols,
                logical_row_chunk,
                physical_row_chunk,
                xdl,
                col_chunk,
                col_inner,
                padding_value,
            )

        self.assertEqual(f(values), expected, atol=0, rtol=0)
        self.assertEqual(torch.compile(f)(values), expected, atol=0, rtol=0)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
