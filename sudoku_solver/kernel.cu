
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "sequential_solver.h"

int main()
{
    // 100 random sudokus solved sequentially
    for (int i = 0; i < 100; i++)
    {
        
        sequential_solver seq_solv(sudoku_board);
        seq_solv.solve();
    }

	return EXIT_SUCCESS;
}