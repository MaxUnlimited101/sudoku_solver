#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helpers.h"
#include "cuda_solver.cuh"


int main()
{
    //generate_sample(50000);
    test_sequential("./tests/extreme/10000");

	return EXIT_SUCCESS;
}