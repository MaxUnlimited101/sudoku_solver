#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helpers.h"
#include "cuda_solver.cuh"


int main()
{
    //generate_sample(10000);
    generate_sample(5);
    //test_sequential("./tests/extreme/10000");
    test_cuda("./tests/medium/5");

	return EXIT_SUCCESS;
}