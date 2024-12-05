
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <string>
#include "helpers.h"


int main()
{
    //generate_sample(50000);
    test_sequential("./tests/extreme/10000");

	return EXIT_SUCCESS;
}