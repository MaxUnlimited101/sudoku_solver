#pragma once
#include <cuda_runtime.h>
#include <limits>   
#include "helpers.h"
#include <iostream>

void test_cuda(const std::string& path);
void check_cuda_error(cudaError_t res);
void solve_sudoku_cuda(const std::vector<board>& boards, int threads_per_block, int blocks);

__global__ void BFS_kernel(char* old_boards, char* new_boards, int old_boards_c, int* board_index, int* empty_spaces, int* empty_space_count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    while (index < old_boards_c)
    {
        bool f = 0;

        for (int i = index * N * N; (i < (index * N * N) + N * N) && (f == 0); i++)
        {
            if (old_boards[i] == 0)
            {
                f = true;
                int t = i - N * N * index;
                int r = t / N, c = t % N;

                for (char num = 1; num <= N; num++)
                {
                    bool works = 1;
                    for (int j = 0; j < N; j++)
                    {
                        if (old_boards[index * N * N + r * N + j] == num)
                        {
                            works = 0;
                            break;
                        }
                    }

                    if (!works)
                    {
                        continue;
                    }

                    for (int j = 0; j < N; j++)
                    {
                        if (old_boards[index * N * N + j * N + c] == num)
                        {
                            works = 0;
                            break;
                        }
                    }

                    if (!works)
                    {
                        continue;
                    }

                    for (int j = n * (r / n); j < n; j++)
                    {
                        for (int k = n * (c / n); k < n; k++)
                        {
                            if (old_boards[N * N * index + j * N + k] == num)
                            {
                                works = 0;
                                break;
                            }
                        }
                    }

                    if (works)
                    {
                        int next_board_ind = atomicAdd(board_index, 1), empty_index = 0;

                        for (int j = 0; j < N; j++)
                        {
                            for (int k = 0; k < N; k++)
                            {
                                new_boards[next_board_ind * N * N + j * N + k] = old_boards[index * N * N + j * N + k];
                                if (old_boards[index * N * N + j * N + k] == 0 && !(j == r && k == c))
                                {
                                    empty_spaces[empty_index + N * N * next_board_ind] = j * N + k;
                                    empty_index++;
                                }
                            }
                        }
                        empty_space_count[next_board_ind] = empty_index;
                        new_boards[next_board_ind * N * N + N * r + c] = num;
                    }
                }
            }
        }
        index += blockDim.x * gridDim.x;
    }
}

__global__ void backtrack_kernel(int* boards, int boards_count , int* empty_spaces, int* empty_spaces_count, int* finished, int* solved)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int *cur_board, *cur_empty_spaces, cur_empty_spaces_count;

    while ((*finished == 0) && (index < boards_count))
    {

    }
}

void check_cuda_error(cudaError_t res)
{
    if (res != cudaSuccess)
    {
        std::cerr << "CUDA ERROR! '" << cudaGetErrorString(res) << '\'' << std::endl;
        exit(EXIT_FAILURE);
    }
}

void solve_sudoku_cuda(const std::vector<board>& boards, int threads_per_block, int blocks)
{
    char* d_new_boards, *d_old_boards;
    int *d_empty_spaces, *d_empty_space_count, *d_board_index;

    int max_bfs = pow(2, 25);

    check_cuda_error(cudaMalloc(&d_new_boards, sizeof(char) * max_bfs));
    check_cuda_error(cudaMalloc(&d_old_boards, sizeof(char) * max_bfs));
    check_cuda_error(cudaMalloc(&d_empty_spaces, sizeof(int) * max_bfs));
    check_cuda_error(cudaMalloc(&d_empty_space_count, sizeof(int) * (max_bfs / (N * N) + 1)));
    check_cuda_error(cudaMalloc(&d_board_index, sizeof(int)));

    check_cuda_error(cudaMemset(&d_board_index, 0, sizeof(int)));
    check_cuda_error(cudaMemset(&d_new_boards, 0, max_bfs * sizeof(int)));

    char* arr = new char[N * N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            arr[i * N + j] = boards[0][i][j];
        }
    }

    // !!! change it to accept multiple boards
    check_cuda_error(cudaMemcpy(d_old_boards, arr, N * N * sizeof(char), cudaMemcpyHostToDevice));

    BFS_kernel <<< blocks, threads_per_block >>>(d_old_boards, d_new_boards, 1, d_board_index, d_empty_spaces, d_empty_space_count);

    check_cuda_error(cudaGetLastError());

    // iters must be even!
    int h_count, iters = 16;

    for (int i = 0; i < iters; i++)
    {
        check_cuda_error(cudaMemcpy(&h_count, d_board_index, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "Total boards after an iteration " << i << ": " << h_count << std::endl;

        check_cuda_error(cudaMemset(d_board_index, 0, sizeof(int)));

        if (i % 2 == 0)
        {
            BFS_kernel<<<  >>>();
        }
        else
        {
            BFS_kernel<<<  >>>();
        }
        check_cuda_error(cudaGetLastError());
    }

    check_cuda_error(cudaMemcpy(&h_count, d_board_index, sizeof(int), cudaMemcpyDeviceToHost));

    char *d_finished, *d_solved;

    check_cuda_error(cudaMalloc(&d_finished, sizeof(char)));
    check_cuda_error(cudaMalloc(&d_solved, N * N * sizeof(char)));

    check_cuda_error(cudaMemset(&d_finished, 0, sizeof(char)));
    check_cuda_error(cudaMemset(&d_solved, 0, N * N * sizeof(char)));

    backtrack_kernel<<<  >>>();

    char* solved = new char[N * N];

    memset(solved, 0, N * N * sizeof(char));

    check_cuda_error(cudaMemcpy(solved, d_solved, N * N * sizeof(char), cudaMemcpyDeviceToHost));

    check_cuda_error(cudaFree(d_empty_spaces));
    check_cuda_error(cudaFree(d_empty_space_count));
    check_cuda_error(cudaFree(d_new_boards));
    check_cuda_error(cudaFree(d_old_boards));
    check_cuda_error(cudaFree(d_board_index));
    check_cuda_error(cudaFree(d_finished));
    check_cuda_error(cudaFree(d_solved));

    delete[] solved;
    delete[] arr;
}

// random sudokus solved on GPU
// path - Path to file with sudokus
void test_cuda(const std::string& path)
{
    std::vector<board> rng_boards = load_sudokus(path);
    int n = rng_boards.size();

    std::cout << "Loaded " << n << " boards, solving them on GPU... NOW" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // cuda kernel here...
    solve_sudoku_cuda(rng_boards, 512, 512);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to solve " << n << " boards on GPU (microseconds): " << duration.count() << std::endl;
}

