#pragma once
#include <cuda_runtime.h>
#include <limits>   
#include "helpers.h"
#include <iostream>
#include <cstdio>

void test_cuda(const std::string& path);
void solve_sudoku_cuda(const std::vector<board>& boards, int blocks, int threads_per_block);

__device__ bool is_valid_board(const int* board)
{
    bool visited[N] = { false };

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int v = board[i * N + j];

            if (v != 0)
            {
                if (visited[v - 1])
                {
                    return false;
                }
                else
                {
                    visited[v - 1] = true;
                }
            }
        }
        for (int k = 0; k < N; k++)
            visited[k] = false;
    }

    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            int v = board[j * N + i];

            if (v != 0)
            {
                if (visited[v - 1])
                {
                    return false;
                }
                else
                {
                    visited[v - 1] = true;
                }
            }
        }
        for (int k = 0; k < N; k++)
            visited[k] = false;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < N; k++)
                visited[k] = false;

            for (int k = 0; k < n; k++)
            {
                for (int l = 0; l < n; l++)
                {
                    int v = board[(n * i + k) * N + (n * j + l)];
                    if (v != 0)
                    {
                        if (visited[v - 1])
                        {
                            return false;
                        }
                        else
                        {
                            visited[v - 1] = true;
                        }
                    }
                }
            }
        }
    }
    return true;
}

__global__ void BFS_kernel(int* old_boards, int* new_boards, int old_boards_c, int* board_index, int* empty_spaces, int* empty_space_count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    while (index < old_boards_c)
    {
        bool f = false;

        for (int i = index * N * N; (i < (index * N * N) + N * N) && (f == false); i++)
        {
            if (old_boards[i] == 0)
            {
                f = true;
                int t = i - N * N * index;
                int r = t / N, c = t % N;

                for (int num = 1; num <= N; num++)
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
                        int next_board_ind = atomicAdd(board_index, 1);
                        int empty_index = 0;

                        for (int j = 0; j < N; j++)
                        {
                            for (int k = 0; k < N; k++)
                            {
                                new_boards[next_board_ind * N * N + j * N + k] = old_boards[index * N * N + j * N + k];
                                if (old_boards[index * N * N + j * N + k] == 0 && (j != r || k != c))
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

__global__ void backtrack_kernel(int* boards, int boards_count, int* empty_spaces, int* empty_spaces_count, int* finished, int* solved)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    int* cur_board;
    int* cur_empty_spaces, cur_empty_spaces_count;

    while ((*finished == 0) && (index < boards_count))
    {
        int empty_ind = 0;

        cur_board = boards + index * N * N;
        cur_empty_spaces = empty_spaces + index * N * N;
        cur_empty_spaces_count = empty_spaces_count[index];

        printf("Backtracking index: %d, cur_empty_sp_c: %d, \n", index, cur_empty_spaces_count);

        while ((empty_ind >= 0) && (empty_ind < cur_empty_spaces_count) && (*finished == 0))
        {            
            cur_board[cur_empty_spaces[empty_ind]]++;

            if (!is_valid_board(cur_board))
            {
                if (cur_board[cur_empty_spaces[empty_ind]] > N)
                {
                    cur_board[cur_empty_spaces[empty_ind]] = 0;
                    empty_ind--;
                }
            }
            else
            {
                empty_ind++;
            }
        }
        if (empty_ind == cur_empty_spaces_count && (*finished == 0))
        {
            *finished = 1;

            for (int i = 0; i < N * N; i++)
            {
                solved[i] = cur_board[i];
            }
        }
        index += gridDim.x * blockDim.x;
    }
}

//void check_cuda_error(cudaError_t res)
#define check_cuda_error(res) \
{ \
    if (res != cudaSuccess) \
    { \
        std::cerr << "CUDA ERROR! '" << cudaGetErrorString(res) << '\'' << std::endl; \
        std::cerr << "FILE: '" << __FILE__ << "'; LINE: '" << __LINE__ << "'" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void solve_sudoku_cuda(const std::vector<board>& boards, int blocks, int threads_per_block)
{
    int* d_new_boards, * d_old_boards;
    int* d_empty_spaces, * d_empty_space_count, * d_board_index;

    int max_bfs = pow(2, 20);

    check_cuda_error(cudaMalloc(&d_new_boards, sizeof(int) * max_bfs));
    check_cuda_error(cudaMalloc(&d_old_boards, sizeof(int) * max_bfs));
    check_cuda_error(cudaMalloc(&d_empty_spaces, sizeof(int) * max_bfs));
    check_cuda_error(cudaMalloc(&d_empty_space_count, sizeof(int) * (max_bfs / (N * N) + 1)));
    check_cuda_error(cudaMalloc(&d_board_index, 1 * sizeof(int)));

    check_cuda_error(cudaMemset(d_board_index, 0, 1 * sizeof(int)));
    check_cuda_error(cudaMemset(d_new_boards, 0, max_bfs * sizeof(int)));

    int* arr = new int[N * N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            arr[i * N + j] = boards[4][i][j];
        }
    }

    // TODO: !!! change it to accept multiple boards
    check_cuda_error(cudaMemcpy(d_old_boards, arr, N * N * sizeof(int), cudaMemcpyHostToDevice));

    BFS_kernel <<< blocks, threads_per_block >>> (d_old_boards, d_new_boards, 1, d_board_index, d_empty_spaces, d_empty_space_count);

    check_cuda_error(cudaGetLastError());

    // iters must be even!
    int h_count, iters = 6;

    for (int i = 0; i < iters; i++)
    {
        check_cuda_error(cudaMemcpy(&h_count, d_board_index, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "Total boards after an iteration " << i << ": " << h_count << std::endl;

        check_cuda_error(cudaMemset(d_board_index, 0, sizeof(int)));

        if (i % 2 == 0)
        {
            BFS_kernel <<< blocks, threads_per_block >>> (d_new_boards, d_old_boards,
                h_count, d_board_index, d_empty_spaces, d_empty_space_count);
        }
        else
        {
            BFS_kernel <<< blocks, threads_per_block >>> (d_old_boards, d_new_boards,
                h_count, d_board_index, d_empty_spaces, d_empty_space_count);
        }
        check_cuda_error(cudaGetLastError());
    }

    check_cuda_error(cudaMemcpy(&h_count, d_board_index, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Total boards: %d\n", h_count);

    int* d_finished;
    int* d_solved;

    check_cuda_error(cudaMalloc(&d_finished, sizeof(int)));
    check_cuda_error(cudaMalloc(&d_solved, N * N * sizeof(int)));

    check_cuda_error(cudaMemset(d_finished, 0, sizeof(int)));
    check_cuda_error(cudaMemset(d_solved, 0, N * N * sizeof(int)));

    backtrack_kernel <<< blocks, threads_per_block >>> (d_new_boards, h_count, d_empty_spaces, d_empty_space_count, d_finished, d_solved);

    check_cuda_error(cudaGetLastError());

    int* solved = new int[N * N];
    memset(solved, 0, N * N * sizeof(int));

    std::cout << "Synchronising..." << std::endl;
    check_cuda_error(cudaDeviceSynchronize());
    std::cout << "Synchronised..." << std::endl;

    check_cuda_error(cudaMemcpy(solved, d_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost));

    // here we can print solved
    board b(N, std::vector<char>(N, 0));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            b[i][j] = solved[i * N + j];
        }
    }
    print_sudoku(b);

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
    solve_sudoku_cuda(rng_boards, n, 1024);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to solve " << n << " boards on GPU (microseconds): " << duration.count() << std::endl;
}
