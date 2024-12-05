#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#define N 9

using board = std::vector<std::vector<char>>;

bool is_valid(board& board, int row, int col, int num);
bool solve_sudoku(board& board, int row, int col);
void remove_cells(board& board, int n);
void print_sudoku(const board& board);
std::vector<board> load_sudokus(const std::string& path);
void save_sudokus_to_file(const std::string& path, const std::vector<board>& boards);
void generate_sample(const int n);
void test_sequential(const std::string& path);
void test_cuda(const std::string& path);

void rotate_matrix(board& matrix) {
    int n = matrix.size();

    // Transpose the matrix
    for (int i = 0; i < n; ++i) 
    {
        for (int j = i; j < n; ++j) 
        {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }

    // Reverse each row
    for (int i = 0; i < n; ++i) 
    {
        std::reverse(matrix[i].begin(), matrix[i].end());
    }
}

// Generate random sudoku and delete at most how_many_cells_to_delete cells
void generate_sudoku(int how_many, std::vector<board>& boards, int how_many_cells_to_delete)
{
    srand(time(NULL));
    for (int i = 0; i < how_many; i++)
    {
        board board(9, std::vector<char>(9, 0));
        solve_sudoku(board, 0, 0);

        // randomly rotate the board
        int c = rand() % 13;
        for (int j = 0; j < c; j++)
        {
            rotate_matrix(board);
        }

        // 17 is the minimum for sudoku board with exactly one unique solution.
        // But I assume any valid board is valid (so there can be many solutions, 
        // I return first solution to be found)

        int difficultyLevel = rand() % (how_many_cells_to_delete) + 1;
        remove_cells(board, difficultyLevel);
        boards.push_back(board);
    }
}

bool is_valid(board& board, int row, int col, int num) 
{
    for (int i = 0; i < 9; i++) 
    {
        if (board[row][i] == num || board[i][col] == num) 
        {
            return false;
        }
    }

    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = 0; i < 3; i++) 
    {
        for (int j = 0; j < 3; j++) 
        {
            if (board[startRow + i][startCol + j] == num) 
            {
                return false;
            }
        }
    }
    return true;

}

bool solve_sudoku(board& board, int row, int col) 
{
    if (row == 9) 
    {
        return true;
    }

    if (col == 9) 
    {
        return solve_sudoku(board, row + 1, 0);
    }

    if (board[row][col] != 0) 
    {
        return solve_sudoku(board, row, col + 1);
    }

    for (int num = 1; num <= 9; num++) 
    {
        if (is_valid(board, row, col, num)) 
        {
            board[row][col] = num;
            if (solve_sudoku(board, row, col + 1)) 
            {
                return true;
            }
            board[row][col] = 0;
        }
    }
    return false;

}

void remove_cells(board& board, int n)
{
    srand(time(NULL));
    while (n > 0) 
    {
        int row = rand() % 9;
        int col = rand() % 9;
        if (board[row][col] != 0) 
        {
            board[row][col] = 0;
            n--;
        }
    }
}

void print_sudoku(const board& board) 
{
    int row, column;
    for (row = 0; row < board.size(); row++)
    {
        for (column = 0; column < board[0].size(); column++)
        {
            std::cout << (board[row][column] + '0') + " ";
            if ((column - 2) % 3 == 0 && column != board.size() - 1)
                std::cout << "| ";
        }
        std::cout << "\n";
        if ((row - 2) % 3 == 0 && row != board.size()-1)
            std::cout << "---------------------\n";
    }
}

void save_sudokus_to_file(const std::string& path, const std::vector<board>& boards)
{
    std::ofstream out(path);
    for (board b : boards)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                out << (b[i][j] + '0');
            }
        }
        out << std::endl;
    }
    out.close();
}

std::vector<board> load_sudokus(const std::string& path)
{
    std::vector<board> res;
    std::ifstream in(path);
    std::string inp;

    while (std::getline(in, inp))
    {
        board b(N, std::vector<char>(N, 0));
        // unfortunately won't work for 81x81
        for (int i = 0; i < N * N; i++)
        {
            b[i / N][i % N] = inp[i] - '0';
        }
        res.push_back(b);
    }
    in.close();
    return res;
}

// Random sample of n sudokus for each difficulty.
void generate_sample(const int n)
{
    std::vector<board> boards;

    // 20 - easy
    // 40 - medium
    // 60 - hard
    // 81 - extreme

    std::cout << "Generating " << n << " random sudokus for each difficulty" << std::endl;

    generate_sudoku(n, boards, 20);
    save_sudokus_to_file("./tests/easy/" + std::to_string(n), boards);
    boards.clear();

    generate_sudoku(n, boards, 40);
    save_sudokus_to_file("./tests/medium/" + std::to_string(n), boards);
    boards.clear();

    generate_sudoku(n, boards, 60);
    save_sudokus_to_file("./tests/hard/" + std::to_string(n), boards);
    boards.clear();

    generate_sudoku(n, boards, 81);
    save_sudokus_to_file("./tests/extreme/" + std::to_string(n), boards);
    boards.clear();

    std::cout << "Done!" << std::endl;
}

// random sudokus solved sequentially
// path - Path to file with sudokus
void test_sequential(const std::string& path)
{
    // 50 mins for n = 100_000 (on my pc), random difficulty

    std::vector<board> rng_boards = load_sudokus(path);
    int n = rng_boards.size();

    std::cout << "Loaded " << n << " boards, solving them sequentially... NOW" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        solve_sudoku(rng_boards[i], 0, 0);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to solve " << n << " boards sequentially (microseconds): " << duration.count() << std::endl;
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

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to solve " << n << " boards on GPU (microseconds): " << duration.count() << std::endl;
}