#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>

using board = std::vector<std::vector<char>>;

bool is_valid(board& board, int row, int col, int num);
bool solve_sudoku(board& board, int row, int col);
void remove_cells(board& board, int n);
void print_sudoku(const board& board);

void generate_sudoku(int how_many, std::vector<board>& boards)
{
    srand(time(NULL));
    for (int i = 0; i < how_many; i++)
    {
        board board(9, std::vector<char>(9, 0));
        solve_sudoku(board, 0, 0);

        // 17 is the minimum for sudoku board with exactly one unique solution.
        // But I assume any valid board is valid (so there can be many solutions, 
        // I return first solution to be found)

        int difficultyLevel = rand() % (81) + 1;
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
            std::cout << std::to_string(board[row][column]) + " ";
            if ((column - 2) % 3 == 0)
                std::cout << "| ";
        }
        std::cout << "\n";
        if ((column - 2) % 3 == 0)
            std::cout << "---------------------\n";
    }
}