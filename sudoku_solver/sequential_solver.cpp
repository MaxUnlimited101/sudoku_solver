// Huge thanks to 
// https://github.com/AnnaTz/sudoku-solver-cpp/blob/main/sudoku_solver.cpp

// Sequential sudoku solver using backtracking

#include<stdio.h>
#include "sequential_solver.h"

sequential_solver::sequential_solver(const board& sudoku)
{
    this->sudoku = sudoku;
    temp.assign(sudoku.size(), {});
    for (int i = 0; i < sudoku.size(); i++)
    {
        temp[i].assign(sudoku.size(), 0);
    }
    this->solved = false;
}

sequential_solver::~sequential_solver()
{
}

int sequential_solver::checkrow(int row, int num)
{
    int column;
    for (column = 0; column < 9; column++)
        if (sudoku[row][column] == num)
            return 0;
    return 1;
}

int sequential_solver::checkcolumn(int column, int num)
{
    int row;
    for (row = 0; row < 9; row++)
        if (sudoku[row][column] == num)
            return 0;
    return 1;
}

int sequential_solver::checkgrid(int grid_row, int grid_column, int num)
{
    int row, column;
    for (row = grid_row * 3; row < (grid_row + 1) * 3; row++)
        for (column = grid_column * 3; column < (grid_column + 1) * 3; column++)
            if (sudoku[row][column] == num)
                return 0;
    return 1;
}

std::tuple<bool, std::vector<board>> sequential_solver::alternatives()
{
    int row, column, n, counter;
    std::vector<board> alt;
    for (row = 0; row < 9; row++)
        for (column = 0; column < 9; column++)
            if (sudoku[row][column] == 0)
            {
                for (n = 1; n < 10; n++)
                    if (checkrow(row, n) && checkcolumn(column, n) && checkgrid(row / 3, column / 3, n))
                        alt[row][column].push_back(n);
                if (alt[row][column].empty())
                    return std::make_tuple(false, alt);
            }
    return std::make_tuple(true, alt);
}

bool sequential_solver::fill(std::vector<board>& alt)
{
    bool flag = false;
    for (int row = 0; row < 9; row++)
        for (int column = 0; column < 9; column++)
            if (alt[row][column].size() == 1 && alt[row][column][0] != 0)
            {
                sudoku[row][column] = alt[row][column][0];
                alt[row][column].clear();
                flag = true;
            }
    return flag;
}

std::tuple<int, int> sequential_solver::best(std::vector<board>& alt)
{
    for (int i = 2; i < 10; i++)
        for (int row = 0; row < 9; row++)
            for (int column = 0; column < 9; column++)
                if (alt[row][column].size() == i)
                    return std::make_tuple(row, column);
    return std::make_tuple(-1, -1);
}

void sequential_solver::solve(board& state, std::vector<board>& alt)
{
    if (solved)
        return;

    std::tuple<bool, std::vector<board>> val;

    while (fill(state, alt))
    {
        val = alternatives(state);
        if (!std::get<0>(val))
            return;
        else
            alt = std::get<1>(val);
    }

    std::tuple<int, int> pos = best(alt);
    int row = std::get<0>(pos);
    int column = std::get<1>(pos);

    if (row == -1 && column == -1)
    {
        if (!solved)
        {
            sudoku = state;
            solved = true;
        }
    }
    else
        for (int v : alt[row][column])
        {
            temp = state;
            temp[row][column] = v;
            val = alternatives(temp);
            if (!std::get<0>(val))
                continue;
            else
                solve(temp, std::get<1>(val));
        }
}

void sequential_solver::solve()
{
    this->sudoku = sudoku;
    temp.assign(sudoku.size(), {});
    for (int i = 0; i < sudoku.size(); i++)
    {
        temp[i].assign(sudoku.size(), 0);
    }
    this->solved = false;

    auto val = alternatives();
    if (std::get<0>(val))
        solve(sudoku, std::get<1>(val));

    if (solved)
    {
        std::cout << "Sudoku solved!\n";
        print_sudoku(sudoku);
    }
    else
        std::cout << "Sudoku could not be solved.\n";
}