// Huge thanks to 
// https://github.com/AnnaTz/sudoku-solver-cpp/blob/main/sudoku_solver.cpp

// Sequential sudoku solver using backtracking
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include "helpers.h"

class sequential_solver
{
public:
	sequential_solver(const board& sudoku);
	~sequential_solver();
	void solve();

private:
	void solve(board& state, std::vector<board>& alt);
	int checkrow(int row, int num);
	int checkcolumn(int column, int num);
	int checkgrid(int grid_row, int grid_column, int num);
	std::tuple<bool, std::vector<board>> alternatives();
	bool fill(std::vector<board>& alt);
	std::tuple<int, int> best(std::vector<board>& alt);

private:
	board sudoku;
	board temp;
	bool solved;
};

