#include <pinyon.h>

void random_matrix(prng device, const size_t rows, const size_t cols, const int discrete = 20)
{
    const size_t entries = rows * cols;

    std::vector<double> rpd, cpd;

    Matrix<PairReal<double>> matrix{rows, cols};

    for (int i = 0; i < entries; ++i)
    {
        rpd.emplace_back(device.uniform());
        cpd.emplace_back(device.uniform());

        matrix[i].row_value = rpd[i];
        matrix[i].col_value = cpd[i];
    }

    std::vector<double> row_strategy, col_strategy;
    LRSNash::solve(matrix, row_strategy, col_strategy, discrete);

    auto expl = math::exploitability(matrix, row_strategy, col_strategy);
    double expl_ = static_cast<double>(expl);
    assert (discrete * expl_ < 1);
}

int main()
{
    prng device{0};

    const int discrete = 100;
    const size_t trials = 100;
    for (int rows = 2; rows <= 9; ++rows)
    {
        for (int cols = 2; cols <= 9; ++cols)
        {
            for (size_t trial = 0; trial < trials; ++trial)
            {
                random_matrix(device, rows, cols, discrete);
            }
        }
    }
};