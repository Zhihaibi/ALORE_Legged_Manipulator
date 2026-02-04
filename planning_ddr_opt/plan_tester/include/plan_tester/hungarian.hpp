#ifndef HUNGARIAN_HPP
#define HUNGARIAN_HPP

#include <vector>
#include <limits>
#include <Eigen/Eigen>

// A C++ implementation of the Hungarian algorithm for solving the assignment problem.
// It finds the minimum cost matching in a bipartite graph.
class HungarianAlgorithm
{
public:
    double solve(const Eigen::MatrixXd &costMatrix, std::vector<int> &assignment)
    {
        int n = costMatrix.rows();
        int m = costMatrix.cols();

        std::vector<double> u(n + 1, 0), v(m + 1, 0);
        std::vector<int> p(m + 1, 0), way(m + 1, 0);

        for (int i = 1; i <= n; ++i)
        {
            p[0] = i;
            int j0 = 0;
            std::vector<double> minv(m + 1, std::numeric_limits<double>::max());
            std::vector<bool> used(m + 1, false);

            do
            {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                double delta = std::numeric_limits<double>::max();

                for (int j = 1; j <= m; ++j)
                {
                    if (!used[j])
                    {
                        double cur = costMatrix(i0 - 1, j - 1) - u[i0] - v[j];
                        if (cur < minv[j])
                        {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta)
                        {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for (int j = 0; j <= m; ++j)
                {
                    if (used[j])
                    {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    }
                    else
                    {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);

            do
            {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }

        assignment.resize(n);
        for (int j = 1; j <= m; ++j)
        {
            if (p[j] > 0)
            {
                assignment[p[j] - 1] = j - 1;
            }
        }

        return -v[0];
    }
};

#endif // HUNGARIAN_HPP