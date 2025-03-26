#ifndef __ZONOOPT_CHOLESKY_UTILITIES_HPP__
#define __ZONOOPT_CHOLESKY_UTILITIES_HPP__

#include "Eigen/Sparse"
#include "Eigen/Dense"

namespace ZonoOpt
{

template <typename float_type>
struct LDLT_data
{
    Eigen::SparseMatrix<float_type> L;
    Eigen::DiagonalMatrix<float_type, -1> Dinv;
    Eigen::PermutationMatrix<-1, -1, int> P, Pinv;
};


template <typename float_type>
void get_LDLT_data(const Eigen::SimplicialLDLT<Eigen::SparseMatrix<float_type>>& solver, LDLT_data<float_type>& data)
{
    data.L = solver.matrixL();
    data.Dinv = solver.vectorD().cwiseInverse().asDiagonal();
    data.P = solver.permutationP();
    data.Pinv = solver.permutationPinv();
}


template <typename float_type>
Eigen::Vector<float_type, -1> solve_LDLT(const LDLT_data<float_type>& data, const Eigen::Vector<float_type, -1>& b)
{
    Eigen::Vector<float_type, -1> bbar = data.P*b;
    Eigen::Vector<float_type, -1> y = data.Dinv*data.L.template triangularView<Eigen::Lower>().solve(bbar);
    Eigen::Vector<float_type, -1> xbar = data.L.transpose().template triangularView<Eigen::Upper>().solve(y);
    return data.Pinv*xbar;
}


} // end namespace ZonoOpt

#endif