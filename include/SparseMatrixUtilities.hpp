#ifndef __ZONOOPT_SPARSEMATRIXUTILITIES_HPP__
#define __ZONOOPT_SPARSEMATRIXUTILITIES_HPP__

#include <Eigen/Sparse>
#include <vector>
#include <exception>

namespace ZonoOpt
{

template <typename T>
Eigen::SparseMatrix<T> hcat(const Eigen::SparseMatrix<T> &A, const Eigen::SparseMatrix<T> &B)
{
    if (A.rows() != B.rows())
    {
        throw std::invalid_argument("hcat: number of rows must match.");
    }

    Eigen::SparseMatrix<T> C(A.rows(), A.cols() + B.cols());
    std::vector<Eigen::Triplet<T>> tripvec;
    tripvec.reserve(A.nonZeros() + B.nonZeros());

    for (int k=0; k<A.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it)
        {
            tripvec.push_back(Eigen::Triplet<T>(it.row(), it.col(), it.value()));
        }
    }

    for (int k=0; k<B.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(B, k); it; ++it)
        {
            tripvec.push_back(Eigen::Triplet<T>(it.row(), it.col()+A.cols(), it.value()));
        }
    }

    C.setFromTriplets(tripvec.begin(), tripvec.end());
    return C;
}


template<typename T>
Eigen::SparseMatrix<T> vcat(const Eigen::SparseMatrix<T> &A, const Eigen::SparseMatrix<T> &B)
{
    if (A.cols() != B.cols())
    {
        throw std::invalid_argument("vcat: number of columns must match.");
    }

    Eigen::SparseMatrix<T> C(A.rows() + B.rows(), A.cols());
    std::vector<Eigen::Triplet<T>> tripvec;
    tripvec.reserve(A.nonZeros() + B.nonZeros());

    for (int k=0; k<A.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it)
        {
            tripvec.push_back(Eigen::Triplet<T>(it.row(), it.col(), it.value()));
        }
    }

    for (int k=0; k<B.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(B, k); it; ++it)
        {
            tripvec.push_back(Eigen::Triplet<T>(it.row()+A.rows(), it.col(), it.value()));
        }
    }

    C.setFromTriplets(tripvec.begin(), tripvec.end());
    return C;
}

// get triplets for matrix
template <typename T>
void get_triplets_offset(const Eigen::SparseMatrix<T> &mat, std::vector<Eigen::Triplet<T>> &triplets, 
            int i_offset, int j_offset)
{
    // check validity
    if (i_offset < 0 || j_offset < 0)
    {
        throw std::invalid_argument("get_triplets_offset: offsets must be non-negative.");
    }

    // get triplets
    for (int k=0; k<mat.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it)
        {
            triplets.push_back(Eigen::Triplet<T>(it.row() + i_offset, it.col() + j_offset, it.value()));
        }
    }
}

// remove redundant constraints, A*x = b
template <typename T>
void remove_redundant(Eigen::SparseMatrix<T>& A, Eigen::Vector<T,-1>& b)
{
    // check for empty input matrix
    if (A.rows() == 0 || A.cols() == 0)
    {
        A.resize(0, 0);
        b.resize(0);
        return;
    }

    // transpose
    Eigen::SparseMatrix<T> At = A.transpose();

    // compute QR decomposition
    Eigen::SparseQR<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> qr;
    qr.analyzePattern(At);
    qr.factorize(At);

    // get the permutation matrix and its indices
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = qr.colsPermutation();
    Eigen::VectorXi P_indices = P.indices();

    // QR solver puts linearly dependent rows at end
    std::vector<Eigen::Triplet<T>> tripvec;
    for (int i=0; i<qr.rank(); i++)
    {
        tripvec.push_back(Eigen::Triplet<T>(P_indices(i), i, 1.0));
    }

    Eigen::SparseMatrix<T> P_full (At.cols(), qr.rank());
    P_full.setFromTriplets(tripvec.begin(), tripvec.end());

    // remove redundant constraints
    A = (At * P_full).transpose();
    b = (b.transpose() * P_full).transpose();
}


} // end namespace ZonoOpt

#endif