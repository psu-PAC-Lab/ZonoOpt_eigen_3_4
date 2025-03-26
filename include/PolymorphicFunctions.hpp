#ifndef __ZONOOPT_POLYMORPHIC_FUNCTIONS_HPP__
#define __ZONOOPT_POLYMORPHIC_FUNCTIONS_HPP__

#include "AbstractZono.hpp"
#include "Zono.hpp"
#include "ConZono.hpp"
#include "HybZono.hpp"
#include "SparseMatrixUtilities.hpp"

#include <exception>
#include <sstream>
#include <algorithm>
#include <utility>
#include <memory>

namespace ZonoOpt
{

// print to ostream
template <typename T>
std::ostream& operator<<(std::ostream &os, const AbstractZono<T> &Z)
{
    os << Z.print();
    return os;
}

// affine map
template<typename float_type>
std::unique_ptr<AbstractZono<float_type>> affine_map(const AbstractZono<float_type>& Z,
            const Eigen::SparseMatrix<float_type>& R, const Eigen::Vector<float_type, -1>& s = Eigen::Vector<float_type, -1>()) 
{
    // check dimensions
    Eigen::Vector<float_type, -1> s_def;
    const Eigen::Vector<float_type, -1> * s_ptr = nullptr;
    if (s.size() == 0) // default argument
    {
        s_def.resize(R.rows());
        s_def.setZero();
        s_ptr = &s_def;
    }
    else
    {
        s_ptr = &s;
    }

    if (R.cols() != Z.n || R.rows() != s_ptr->size())
    {
        throw std::invalid_argument("Linear_map: invalid input dimensions.");
    }

    // apply affine map
    Eigen::SparseMatrix<float_type> Gc = R*Z.Gc;
    Eigen::SparseMatrix<float_type> Gb = R*Z.Gb;
    Eigen::Vector<float_type, -1> c = R*Z.c + *s_ptr;

    // output correct type
    if (Z.is_hybzono())
        return std::make_unique<HybZono<float_type>>(Gc, Gb, c, Z.Ac, Z.Ab, Z.b, Z.zero_one_form);
    else if (Z.is_conzono())
        return std::make_unique<ConZono<float_type>>(Gc, c, Z.A, Z.b, Z.zero_one_form);
    else if (Z.is_zono())
        return std::make_unique<Zono<float_type>>(Gc, c, Z.zero_one_form);
    else
        return std::make_unique<Point<float_type>>(c);
}

template <typename float_type>
std::unique_ptr<AbstractZono<float_type>> project_onto_dims(const AbstractZono<float_type>& Z, std::vector<int>& dims)
{
    // make sure all dims are >= 0 and < n
    for (auto it=dims.begin(); it!=dims.end(); ++it)
    {
        if (*it < 0 || *it >= Z.n)
        {
            throw std::invalid_argument("Project onto dims: invalid dimension.");
        }
    }

    // build affine map matrix
    Eigen::SparseMatrix<float_type> R (dims.size(), Z.n);
    std::vector<Eigen::Triplet<float_type>> tripvec;
    for (int i=0; i<dims.size(); i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(i, dims[i], 1));
    }
    R.setFromTriplets(tripvec.begin(), tripvec.end());

    // apply affine map
    return affine_map<float_type>(Z, R);
}

// Mikowski sum
template<typename float_type>
std::unique_ptr<AbstractZono<float_type>> minkowski_sum(const AbstractZono<float_type>& Z1, 
    AbstractZono<float_type>& Z2)
{
    // check dimensions
    if (Z1.n != Z2.n)
    {
        throw std::invalid_argument("Minkowski sum: n dimensions must match.");
    }

    // make sure Z1 and Z2 both using same generator range
    if (Z1.zero_one_form != Z2.zero_one_form)
    {
        Z2.convert_form();
    }

    std::vector<Eigen::Triplet<float_type>> tripvec;

    Eigen::SparseMatrix<float_type> Gc = hcat(Z1.Gc, Z2.Gc);
    Eigen::SparseMatrix<float_type> Gb = hcat(Z1.Gb, Z2.Gb);
    Eigen::Vector<float_type, -1> c = Z1.c + Z2.c;

    Eigen::SparseMatrix<float_type> Ac (Z1.nC + Z2.nC, Z1.nGc + Z2.nGc);
    get_triplets_offset(Z1.Ac, tripvec, 0, 0);
    get_triplets_offset(Z2.Ac, tripvec, Z1.nC, Z1.nGc);
    Ac.setFromTriplets(tripvec.begin(), tripvec.end());

    tripvec.clear();
    Eigen::SparseMatrix<float_type> Ab (Z1.nC + Z2.nC, Z1.nGb + Z2.nGb);
    get_triplets_offset(Z1.Ab, tripvec, 0, 0);
    get_triplets_offset(Z2.Ab, tripvec, Z1.nC, Z1.nGb);
    Ab.setFromTriplets(tripvec.begin(), tripvec.end());

    Eigen::Vector<float_type, -1> b (Z1.nC + Z2.nC);
    b.segment(0, Z1.nC) = Z1.b;
    b.segment(Z1.nC, Z2.nC) = Z2.b;

    // return correct output type
    if (Z1.is_hybzono() || Z2.is_hybzono())
        return std::make_unique<HybZono<float_type>>(Gc, Gb, c, Ac, Ab, b, Z1.zero_one_form);
    else if (Z1.is_conzono() || Z2.is_conzono())
        return std::make_unique<ConZono<float_type>>(Gc, c, Ac, b, Z1.zero_one_form);
    else
        return std::make_unique<Zono<float_type>>(Gc, c, Z1.zero_one_form);
}

// pontryagin difference
template<typename float_type>
std::unique_ptr<AbstractZono<float_type>> pontry_diff(const AbstractZono<float_type>& Z1, AbstractZono<float_type>& Z2)
{
    // check dimensions
    if (Z1.n != Z2.n)
    {
        throw std::invalid_argument("Pontryagin difference: n dimensions must match.");
    }

    // require Z2 to be a zonotope or point
    if (!(Z2.is_zono() || Z2.is_point()))
    {
        throw std::invalid_argument("Pontryagin difference: Z2 must be a zonotope or point.");
    }

    // make sure Z1 and Z2 both using same generator range
    if (Z1.zero_one_form != Z2.zero_one_form)
    {
        Z2.convert_form();
    }

    // init Zout
    std::unique_ptr<AbstractZono<float_type>> Z_out;
    if (Z1.is_point())
        Z_out = std::make_unique<Point<float_type>>(Z1.c);
    else if (Z1.is_zono())
        Z_out = std::make_unique<Zono<float_type>>(Z1.G, Z1.c, Z1.zero_one_form);
    else if (Z1.is_conzono())
        Z_out = std::make_unique<ConZono<float_type>>(Z1.G, Z1.c, Z1.A, Z1.b, Z1.zero_one_form);
    else if (Z1.is_hybzono())
        Z_out = std::make_unique<HybZono<float_type>>(Z1.Gc, Z1.Gb, Z1.c, Z1.Ac, Z1.Ab, Z1.b, Z1.zero_one_form);
    else
        throw std::invalid_argument("Pontryagin difference: unknown Z1 type.");

    Z_out->c -= Z2.c;

    // iteratively compute pontryagin difference from columns of Z2 generator matrix
    Eigen::Matrix<float_type, -1, -1> G2 = Z2.G.toDense();
    Eigen::Vector<float_type, -1> c_plus, c_minus;
    std::unique_ptr<AbstractZono<float_type>> Z_plus, Z_minus;

    for (int i=0; i<Z2.nG; i++)
    {
        c_plus = Z_out->c + G2.col(i);
        c_minus = Z_out->c - G2.col(i);
        
        if (Z_out->is_point())
        {
            Z_plus = std::make_unique<Point<float_type>>(c_plus);
            Z_minus = std::make_unique<Point<float_type>>(c_minus);
        }
        else if (Z_out->is_zono())
        {
            Z_plus = std::make_unique<Zono<float_type>>(Z_out->G, c_plus, Z_out->zero_one_form);
            Z_minus = std::make_unique<Zono<float_type>>(Z_out->G, c_minus, Z_out->zero_one_form);
        }
        else if (Z_out->is_conzono())
        {
            Z_plus = std::make_unique<ConZono<float_type>>(Z_out->G, c_plus, Z_out->A, Z_out->b, Z_out->zero_one_form);
            Z_minus = std::make_unique<ConZono<float_type>>(Z_out->G, c_minus, Z_out->A, Z_out->b, Z_out->zero_one_form);
        }
        else if (Z_out->is_hybzono())
        {
            Z_plus = std::make_unique<HybZono<float_type>>(Z_out->Gc, Z_out->Gb, c_plus, Z_out->Ac, Z_out->Ab, Z_out->b, Z_out->zero_one_form);
            Z_minus = std::make_unique<HybZono<float_type>>(Z_out->Gc, Z_out->Gb, c_minus, Z_out->Ac, Z_out->Ab, Z_out->b, Z_out->zero_one_form);
        }
        else
        {
            throw std::invalid_argument("Pontryagin difference: unknown Z_out type.");
        }

        Z_out = intersection(*Z_plus, *Z_minus);
    }

    return Z_out;
}

// intersection
template <typename float_type>
std::unique_ptr<AbstractZono<float_type>> intersection(const AbstractZono<float_type>& Z1, 
    AbstractZono<float_type>& Z2, const Eigen::SparseMatrix<float_type>& R=Eigen::SparseMatrix<float_type>())
{
    // handle default arguments
    const Eigen::SparseMatrix<float_type> * R_ptr = nullptr;
    Eigen::SparseMatrix<float_type> R_def;
    if (R.rows() == 0 && R.cols() == 0)
    {
        R_def.resize(Z1.n, Z1.n);
        R_def.setIdentity();
        R_ptr = &R_def;
    }
    else
    {
        R_ptr = &R;
    }

    // check dimensions
    if (R_ptr->rows() != Z2.n || R_ptr->cols() != Z1.n)
    {
        throw std::invalid_argument("Intersection: inconsistent input dimensions.");
    }

    // make sure Z1 and Z2 both using same generator range
    if (Z1.zero_one_form != Z2.zero_one_form)
    {
        Z2.convert_form();
    }

    // compute intersection
    Eigen::SparseMatrix<float_type> Gc = Z1.Gc;
    Gc.conservativeResize(Z1.n, Z1.nGc + Z2.nGc);

    Eigen::SparseMatrix<float_type> Gb = Z1.Gb;
    Gb.conservativeResize(Z1.n, Z1.nGb + Z2.nGb);

    Eigen::Vector<float_type, -1> c = Z1.c;

    std::vector<Eigen::Triplet<float_type>> tripvec;
    Eigen::SparseMatrix<float_type> Ac (Z1.nC + Z2.nC + R_ptr->rows(), Z1.nGc + Z2.nGc);
    get_triplets_offset(Z1.Ac, tripvec, 0, 0);
    get_triplets_offset(Z2.Ac, tripvec, Z1.nC, Z1.nGc);
    Eigen::SparseMatrix<float_type> RZ1Gc = (*R_ptr)*Z1.Gc;
    get_triplets_offset(RZ1Gc, tripvec, Z1.nC + Z2.nC, 0);
    Eigen::SparseMatrix<float_type> mZ2Gc = -Z2.Gc;
    get_triplets_offset(mZ2Gc, tripvec, Z1.nC + Z2.nC, Z1.nGc);
    Ac.setFromTriplets(tripvec.begin(), tripvec.end());

    tripvec.clear();
    Eigen::SparseMatrix<float_type> Ab (Z1.nC + Z2.nC + R_ptr->rows(), Z1.nGb + Z2.nGb);
    get_triplets_offset(Z1.Ab, tripvec, 0, 0);
    get_triplets_offset(Z2.Ab, tripvec, Z1.nC, Z1.nGb);
    Eigen::SparseMatrix<float_type> RZ1Gb = (*R_ptr)*Z1.Gb;
    get_triplets_offset(RZ1Gb, tripvec, Z1.nC + Z2.nC, 0);
    Eigen::SparseMatrix<float_type> mZ2Gb = -Z2.Gb;
    get_triplets_offset(mZ2Gb, tripvec, Z1.nC + Z2.nC, Z1.nGb);
    Ab.setFromTriplets(tripvec.begin(), tripvec.end());


    Eigen::Vector<float_type, -1> b (Z1.nC + Z2.nC + R_ptr->rows());
    b.segment(0, Z1.nC) = Z1.b;
    b.segment(Z1.nC, Z2.nC) = Z2.b;
    b.segment(Z1.nC + Z2.nC, R_ptr->rows()) = Z2.c - (*R_ptr)*Z1.c;

    // return correct output type
    if (Z1.is_hybzono() || Z2.is_hybzono())
        return std::make_unique<HybZono<float_type>>(Gc, Gb, c, Ac, Ab, b, Z1.zero_one_form);
    else
        return std::make_unique<ConZono<float_type>>(Gc, c, Ac, b, Z1.zero_one_form);
}

template <typename float_type>
std::unique_ptr<AbstractZono<float_type>> intersection_over_dims(const AbstractZono<float_type>& Z1, 
    AbstractZono<float_type>& Z2, std::vector<int>& dims)
{
    // check dimensions
    if (Z2.n != dims.size())
    {
        throw std::invalid_argument("Intersection over dims: Z2.n must equal number of dimensions.");
    }

    // make sure dims are >=0 and <Z1.n
    for (auto it=dims.begin(); it!=dims.end(); ++it)
    {
        if (*it < 0 || *it >= Z1.n)
        {
            throw std::invalid_argument("Intersection over dims: invalid dimension.");
        }
    }

    // build projection matrix
    Eigen::SparseMatrix<float_type> R (dims.size(), Z1.n);
    std::vector<Eigen::Triplet<float_type>> tripvec;
    for (int i=0; i<dims.size(); i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(i, dims[i], 1));
    }
    R.setFromTriplets(tripvec.begin(), tripvec.end());

    // generalized intersection
    return intersection(Z1, Z2, R);
}

template <typename float_type>
std::unique_ptr<AbstractZono<float_type>> halfspace_intersection(AbstractZono<float_type>& Z, const Eigen::SparseMatrix<float_type>& H, 
    const Eigen::Vector<float_type, -1>& f, const Eigen::SparseMatrix<float_type>& R=Eigen::SparseMatrix<float_type>())
{
    // handle default arguments
    const Eigen::SparseMatrix<float_type> * R_ptr = nullptr;
    Eigen::SparseMatrix<float_type> R_def;
    if (R.rows() == 0 && R.cols() == 0)
    {
        R_def.resize(Z.n, Z.n);
        R_def.setIdentity();
        R_ptr = &R_def;
    }
    else
    {
        R_ptr = &R;
    }

    // check dimensions
    if (R_ptr->rows() != H.cols() || R_ptr->cols() != Z.n || H.rows() != f.size())
    {
        throw std::invalid_argument("Halfspace intersection: inconsistent input dimensions.");
    }

    // make sure Z is using [-1,1] generators
    if (Z.zero_one_form)
    {
        Z.convert_form();
    }

    // declare
    int nH = H.rows();
    std::vector<Eigen::Triplet<float_type>> tripvec;

    // get zonotope distance of halfspace to furthest vertices of zonotopes
    Eigen::Array<float_type, -1, -1> HRG = (H*(*R_ptr)*Z.G).toDense().array();
    Eigen::Vector<float_type, -1> sum_HRG = HRG.abs().rowwise().sum();
    Eigen::Vector<float_type, -1> d_max = f - H*(*R_ptr)*Z.c + sum_HRG;
    Eigen::SparseMatrix<float_type> diag_d_max_2 (nH, nH);
    for (int i=0; i<nH; i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(i, i, d_max(i)/2));
    }
    diag_d_max_2.setFromTriplets(tripvec.begin(), tripvec.end());

    // build halfspace intersection
    Eigen::SparseMatrix<float_type> Gc = Z.Gc;
    Gc.conservativeResize(Z.n, Z.nGc + nH);

    Eigen::SparseMatrix<float_type> Gb = Z.Gb;
    Eigen::Vector<float_type, -1> c = Z.c;

    tripvec.clear();
    Eigen::SparseMatrix<float_type> Ac (Z.nC + nH, Z.nGc + nH);
    get_triplets_offset(Z.Ac, tripvec, 0, 0);
    Eigen::SparseMatrix<float_type> HRGc = H*(*R_ptr)*Z.Gc;
    get_triplets_offset(HRGc, tripvec, Z.nC, 0);
    get_triplets_offset(diag_d_max_2, tripvec, Z.nC, Z.nGc);
    Ac.setFromTriplets(tripvec.begin(), tripvec.end());

    tripvec.clear();
    Eigen::SparseMatrix<float_type> Ab (Z.nC + nH, Z.nGb);
    get_triplets_offset(Z.Ab, tripvec, 0, 0);
    Eigen::SparseMatrix<float_type> HRGb = H*(*R_ptr)*Z.Gb;
    get_triplets_offset(HRGb, tripvec, Z.nC, 0);
    Ab.setFromTriplets(tripvec.begin(), tripvec.end());

    Eigen::Vector<float_type, -1> b (Z.nC + nH);
    b.segment(0, Z.nC) = Z.b;
    b.segment(Z.nC, nH) = f - H*(*R_ptr)*Z.c - d_max/2;

    // return correct output type
    if (Z.is_hybzono())
        return std::make_unique<HybZono<float_type>>(Gc, Gb, c, Ac, Ab, b, Z.zero_one_form);
    else
        return std::make_unique<ConZono<float_type>>(Gc, c, Ac, b, Z.zero_one_form);
}

template <typename float_type>
std::unique_ptr<AbstractZono<float_type>> union_of_many(std::vector<AbstractZono<float_type>*>& Zs)
{
    // check we are taking a union of at least one zonotope
    if (Zs.size() == 0)
    {
        throw std::invalid_argument("Union: empty input vector.");
    }

    // check dimensions
    int n = Zs[0]->n;
    for (auto it=Zs.begin(); it!=Zs.end(); ++it)
    {
        if ((*it)->n != n)
        {
            throw std::invalid_argument("Union: inconsistent dimensions.");
        }
    }

    // make sure all Zs are using [0,1] generators
    for (auto it=Zs.begin(); it!=Zs.end(); ++it)
    {
        if (!(*it)->zero_one_form)
        {
            (*it)->convert_form();
        }
    }

    // declare
    std::vector<Eigen::Triplet<float_type>> tripvec;
    int rows = 0, cols = 0;
    std::vector<int> idx_sum_to_1;

    // generators

    // Gc
    rows = Zs[0]->n;
    for (auto it=Zs.begin(); it!=Zs.end(); ++it)
    {
        get_triplets_offset((*it)->Gc, tripvec, 0, cols);
        cols += (*it)->nGc + 1;
    }
    Eigen::SparseMatrix<float_type> Gc (rows, cols);
    Gc.setFromTriplets(tripvec.begin(), tripvec.end());

    // Gb
    tripvec.clear();
    cols = 0;
    rows = Zs[0]->n;
    for (auto it=Zs.begin(); it!=Zs.end(); ++it)
    {
        get_triplets_offset((*it)->Gb, tripvec, 0, cols);
        cols += (*it)->nGb;
        for (int i=0; i<(*it)->n; i++)
        {
            tripvec.push_back(Eigen::Triplet<float_type>(i, cols, (*it)->c(i)));
        }
        cols += 1;
    }
    Eigen::SparseMatrix<float_type> Gb (rows, cols);
    Gb.setFromTriplets(tripvec.begin(), tripvec.end());

    // c
    Eigen::Vector<float_type, -1> c (Zs[0]->n);
    c.setZero();

    // constraints
    
    // Ac
    tripvec.clear();
    rows = 0;
    cols = 0;
    for (auto it=Zs.begin(); it!=Zs.end(); ++it)
    {
        // populate first row
        for (int i=0; i<(*it)->nGc; i++)
        {
            tripvec.push_back(Eigen::Triplet<float_type>(rows, cols+i, 1));
        }
        tripvec.push_back(Eigen::Triplet<float_type>(rows, cols+(*it)->nGc, (float_type) (*it)->nG));
        
        // increment
        rows++;

        // equality constraints
        get_triplets_offset((*it)->Ac, tripvec, rows, cols);

        // increment
        rows += (*it)->nC;
        cols += (*it)->nGc + 1;
    }
    Eigen::SparseMatrix<float_type> Ac (rows+1, cols); // last row all zeroes
    Ac.setFromTriplets(tripvec.begin(), tripvec.end());

    // Ab
    tripvec.clear();
    rows = 0;
    cols = 0;
    for (auto it=Zs.begin(); it!=Zs.end(); ++it)
    {
        // populate first row
        for (int i=0; i<(*it)->nGb; i++)
        {
            tripvec.push_back(Eigen::Triplet<float_type>(rows, cols+i, 1));
        }
        tripvec.push_back(Eigen::Triplet<float_type>(rows, cols+(*it)->nGb, (float_type) -(*it)->nG));
        
        // increment
        rows++;

        // equality constraints
        get_triplets_offset((*it)->Ab, tripvec, rows, cols);

        // last column
        for (int i=0; i<(*it)->nC; i++)
        {
            tripvec.push_back(Eigen::Triplet<float_type>(rows+i, cols+(*it)->nGb, -(*it)->b(i)));
        }

        // increment
        rows += (*it)->nC;
        cols += (*it)->nGb + 1;

        // track sum-to-1 binaries
        idx_sum_to_1.push_back(cols-1);
    }
    // sum to 1 constraint
    for (auto it=idx_sum_to_1.begin(); it!=idx_sum_to_1.end(); ++it)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(rows, *it, 1));
    }
    rows++;
    Eigen::SparseMatrix<float_type> Ab (rows, cols);
    Ab.setFromTriplets(tripvec.begin(), tripvec.end());

    // b
    Eigen::Vector<float_type, -1> b (Ab.rows());
    b.setZero();
    b(Ab.rows()-1) = 1.0;

    // output
    return std::make_unique<HybZono<float_type>>(Gc, Gb, c, Ac, Ab, b, true);
}

template <typename float_type>
std::unique_ptr<AbstractZono<float_type>> cartesian_product(const AbstractZono<float_type>& Z1, 
    AbstractZono<float_type>& Z2)
{
    // make sure Z1 and Z2 both using same generator range
    if (Z1.zero_one_form != Z2.zero_one_form)
    {
        Z2.convert_form();
    }

    // declare
    std::vector<Eigen::Triplet<float_type>> tripvec;

    // take Cartesian product
    Eigen::SparseMatrix<float_type> Gc (Z1.n + Z2.n, Z1.nGc + Z2.nGc);
    get_triplets_offset(Z1.Gc, tripvec, 0, 0);
    get_triplets_offset(Z2.Gc, tripvec, Z1.n, Z1.nGc);
    Gc.setFromTriplets(tripvec.begin(), tripvec.end());

    tripvec.clear();
    Eigen::SparseMatrix<float_type> Gb (Z1.n + Z2.n, Z1.nGb + Z2.nGb);
    get_triplets_offset(Z1.Gb, tripvec, 0, 0);
    get_triplets_offset(Z2.Gb, tripvec, Z1.n, Z1.nGb);
    Gb.setFromTriplets(tripvec.begin(), tripvec.end());

    Eigen::Vector<float_type, -1> c (Z1.n + Z2.n);
    c.segment(0, Z1.n) = Z1.c;
    c.segment(Z1.n, Z2.n) = Z2.c;

    tripvec.clear();
    Eigen::SparseMatrix<float_type> Ac (Z1.nC + Z2.nC, Z1.nGc + Z2.nGc);
    get_triplets_offset(Z1.Ac, tripvec, 0, 0);
    get_triplets_offset(Z2.Ac, tripvec, Z1.nC, Z1.nGc);
    Ac.setFromTriplets(tripvec.begin(), tripvec.end());

    tripvec.clear();
    Eigen::SparseMatrix<float_type> Ab (Z1.nC + Z2.nC, Z1.nGb + Z2.nGb);
    get_triplets_offset(Z1.Ab, tripvec, 0, 0);
    get_triplets_offset(Z2.Ab, tripvec, Z1.nC, Z1.nGb);
    Ab.setFromTriplets(tripvec.begin(), tripvec.end());

    Eigen::Vector<float_type, -1> b (Z1.nC + Z2.nC);
    b.segment(0, Z1.nC) = Z1.b;
    b.segment(Z1.nC, Z2.nC) = Z2.b;

    // return correct output type
    if (Z1.is_hybzono() || Z2.is_hybzono())
        return std::make_unique<HybZono<float_type>>(Gc, Gb, c, Ac, Ab, b, Z1.zero_one_form);
    else if (Z1.is_conzono() || Z2.is_conzono())
        return std::make_unique<ConZono<float_type>>(Gc, c, Ac, b, Z1.zero_one_form);
    else if (Z1.is_zono() || Z2.is_zono())
        return std::make_unique<Zono<float_type>>(Gc, c, Z1.zero_one_form);
    else
        return std::make_unique<Point<float_type>>(c);
}


} // namespace ZonoOpt

#endif