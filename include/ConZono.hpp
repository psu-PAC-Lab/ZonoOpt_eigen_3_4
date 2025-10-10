#ifndef ZONOOPT_CONZONO_HPP_
#define ZONOOPT_CONZONO_HPP_

/**
 * @file ConZono.hpp
 * @author Josh Robbins (jrobbins@psu.edu)
 * @brief Constrained zonotope class for ZonoOpt library.
 * @version 1.0
 * @date 2025-06-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "HybZono.hpp"

namespace ZonoOpt
{

using namespace detail;

/**
 * @brief Constrained zonotope class.
 *
 * A constrained zonotope is defined as:
 * Z = {G * xi + c | A * xi = b, xi in [-1, 1]^nG}.
 * Equivalently, the following shorthand can be used: Z = <G, c, A, b>.
 * Optionally, in 0-1 form, the factors are xi in [0, 1]^nG.
 * The set dimension is n, and the number of equality constraints is nC.
 * 
 */
class ConZono : public HybZono
{
    public:

        // constructors

        /**
         * @brief Default constructor for ConZono class
         *
         */
        ConZono() { sharp = true; }

        /**
         * @brief ConZono constructor
         *
         * @param G generator matrix
         * @param c center
         * @param A constraint matrix
         * @param b constraint vector
         * @param zero_one_form true if set is in 0-1 form
         */
        ConZono(const Eigen::SparseMatrix<zono_float>& G, const Eigen::Vector<zono_float, -1>& c,
            const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b,
            const bool zero_one_form=false)
        {
            set(G, c, A, b, zero_one_form);
            sharp = true;
        }

        // virtual destructor
        ~ConZono() override = default;

        /**
         * @brief Clone method for polymorphic behavior.
         */
        HybZono* clone() const override
        {
            return new ConZono(*this);
        }

        // set method
        /**
         * @brief Reset constrained zonotope object with the given parameters.
         * 
         * @param G generator matrix
         * @param c center
         * @param A constraint matrix
         * @param b constraint vector
         * @param zero_one_form true if set is in 0-1 form
         */
        void set(const Eigen::SparseMatrix<zono_float>& G, const Eigen::Vector<zono_float, -1>& c,
            const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b, 
            bool zero_one_form=false);

        /**
         * @brief Execute constraint reduction algorithm from Scott et. al. 2016
         *
         * Removes one constraint and one generator from the constrained zonotope.
         * The resulting set is an over-approximation of the original set.
         */
        virtual void constraint_reduction();
        
        // generator conversion between [-1,1] and [0,1]
        void convert_form() override;

        // over-approximate as zonotope

        /**
         * @brief Compute outer approximation of constrained zonotope as zonotope using SVD
         * @return Zonotope over-approximation
         */
        virtual std::unique_ptr<Zono> to_zono_approx() const;

        // display methods
        std::string print() const override;

    protected:

        OptSolution qp_opt(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
            zono_float c, const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b,
            const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr) const;

        Eigen::Vector<zono_float, -1> do_optimize_over(
            const Eigen::SparseMatrix<zono_float> &P, const Eigen::Vector<zono_float, -1> &q, zono_float c,
            const OptSettings &settings, OptSolution* solution) const override;

        Eigen::Vector<zono_float, -1> do_project_point(const Eigen::Vector<zono_float, -1>& x,
            const OptSettings &settings, OptSolution* solution) const override;

        bool do_is_empty(const OptSettings &settings, OptSolution* solution) const override;

        zono_float do_support(const Eigen::Vector<zono_float, -1>& d, const OptSettings &settings,
            OptSolution* solution) override;

        bool do_contains_point(const Eigen::Vector<zono_float, -1>& x, const OptSettings &settings,
            OptSolution* solution) const override;

        Box do_bounding_box(const OptSettings &settings, OptSolution*) override;

        std::unique_ptr<HybZono> do_complement(zono_float delta_m, bool, const OptSettings&,
            OptSolution*, int, int) override;
};

// forward declarations

/**
* @brief Builds a constrained zonotope from a vertex representation polytope.
*
* @param Vpoly vertices of V-rep polytope
* @return constrained zonotope
* @ingroup ZonoOpt_SetupFunctions
*
* Vpoly is a matrix where each row is a vertex of the polytope.
*/
std::unique_ptr<ConZono> vrep_2_conzono(const Eigen::Matrix<zono_float, -1, -1> &Vpoly);


// implementation
inline void ConZono::set(const Eigen::SparseMatrix<zono_float>& G, const Eigen::Vector<zono_float, -1>& c,
    const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b, 
    const bool zero_one_form)
{
    // check dimensions
    if (G.rows() != c.size() || A.rows() != b.size() || G.cols() != A.cols())
    {
    throw std::invalid_argument("ConZono: inconsistent dimensions.");
    }

    // conzono parameters
    this->G = G;
    this->A = A;
    this->c = c;
    this->b = b;
    this->nG = static_cast<int>(G.cols());
    this->nC = static_cast<int>(A.rows());
    this->n = static_cast<int>(G.rows());
    this->zero_one_form = zero_one_form;

    // abstract zono parameters
    this->nGc = this->nG;
    this->nGb = 0;
    this->Gc = this->G;
    this->Gb.resize(this->n, 0);
    this->Ac = this->A;
    this->Ab.resize(0, 0);
}

inline void ConZono::convert_form()
{
    Eigen::Vector<zono_float, -1> c, b;
    Eigen::SparseMatrix<zono_float> G, A;

    if (!this->zero_one_form) // convert to [0,1] generators
    {
        c = this->c - this->G*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        b = this->b + this->A*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        G = 2.0*this->G;
        A = 2.0*this->A;

        set(G, c, A, b, true);
    }
    else // convert to [-1,1] generators
    {
        c = this->c + 0.5*this->G*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        b = this->b - 0.5*this->A*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        G = 0.5*this->G;
        A = 0.5*this->A;

        set(G, c, A, b, false);
    }
}

inline std::string ConZono::print() const
{
    std::stringstream ss;
    ss << "ConZono: " << std::endl;
    ss << "n: " << this->n << std::endl;
    ss << "nG: " << this->nG << std::endl;
    ss << "nC: " << this->nC << std::endl;
    ss << "G: " << Eigen::Matrix<zono_float, -1, -1>(this->G) << std::endl;
    ss << "c: " << this->c << std::endl;
    ss << "A: " << Eigen::Matrix<zono_float, -1, -1>(this->A) << std::endl;
    ss << "b: " << this->b << std::endl;
    ss << "zero_one_form: " << this->zero_one_form;
    return ss.str();
}

inline Eigen::Vector<zono_float, -1> ConZono::do_optimize_over(
    const Eigen::SparseMatrix<zono_float> &P, const Eigen::Vector<zono_float, -1> &q, const zono_float c,
    const OptSettings &settings, OptSolution* solution) const
{
    // check dimensions
    if (P.rows() != this->n || P.cols() != this->n || q.size() != this->n)
    {
        throw std::invalid_argument("Optimize over: inconsistent dimensions.");
    }

    // cost
    Eigen::SparseMatrix<zono_float> P_fact = this->G.transpose()*P*this->G;
    Eigen::Vector<zono_float, -1> q_fact = this->G.transpose()*(P*this->c + q);
    zono_float delta_c = (0.5*this->c.transpose()*P*this->c + q.transpose()*this->c)(0);

    // solve QP
    OptSolution sol = this->qp_opt(P_fact, q_fact, c+delta_c, this->A, this->b, settings, solution);

    // check feasibility and return solution
    if (sol.infeasible)
        return Eigen::Vector<zono_float, -1>(this->nG);
    else
        return this->G*sol.z + this->c;
}

inline Eigen::Vector<zono_float, -1> ConZono::do_project_point(const Eigen::Vector<zono_float, -1>& x,
    const OptSettings& settings, OptSolution* solution) const
{
    // check dimensions
    if (this->n != x.size())
    {
        throw std::invalid_argument("Point projection: inconsistent dimensions.");
    }

    // cost
    Eigen::SparseMatrix<zono_float> P = this->G.transpose()*this->G;
    Eigen::Vector<zono_float, -1> q = this->G.transpose()*(this->c-x);
    
    // solve QP
    const OptSolution sol = this->qp_opt(P, q, 0, this->A, this->b, settings, solution);

    // check feasibility and return solution
    if (sol.infeasible)
        throw std::invalid_argument("Point projection: infeasible");
    else
        return this->G*sol.z + this->c;
}

inline bool ConZono::do_is_empty(const OptSettings& settings, OptSolution* solution) const
{
    // trivial case
    if (this->n == 0)
        return true;

    // cost
    Eigen::SparseMatrix<zono_float> P (this->nG, this->nG);
    P.setIdentity();
    Eigen::Vector<zono_float, -1> q = Eigen::Vector<zono_float, -1>::Zero(this->nG);

    // solve QP
    OptSolution sol = this->qp_opt(P, q, 0, this->A, this->b, settings, solution);

    // check infeasibility flag
    return sol.infeasible;
}

inline zono_float ConZono::do_support(const Eigen::Vector<zono_float, -1>& d,
            const OptSettings& settings, OptSolution* solution)
{
    // check dimensions
    if (this->n != d.size())
    {
        throw std::invalid_argument("Support: inconsistent dimensions.");
    }

    // cost
    Eigen::SparseMatrix<zono_float> P (this->nG, this->nG);
    Eigen::Vector<zono_float, -1> q = -this->G.transpose()*d;
    
    // solve QP
    const OptSolution sol = this->qp_opt(P, q, 0, this->A, this->b, settings, solution);

    // check feasibility and return solution
    if (sol.infeasible) // Z is empty
        throw std::invalid_argument("Support: infeasible");
    else
        return d.dot(this->G*sol.z + this->c);
}

inline bool ConZono::do_contains_point(const Eigen::Vector<zono_float, -1>& x,
    const OptSettings& settings, OptSolution* solution) const
{
    // check dimensions
    if (this->n != x.size())
    {
        throw std::invalid_argument("Contains point: inconsistent dimensions.");
    }

    // build QP for ADMM
    Eigen::SparseMatrix<zono_float> P (this->nG, this->nG);
    P.setIdentity();
    Eigen::Vector<zono_float, -1> q (this->nG);
    q.setZero(); // zeros
    Eigen::SparseMatrix<zono_float> A = vcat<zono_float>(this->A, this->G);
    Eigen::Vector<zono_float, -1> b (this->nC + this->n);
    b.segment(0, this->nC) = this->b;
    b.segment(this->nC, this->n) = x-this->c;

    const OptSolution sol = this->qp_opt(P, q, 0, A, b, settings, solution);

    // check feasibility and return solution
    return !(sol.infeasible);
}

inline OptSolution ConZono::qp_opt(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
    const zono_float c, const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b,
    const OptSettings &settings, OptSolution* solution) const
{
    // setup QP
    Eigen::Vector<zono_float, -1> xi_lb (this->nG);
    if (this->zero_one_form)
        xi_lb.setZero();
    else
        xi_lb.setConstant(-1);
    const Eigen::Vector<zono_float, -1> xi_ub = Eigen::Vector<zono_float, -1>::Ones(this->nG);

    const auto data = std::make_shared<ADMM_data>(P, q, A, b, xi_lb, xi_ub, c, settings);
    ADMM_solver solver(data);

    // solve
    OptSolution sol = solver.solve();
    if (solution != nullptr)
        *solution = sol;
    return sol;
}

// bounding box
inline Box ConZono::do_bounding_box(const OptSettings &settings, OptSolution*)
{
    // make sure dimension is at least 1
    if (this->n == 0)
    {
        throw std::invalid_argument("Bounding box: empty set");
    }

    // init search direction for bounding box
    Eigen::Vector<zono_float, -1> d (this->n);
    d.setZero();

    // declarations
    Box box (this->n); // init
    // declare
    zono_float s_neg, s_pos;

    // build QP for ADMM
    const Eigen::SparseMatrix<zono_float> P (this->nG, this->nG);
    Eigen::Vector<zono_float, -1> q = -this->G.transpose()*d;

    // convex solution
    Eigen::Vector<zono_float, -1> xi_lb, xi_ub;
    if (this->zero_one_form)
        xi_lb = Eigen::Vector<zono_float, -1>::Zero(this->nG);
    else
        xi_lb = -1.0*Eigen::Vector<zono_float, -1>::Ones(this->nG);

    xi_ub = Eigen::Vector<zono_float, -1>::Ones(this->nG);

    // build ADMM object
    const auto data = std::make_shared<ADMM_data>(P, q, this->A, this->b, xi_lb, xi_ub, zero, settings);
    ADMM_solver solver(data);

    // get support in all box directions
    for (int i=0; i<this->n; i++)
    {
        // negative direction

        // update QP
        d.setZero();
        d(i) = -1;
        data->q = -this->G.transpose()*d;

        // solve
        OptSolution sol = solver.solve();
        if (sol.infeasible)
            throw std::invalid_argument("Bounding box: Z is empty");
        else
            s_neg = -d.dot(this->G*sol.z + this->c);

        // positive direction

        // update QP
        d.setZero();
        d(i) = 1;
        data->q = -this->G.transpose()*d;

        // solve
        sol = solver.solve();
        if (sol.infeasible)
            throw std::invalid_argument("Bounding box: Z is empty");
        else
            s_pos = d.dot(this->G*sol.z + this->c);

        // store bounds
        box[i] = Interval(s_neg, s_pos);
    }

    return box;
}



} // namespace ZonoOpt


#endif