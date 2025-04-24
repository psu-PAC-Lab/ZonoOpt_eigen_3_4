#ifndef __ZONOOPT_ADMM_HPP__
#define __ZONOOPT_ADMM_HPP__

#include <vector>
#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "CholeskyUtilities.hpp"
#include "IntervalUtilities.hpp"
#include "SparseMatrixUtilities.hpp"

/* 
    Primary reference: 
    Boyd, Stephen, et al. 
    "Distributed optimization and statistical learning via the alternating direction method of multipliers." 
    Foundations and Trends® in Machine learning 3.1 (2011): 1-122.
*/

namespace ZonoOpt
{

template <typename float_type>
struct ADMM_settings
{
    float_type rho = 1; // init value when variable_rho is true
    float_type t_max = std::numeric_limits<float_type>::max();
    int k_max = 5000;
    float_type eps_dual = 1e-2; // convergence tolerance 
    float_type eps_prim = 1e-2; // convergence tolerance
    int k_inf_check = 10; // check infeasibility every k_inf_check iterations
    bool inf_norm_conv = false; // use infinity norm for convergence check (if false, scaled 2-norm is used)
    bool verbose = false;
    int verbosity_interval = 10; // print every verbose_interval iterations
};

template <typename float_type>
struct ADMM_solution
{
    Eigen::Vector<float_type, -1> x, z, u;
    float_type J;
    float_type primal_residual, dual_residual;
    float_type run_time, startup_time;
    int k;
    bool converged, infeasible;
};

// print function
static void print_str(std::stringstream &ss)
{
    #ifdef IS_PYTHON_ENV
    py::print(ss.str());
    #else
    std::cout << ss.str() << std::endl;
    #endif
    ss.str("");
}


// solver based on sparse LDLT decomposition
template <typename float_type>
class ADMM_solver
{
    public:

        // constructor
        ADMM_solver() = default;

        // setup
        void setup(const Eigen::SparseMatrix<float_type> &P,
            const Eigen::Vector<float_type, -1>& q, 
            const Eigen::SparseMatrix<float_type> &A,  
            const Eigen::Vector<float_type, -1>& b, 
            const Eigen::Vector<float_type, -1>& x_l, 
            const Eigen::Vector<float_type, -1>& x_u,
            const ADMM_settings<float_type> & settings = ADMM_settings<float_type>(),
            float_type c = 0)
        {
            // dimensions
            this->n_x = P.rows();
            this->n_cons = A.rows();
            this->sqrt_n_x = std::sqrt((float_type) this->n_x);

            // check dimension consistency
            if (P.cols() != this->n_x || q.size() != this->n_x || A.cols() != this->n_x || b.size() != this->n_cons ||
                x_l.size() != this->n_x || x_u.size() != this->n_x)
            {
                throw std::invalid_argument("ADMM setup: dimension mismatch.");
            }

            // check settings validity
            if (!settings_check(settings))
            {
                throw std::invalid_argument("ADMM setup: invalid settings.");
            }

            // store data
            this->P = P;
            this->q = q;
            this->A = A;
            this->AT = A.transpose();
            this->b = b;
            this->c(0) = c;

            this->x_l = x_l;
            this->x_u = x_u;

            this->x_box.clear();
            for (int i=0; i<x_l.size(); i++)
                this->x_box.push_back(Interval<float_type>(x_l(i), x_u(i)));

            // settings
            this->settings = settings;
            this->settings.rho = settings.rho;

            // flags
            this->is_warmstarted = false;
            this->factorization_system_valid = false;
            this->factorization_AAT_valid = false;
        }

        // update methods
        void update_P(const Eigen::SparseMatrix<float_type>& P)
        {
            this->P = P;
            this->n_x = this->P.rows();
            this->sqrt_n_x = std::sqrt((float_type) this->n_x);

            // flags
            this->factorization_system_valid = false;
        }

        void update_q(const Eigen::Vector<float_type, -1>& q)
        {
            this->q = q;
        }

        void update_A(const Eigen::SparseMatrix<float_type>& A)
        {
            this->A = A;
            this->AT = A.transpose();
            this->n_cons = this->A.rows();

            // flags
            this->factorization_system_valid = false;
            this->factorization_AAT_valid = false;
        }

        void update_b(const Eigen::Vector<float_type, -1>& b)
        {
            this->b = b;
        }

        void update_bounds(const Eigen::Vector<float_type, -1>& x_l, 
            const Eigen::Vector<float_type, -1>& x_u)
        {
            this->x_l = x_l;
            this->x_u = x_u;

            this->x_box.clear();
            for (int i=0; i<x_l.size(); i++)
                this->x_box.push_back(Interval<float_type>(x_l(i), x_u(i)));
        }

        void update_c(float_type c)
        {
            this->c(0) = c;
        }

        void update_settings(const ADMM_settings<float_type>& settings)
        {
            // check settings validity
            if (!settings_check(settings))
            {
                throw std::invalid_argument("ADMM setup: invalid settings.");
            }

            if (settings.rho != this->settings.rho)
                this->factorization_system_valid = false;
            this->settings = settings;
        }

        // warm start
        void warmstart(const Eigen::Vector<float_type, -1>& x0, 
            const Eigen::Vector<float_type, -1>& u0)
        {
            // copy in warm start variables
            this->x0 = x0;
            this->u0 = u0;
            
            // set flag
            this->is_warmstarted = true;
        }

        // optional pre-factorization
        void factorize()
        {
            this->factorize_system();
            this->factorize_AAT();
        }

        // solve
        ADMM_solution<float_type> solve()
        {
            // start timer
            auto start = std::chrono::high_resolution_clock::now();
            auto t0 = std::chrono::high_resolution_clock::now();
            float_type run_time;

            // verbosity
            std::stringstream ss;

            // check that problem data is consistent
            if (!this->check_problem_dimensions())
            {
                throw std::invalid_argument("ADMM solve: inconsistent problem data dimensions.");
            }
            if (this->settings.verbose)
            {
                ss << "Solving ADMM problem with " << this->n_x << " variables and " << this->n_cons << " constraints.";
                print_str(ss);
            }

            // check if equilibration is required
            if (!this->factorization_system_valid)
            {
                t0 = std::chrono::high_resolution_clock::now();
                this->factorize_system();
                if (this->settings.verbose)
                {
                    run_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    ss << "M factorization time = " << run_time << " sec";
                    print_str(ss);
                }
            }
            if (!this->factorization_AAT_valid)
            {
                t0 = std::chrono::high_resolution_clock::now();
                this->factorize_AAT();
                if (this->settings.verbose)
                {
                    run_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    ss << "A*A^T factorization time = " << run_time << " sec";
                    print_str(ss);
                }
            }

            // initial values
            Eigen::Vector<float_type, -1> xk, zk, uk, zkm1, rhs, x_nu;
            if (this->is_warmstarted)
            {
                xk = this->x0;
                uk = this->u0;
            }
            else
            {
                xk = 0.5*(this->x_l + this->x_u);
                uk = Eigen::Vector<float_type, -1>::Zero(this->n_x);
            }
            zk = xk;
            rhs = Eigen::Vector<float_type, -1>::Zero(this->n_x + this->n_cons);
            rhs.segment(this->n_x, this->n_cons) = this->b; // unchanging
            zkm1 = zk;

            // init residuals
            float_type rp_k, rd_k;

            // init loop
            int k = 0;
            run_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count();
            bool converged = false, infeasible = false;

            // log startup time
            float_type startup_time = run_time;

            while ((k < this->settings.k_max) && (run_time < this->settings.t_max) && !converged && !infeasible)
            {
                // x update
                rhs.segment(0, this->n_x) = -this->q + this->settings.rho*(zk - uk);
                x_nu = solve_LDLT<float_type>(this->ldlt_data_system, rhs);
                xk = x_nu.segment(0, this->n_x);

                // z update
                zk = xk + uk;
                zk = zk.cwiseMax(this->x_l).cwiseMin(this->x_u);

                // u update
                uk += xk - zk;

                // check for infeasibility certificate
                if (k % this->settings.k_inf_check == 0)
                {
                    infeasible = this->is_infeasibility_certificate(zk - xk, xk);
                    if (this->settings.verbose && infeasible)
                    {
                        ss << "Infeasibility certificate detected at iteration " << k;
                        print_str(ss);
                    }
                }

                // check convergence
                if (this->settings.inf_norm_conv)
                {
                    rp_k = (xk - zk).cwiseAbs().maxCoeff();
                    rd_k = this->settings.rho*(zk - zkm1).cwiseAbs().maxCoeff();
                    converged = (rp_k < this->settings.eps_prim && rd_k < this->settings.eps_dual);
                }
                else
                {
                    rp_k = (xk - zk).norm();
                    rd_k = this->settings.rho*(zk - zkm1).norm();
                    converged = (rp_k < this->sqrt_n_x*this->settings.eps_prim && rd_k < this->sqrt_n_x*this->settings.eps_dual);
                }

                // increment
                zkm1 = zk;
                k++;

                // get time
                run_time = 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count();

                // verbosity
                if (this->settings.verbose && (k % this->settings.verbosity_interval == 0))
                {
                    ss << "k = " << k << ": primal residual = " << rp_k << ", dual residual = " 
                        << rd_k << ", run time = " << run_time << " sec";
                    print_str(ss);
                }
            }

            // verbosity
            if (this->settings.verbose)
            {
                if (converged)
                {
                    ss << "ADMM converged in " << k << " iterations.";
                    print_str(ss);
                }
                else if (infeasible)
                {
                    ss << "ADMM detected infeasibility.";
                    print_str(ss);
                }
                else
                {
                    ss << "ADMM did not converge in " << k << " iterations.";
                    print_str(ss);
                }
            }

            // reset flags
            this->is_warmstarted = false;

            // build output
            ADMM_solution<float_type> solution;
            solution.x = xk;
            solution.z = zk;
            solution.u = uk;
            solution.J = (0.5*zk.transpose()*this->P*zk + this->q.transpose()*zk + this->c)(0);
            solution.primal_residual = rp_k;
            solution.dual_residual = rd_k;
            solution.run_time = run_time;
            solution.startup_time = startup_time;
            solution.k = k;
            solution.converged = converged;
            solution.infeasible = infeasible;

            return solution;
        }

    protected:
        
        // problem data
        Eigen::SparseMatrix<float_type> P, A, AT;
        Eigen::Vector<float_type, -1> q, b, x_l, x_u;
        Eigen::Vector<float_type, 1> c;
        LDLT_data<float_type> ldlt_data_system, ldlt_data_AAT;
        int n_x, n_cons;
        float_type sqrt_n_x;
        Box<float_type> x_box;

        // warm start
        Eigen::Vector<float_type, -1> x0, u0;

        // settings
        ADMM_settings<float_type> settings;

        // LDLT solver
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float_type>> ldlt_solver_system, ldlt_solver_AAT;

        // flags
        bool is_warmstarted = false, factorization_system_valid = false, factorization_AAT_valid = false;

        // factor problem data
        void factorize_system()
        {
            // system matrix
            Eigen::SparseMatrix<float_type> M (this->n_x + this->n_cons, this->n_x + this->n_cons);
            
            Eigen::SparseMatrix<float_type> I (this->n_x, this->n_x);
            I.setIdentity();
            Eigen::SparseMatrix<float_type> Phi = this->P + this->settings.rho*I;
            
            std::vector<Eigen::Triplet<float_type>> triplets;
            get_triplets_offset<float_type>(Phi, triplets, 0, 0);
            get_triplets_offset<float_type>(this->A, triplets, this->n_x, 0);
            get_triplets_offset<float_type>(this->AT, triplets, 0, this->n_x);
            M.setFromTriplets(triplets.begin(), triplets.end());

            // factorize system matrix
            this->ldlt_solver_system.compute(M);
            get_LDLT_data<float_type>(this->ldlt_solver_system, this->ldlt_data_system);

            if (this->ldlt_solver_system.info() != Eigen::Success)
                throw std::runtime_error("ADMM: factorization of problem data failed, most likely A is not full row rank"); 

            // set flag
            this->factorization_system_valid = true;
        }

        void factorize_AAT()
        {
            // factorize A*AT
            Eigen::SparseMatrix<float_type> AAT = this->A*this->AT;
            this->ldlt_solver_AAT.compute(AAT);
            get_LDLT_data<float_type>(this->ldlt_solver_AAT, this->ldlt_data_AAT);

            if (this->ldlt_solver_AAT.info() != Eigen::Success)
                throw std::runtime_error("ADMM: factorization of A*A^T failed, most likely A is not full row rank");

            // set flag
            this->factorization_AAT_valid = true;
        }

        // check for infeasibility certificate
        bool is_infeasibility_certificate(const Eigen::Vector<float_type, -1>& ek, 
            const Eigen::Vector<float_type, -1>& xk) const
        {
            // project ek onto row space of A (i.e. column space of AT)
            Eigen::Vector<float_type, -1> A_e = this->A*ek;
            Eigen::Vector<float_type, -1> AAT_inv_A_e = solve_LDLT<float_type>(this->ldlt_data_AAT, A_e);
            Eigen::Vector<float_type, -1> ek_proj = this->AT*AAT_inv_A_e;

            // check if this is an infeasibility certificate
            float_type e_x = ek_proj.dot(xk);
            Interval<float_type> e_box = this->x_box.dot(ek_proj);
            return !e_box.contains(e_x, Eigen::NumTraits<float_type>::dummy_precision());
        }

        // check settings are valid
        bool settings_check(const ADMM_settings<float_type>& settings) const
        {
            return (settings.rho > 0 && settings.k_max > 0 && 
                settings.eps_dual >= 0 && settings.eps_prim >= 0 && 
                settings.k_inf_check >= 0 && settings.verbosity_interval > 0);
        };

        bool check_problem_dimensions() const
        {
            bool prob_data_consistent = (this->P.rows() == this->n_x && this->P.cols() == this->n_x && this->q.size() == this->n_x &&
                this->A.rows() == this->n_cons && this->A.cols() == this->n_x && this->b.size() == this->n_cons &&
                this->x_l.size() == this->n_x && this->x_u.size() == this->n_x);
            
            bool warm_start_consistent;
            if (this->is_warmstarted)
                warm_start_consistent = (this->x0.size() == this->n_x && this->u0.size() == this->n_x);
            else
                warm_start_consistent = true;

            return prob_data_consistent && warm_start_consistent;
        }
};


} // end namespace ZonoOpt

#endif