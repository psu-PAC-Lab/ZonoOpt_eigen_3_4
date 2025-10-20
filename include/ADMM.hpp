#ifndef ZONOOPT_ADMM_HPP_
#define ZONOOPT_ADMM_HPP_

/**
 * @file ADMM.hpp
 * @author Josh Robbins (jrobbins@psu.edu)
 * @brief ADMM implementation used within ZonoOpt.
 * @version 1.0
 * @date 2025-06-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */


#include <vector>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <memory>
#include <set>
#include <cmath>
#include <atomic>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "CholeskyUtilities.hpp"
#include "Intervals.hpp"
#include "SparseMatrixUtilities.hpp"
#include "SolverDataStructures.hpp"

/* 
    Primary reference: 
    Boyd, Stephen, et al. 
    "Distributed optimization and statistical learning via the alternating direction method of multipliers." 
    Foundations and Trends® in Machine learning 3.1 (2011): 1-122.
*/

namespace ZonoOpt::detail {
    /**
     * @brief Data structure for ADMM solver.
     *
     */
    struct ADMM_data : std::enable_shared_from_this<ADMM_data>
    {
        Eigen::SparseMatrix<zono_float> P, A, AT;
        Eigen::SparseMatrix<zono_float, Eigen::RowMajor> A_rm;
        Eigen::Vector<zono_float, -1> q, b;
        Eigen::Vector<zono_float, 1> c;
        LDLT_data ldlt_data_M, ldlt_data_AAT;
        int n_x, n_cons;
        zono_float sqrt_n_x;
        std::shared_ptr<Box> x_box;
        OptSettings settings;

        // constructor
        ADMM_data() = default;

        ADMM_data(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
            const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b,
            const Eigen::Vector<zono_float, -1>& x_l, const Eigen::Vector<zono_float, -1>& x_u,
            const zono_float c=0, const OptSettings& settings= OptSettings())
        {
            set(P, q, A, b, x_l, x_u, c, settings);
        }

        // set method
        void set(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
            const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b,
            const Eigen::Vector<zono_float, -1>& x_l, const Eigen::Vector<zono_float, -1>& x_u,
            zono_float c=0, const OptSettings& settings= OptSettings());

        // clone method
        ADMM_data* clone() const;
    };

    // utilities

    /**
     * @brief ADMM solver targeted at constrained zonotope optimization problems.
     *
     */
    class ADMM_solver
    {
    public:

        /**
         * @brief Construct a new admm solver object
         *
         * @param data
         */
        explicit ADMM_solver(const ADMM_data& data);

        /**
         * @brief Construct a new admm solver object
         *
         * @param data
         */
        explicit ADMM_solver(const std::shared_ptr<ADMM_data>& data);

        /**
         * @brief Construct a new admm solver object
         *
         * @param other
         */
        ADMM_solver(const ADMM_solver& other);

        /**
         * @brief Destroy the admm solver object
         *
         */
        virtual ~ADMM_solver() = default;

        /**
         * @brief Warm-starts ADMM solver with primal and dual variables.
         *
         * @param x0
         * @param u0
         */
        virtual void warmstart(const Eigen::Vector<zono_float, -1>& x0,
            const Eigen::Vector<zono_float, -1>& u0);

        /**
         * @brief Optional pre-factorization of problem matrices.
         *
         */
        virtual void factorize();

        /**
         * @brief Solves optimization problem using ADMM.
         *
         * @param stop
         * @return OptSolution
         */
        OptSolution solve(std::atomic<bool>* stop);
        OptSolution solve();

    protected:

        // protected fields
        std::shared_ptr<ADMM_data> data;
        zono_float eps_prim=static_cast<zono_float>(1e-3), eps_dual=static_cast<zono_float>(1e-3);

        // startup method
        bool startup(Box& x_box, OptSolution& solution, const std::set<int>& contract_inds=std::set<int>());

        // core solve method
        virtual void solve_core(const Box& x_box, OptSolution& solution, std::atomic<bool>* stop);

        // warm start
        Eigen::Vector<zono_float, -1> x0, u0;

        // flags
        bool is_warmstarted = false;

        // factor problem data
        void factorize_M() const;

        void factorize_AAT() const;

        // check for infeasibility certificate
        bool is_infeasibility_certificate(const Eigen::Vector<zono_float, -1>& ek,
            const Eigen::Vector<zono_float, -1>& xk, const Box& x_box) const;

        bool check_problem_dimensions() const;
    };

} // end namespace ZonoOpt::detail

#endif