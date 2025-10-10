#ifndef ZONOOPT_HYBZONO_HPP_
#define ZONOOPT_HYBZONO_HPP_

/**
 * @file HybZono.hpp
 * @author Josh Robbins (jrobbins@psu.edu)
 * @brief Hybrid zonotope class for ZonoOpt library.
 * @version 1.0
 * @date 2025-06-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "SparseMatrixUtilities.hpp"
#include "MI_Solver.hpp"
#include "MI_DataStructures.hpp"
#include "Inequality.hpp"
#include "Intervals.hpp"

#include <stdexcept>
#include <limits>
#include <set>

namespace ZonoOpt
{

using namespace detail;

// forward declarations
class Zono;
class ConZono;

/**
 * @brief Hybrid zonotope class
 * 
 * A hybrid zonotope is defined as:
 * Z = {Gc * xi_c + Gb * xi_b + c | Ac * xi_c + Ab * xi_b = b, xi_c in [-1, 1]^nGc, xi_b in {-1, 1}^nGb}.
 * Equivalently, the following shorthand can be used: Z = <Gc, Gb, c, Ac, Ab, b>.
 * Optionally, in 0-1 form, the factors are xi_c in [0, 1]^nGc, xi_b in {0, 1}^nGb. 
 * The set dimension is n, and the number of equality constraints is nC.
 */
class HybZono
{
    public:
        
        // constructors

        /**
         * @brief Default constructor for HybZono class
         * 
         */
        HybZono() = default;

        /**
         * @brief HybZono constructor
         * 
         * @param Gc continuous generator matrix
         * @param Gb binary generator matrix
         * @param c center
         * @param Ac continuous constraint matrix
         * @param Ab binary constraint matrix
         * @param b constraint vector
         * @param zero_one_form true if set is in 0-1 form
         * @param sharp true if set is known to be sharp, i.e., convex relaxation = convex hull
         */
        HybZono(const Eigen::SparseMatrix<zono_float>& Gc, const Eigen::SparseMatrix<zono_float>& Gb, const Eigen::Vector<zono_float, -1>& c,
            const Eigen::SparseMatrix<zono_float>& Ac, const Eigen::SparseMatrix<zono_float>& Ab, const Eigen::Vector<zono_float, -1>& b,
            const bool zero_one_form=false, const bool sharp=false)
        {
            set(Gc, Gb, c, Ac, Ab, b, zero_one_form, sharp);
        } 

        // virtual destructor
        virtual ~HybZono() = default;

        /**
         * @brief Reset hybrid zonotope object with the given parameters.
         * 
         * @param Gc continuous generator matrix
         * @param Gb binary generator matrix
         * @param c center
         * @param Ac continuous constraint matrix
         * @param Ab binary constraint matrix
         * @param b constraint vector
         * @param zero_one_form true if set is in 0-1 form
         * @param sharp true if set is known to be sharp, i.e., convex relaxation = convex hull
         */
        void set(const Eigen::SparseMatrix<zono_float>& Gc, const Eigen::SparseMatrix<zono_float>& Gb, const Eigen::Vector<zono_float, -1>& c,
            const Eigen::SparseMatrix<zono_float>& Ac, const Eigen::SparseMatrix<zono_float>& Ab, const Eigen::Vector<zono_float, -1>& b,
            bool zero_one_form=false, bool sharp=false);

        /**
         * @brief Clone method for polymorphic behavior.
         */
        virtual HybZono* clone() const
        {
            return new HybZono(*this);
        }

        // get methods

        /**
         * @brief Returns dimension of set
         * 
         * @return n 
         */
        virtual int get_n() const { return this->n; }

        /**
         * @brief Returns number of constraints in set definition
         * 
         * @return nC 
         */
        virtual int get_nC() const { return this->nC; }

        /**
         * @brief Returns number of generators in set definition
         * 
         * @return nG 
         */
        virtual int get_nG() const { return this->nG; }

        /**
         * @brief Returns number of continuous generators in set definition
         * 
         * @return nGc 
         */
        virtual int get_nGc() const { return this->nGc; }

        /**
         * @brief Returns number of binary generators in set definition
         * 
         * @return nGb 
         */
        virtual int get_nGb() const { return this->nGb; }

        /**
         * @brief Returns continuous generator matrix
         * 
         * @return Gc
         */
        virtual Eigen::SparseMatrix<zono_float> get_Gc() const { return this->Gc; }

        /**
         * @brief Returns binary generator matrix
         * 
         * @return Gb
         */
        virtual Eigen::SparseMatrix<zono_float> get_Gb() const { return this->Gb; }

        /**
         * @brief Returns generator matrix
         * 
         * @return G 
         */
        virtual Eigen::SparseMatrix<zono_float> get_G() const { return this->G; }

        /**
         * @brief Returns continuous constraint matrix
         * 
         * @return Ac
         */
        virtual Eigen::SparseMatrix<zono_float> get_Ac() const { return this->Ac; }

        /**
         * @brief Returns binary constraint matrix
         * 
         * @return Ab
         */
        virtual Eigen::SparseMatrix<zono_float> get_Ab() const { return this->Ab; }

        /**
         * @brief Returns constraint matrix
         * 
         * @return A
         */
        virtual Eigen::SparseMatrix<zono_float> get_A() const { return this->A; }

        /**
         * @brief Returns center vector
         * 
         * @return c
         */
        virtual Eigen::Vector<zono_float, -1> get_c() const { return this->c; }
        
        /**
         * @brief Returns constraint vector
         * 
         * @return b
         */
        virtual Eigen::Vector<zono_float, -1> get_b() const { return this->b; }

        /**
         * @brief Returns true if factors are in range [0,1], false if they are in range [-1,1].
         * 
         * @return zero_one_form flag
         */
        virtual bool is_0_1_form() const { return this->zero_one_form; }

        /**
         * @brief Returns true if set is known to be sharp
         *
         * @return sharp flag
         * 
         * A set is sharp if its convex relaxation is equal to its convex hull.
         */
        bool is_sharp() const { return this->sharp; }

        /**
         * @brief Converts the set representation between -1-1 and 0-1 forms.
         * 
         * This method converts the set representation between -1-1 and 0-1 forms. 
         * If the set is in -1-1 form, then xi_c in [-1,1] and xi_b in {-1,1}.
         * If the set is in 0-1 form, then xi_c in [0,1] and xi_b in {0,1}.
         */
        virtual void convert_form();

        /**
         * @brief Removes redundant constraints and any unused generators
         * @param contractor_iter number of interval contractor iterations to run
         * 
         * This method uses an interval contractor to detect generators that can be removed. 
         * Additionally, any linearly dependent rows of the constraint matrix A are removed.
         * If the linearly dependent constraints are not consistent (e.g., if A = [1, 0.1; 1, 0.1] and b = [1; 0.8]), 
         * the returned set is not equivalent to the original set.
         * Unused factors are also removed.
         */
        virtual void remove_redundancy(int contractor_iter=100);

        /**
         * @brief Returns convex relaxation of the hybrid zonotope.
         * 
         * @return Constrained zonotope Z = <[Gc, Gb], c, [Ac, Ab,], b>
         * 
         * This method returns the convex relaxation of the hybrid zonotope.
         * If the set is sharp, the convex relaxation is the convex hull.
         */
        virtual std::unique_ptr<ConZono> convex_relaxation() const;

        /**
         * @brief Computes the complement of the set Z.
         *
         * @param delta_m parameter defining range of complement
         * @param remove_redundancy remove redundant constraints and unused generators in get_leaves function call
         * @param settings optimization settings for get_leaves function call
         * @param solution optimization solution for get_leaves function call
         * @param n_leaves maximum number of leaves to return in get_leaves function call
         * @param contractor_iter number of interval contractor iterations in remove_redundancy if using
         * @return Hybrid zonotope complement of the given set
         *
         * Computes the complement according to the method of Bird and Jain:
         * "Unions and Complements of Hybrid Zonotopes"
         * delta_m is a parameter that defines the set over which the complement is defined.
         * For a constrained zonotope, the complement is restricted to the set
         * X = {G \xi + c | A \xi = b, \xi \in [-1-delta_m, 1+delta+m]^{nG}}.
         */
        virtual std::unique_ptr<HybZono> complement(const zono_float delta_m = 100, const bool remove_redundancy=true, const OptSettings &settings=OptSettings(),
            OptSolution* solution=nullptr, const int n_leaves = std::numeric_limits<int>::max(), const int contractor_iter=100)
        {
            return do_complement(delta_m, remove_redundancy, settings, solution, n_leaves, contractor_iter);
        }

        // type checking
        /**
         * @brief Polymorphic type checking: true if set is a point
         * 
         */
        bool is_point() const;

        /**
         * @brief Polymorphic type checking: true if set is a zonotope
         * 
         */
        bool is_zono() const;

        /**
         * @brief Polymorphic type checking: true if set is a constrained zonotope
         * 
         */
        bool is_conzono() const;

        /**
         * @brief Polymorphic type checking: true if set is a hybrid zonotope
         * 
         */
        bool is_hybzono() const;

        /**
         * @brief Polymorphic type checking: true if set is empty set object
         */
        bool is_empty_set() const;
        
        // display methods
        /**
         * @brief Returns set information as a string
         * 
         */
        virtual std::string print() const;

        /**
         * @brief Displays set information to the given output stream.
         * 
         * @param os 
         * @param Z 
         */
        friend std::ostream& operator<<(std::ostream& os, const HybZono& Z);

        // optimization
        /**
         * @brief Solves optimization problem with quadratic objective over the current set
         * 
         * @param P quadratic objective matrix
         * @param q linear objective vector
         * @param c constant term in objective function
         * @param settings optimization settings structure
         * @param solution optimization solution structure pointer, populated with result
         * @return point z in the current set
         * 
         * Solves optimization problem of the form min 0.5*z^T*P*z + q^T*z + c where z is a vector in the current set
         */
        Eigen::Vector<zono_float, -1> optimize_over(
            const Eigen::SparseMatrix<zono_float> &P, const Eigen::Vector<zono_float, -1> &q, zono_float c=0,
            const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr) const
        {
            return do_optimize_over(P, q, c, settings, solution);
        }

        /**
         * @brief Returns the projection of the point x onto the set object.
         * 
         * @param x point to be projected
         * @param settings optimization settings structure
         * @param solution optimization solution structure pointer, populated with result
         * @return point z in the current set
         */
        Eigen::Vector<zono_float, -1> project_point(const Eigen::Vector<zono_float, -1>& x,
            const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr) const
        {
            return do_project_point(x, settings, solution);
        }

        /**
         * @brief Returns true if the set is provably empty, false otherwise.
         * 
         * @param settings optimization settings structure
         * @param solution optimization solution structure pointer, populated with result
         * @return flag indicating whether set is provably empty
         *
         */
        bool is_empty(const OptSettings &settings=OptSettings(),
            OptSolution* solution=nullptr) const
        {
            return do_is_empty(settings, solution);
        }

        /**
         * @brief Computes support function of the set in the direction d.
         * 
         * @param d vector defining direction for support function
         * @param settings optimization settings structure
         * @param solution optimization solution structure pointer, populated with result
         * @return support
         * 
         * Solves max_{z in Z} <z, d> where <., .> is the inner product 
         */
        zono_float support(const Eigen::Vector<zono_float, -1>& d, const OptSettings &settings=OptSettings(),
            OptSolution* solution=nullptr)
        {
            return do_support(d, settings, solution);
        }

        /**
         * @brief Checks whether the point x is contained in the set object.
         * 
         * @param x point to be checked for set containment
         * @param settings optimization settings structure
         * @param solution optimization solution structure pointer, populated with result
         * 
         * False positives are possible; will return true if the optimization converges within the specified tolerances.
         * Will return false only if an infeasibility certificate is found, i.e., false negatives are not possible.
         */
        bool contains_point(const Eigen::Vector<zono_float, -1>& x, const OptSettings &settings=OptSettings(),
            OptSolution* solution=nullptr) const
        {
            return do_contains_point(x, settings, solution);
        }

        /**
         * @brief Computes a bounding box of the set object as a Box object.
         * 
         * @param settings optimization settings structure
         * @param solution optimization solution structure pointer, populated with result
         * @return Box Z_bb
         * 
         * In general, solves 2*n support optimizations where n is the set dimension to compute a bounding box.
         */
        Box bounding_box(const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr)
        {
            return do_bounding_box(settings, solution);
        }

        /**
         * @brief Computes individual constrained zonotopes whose union is the hybrid zonotope object.
         * 
         * @param remove_redundancy flag to make call to remove_redundancy for each identified leaf
         * @param settings optimization settings structure
         * @param solution optimization solution structure pointer, populated with result
         * @param n_leaves max number of leaves to find
         * @param contractor_iter number of interval contractor iterations to run if using remove_redundancy
         * @return vector of constrained zonotopes [Z0, Z1, ...] such that Zi is a subset of the current set for all i
         *
         * Searches for constrained zonotopes that correspond to feasible combinations of the hybrid zonotope binary variables.
         * If the branch and bound converges (i.e., did not hit max time, max number of branch and bound iterations, or max nodes in queue)
         * and the n_leaves argument does not stop the optimization before exhausting all possibilities, then the resulting vector of constrained zonotopes
         * can be unioned to recover the original set. It is possible for a leaf to be the empty set if the optimization converges before detecting an infeasibility certificate.
         */
        std::vector<ConZono> get_leaves(bool remove_redundancy=true, const OptSettings &settings=OptSettings(),
            OptSolution* solution=nullptr, int n_leaves = std::numeric_limits<int>::max(), int contractor_iter=100) const;

        // friend function declarations
        friend std::unique_ptr<HybZono> affine_map(const HybZono& Z,
            const Eigen::SparseMatrix<zono_float>& R, const Eigen::Vector<zono_float, -1>& s);
        friend std::unique_ptr<HybZono> project_onto_dims(const HybZono& Z, const std::vector<int>& dims);
        friend std::unique_ptr<HybZono> minkowski_sum(const HybZono& Z1, HybZono& Z2);
        friend std::unique_ptr<HybZono> pontry_diff(HybZono& Z1, HybZono& Z2, bool exact);
        friend std::unique_ptr<HybZono> intersection(const HybZono& Z1, HybZono& Z2, 
            const Eigen::SparseMatrix<zono_float>& R);
        friend std::unique_ptr<HybZono> intersection_over_dims(const HybZono& Z1, HybZono& Z2, 
            const std::vector<int>& dims);
        friend std::unique_ptr<HybZono> halfspace_intersection(HybZono& Z, const Eigen::SparseMatrix<zono_float>& H, 
            const Eigen::Vector<zono_float, -1>& f, const Eigen::SparseMatrix<zono_float>& R);
        friend std::unique_ptr<HybZono> union_of_many(const std::vector<HybZono*>& Zs, bool preserve_sharpness, bool expose_indicators);
        friend std::unique_ptr<HybZono> cartesian_product(const HybZono& Z1, HybZono& Z2);
        friend std::unique_ptr<HybZono> constrain(HybZono& Z, const std::vector<Inequality> &ineqs, const Eigen::SparseMatrix<zono_float>& R);
        friend std::unique_ptr<HybZono> set_diff(const HybZono& Z1, HybZono& Z2, zono_float delta_m, bool remove_redundancy,
            const OptSettings &settings, OptSolution* solution, int n_leaves, int contractor_iter);
        friend std::unique_ptr<HybZono> vrep_2_hybzono(const std::vector<Eigen::Matrix<zono_float, -1, -1>> &Vpolys, bool expose_indicators);
        friend std::unique_ptr<HybZono> zono_union_2_hybzono(std::vector<Zono> &Zs, bool expose_indicators);

    protected:

        // fields

        /// generator matrix G = [Gc, Gb]
        Eigen::SparseMatrix<zono_float> G = Eigen::SparseMatrix<zono_float>(0, 0);

        /// continuous generator matrix
        Eigen::SparseMatrix<zono_float> Gc = Eigen::SparseMatrix<zono_float>(0, 0);

        /// binary generator matrix
        Eigen::SparseMatrix<zono_float> Gb = Eigen::SparseMatrix<zono_float>(0, 0);

        /// constraint matrix A = [Ac, Ab]
        Eigen::SparseMatrix<zono_float> A = Eigen::SparseMatrix<zono_float>(0, 0);

        /// continuous constraint matrix
        Eigen::SparseMatrix<zono_float> Ac = Eigen::SparseMatrix<zono_float>(0, 0);

        /// binary constraint matrix
        Eigen::SparseMatrix<zono_float> Ab = Eigen::SparseMatrix<zono_float>(0, 0);

        /// center vector
        Eigen::Vector<zono_float, -1> c = Eigen::Vector<zono_float, -1>(0);

        /// constraint vector
        Eigen::Vector<zono_float, -1> b = Eigen::Vector<zono_float, -1>(0);

        /// set dimension
        int n = 0;

        /// total number of factors. nG = nGc + nGb
        int nG = 0;

        /// number of continuous factors
        int nGc = 0;

        /// number of binary factors
        int nGb = 0;

        /// number of constraints
        int nC = 0;

        /// flag to indicate whether the set is in 0-1 or -1-1 form
        bool zero_one_form = false;

        /// flag to indicate whether the set is known to be sharp (i.e., convex relaxation = convex hull)
        bool sharp = false;

        // methods
        virtual Eigen::Vector<zono_float, -1> do_optimize_over(
            const Eigen::SparseMatrix<zono_float> &P, const Eigen::Vector<zono_float, -1> &q, zono_float c,
            const OptSettings &settings, OptSolution* solution) const;

        virtual Eigen::Vector<zono_float, -1> do_project_point(const Eigen::Vector<zono_float, -1>& x,
            const OptSettings &settings, OptSolution* solution) const;

        virtual bool do_is_empty(const OptSettings &settings, OptSolution* solution) const;

        virtual zono_float do_support(const Eigen::Vector<zono_float, -1>& d, const OptSettings &settings,
            OptSolution* solution);

        virtual bool do_contains_point(const Eigen::Vector<zono_float, -1>& x, const OptSettings &settings,
            OptSolution* solution) const;

        virtual Box do_bounding_box(const OptSettings &settings, OptSolution* solution);

        virtual std::unique_ptr<HybZono> do_complement(zono_float, bool remove_redundancy, const OptSettings &settings,
            OptSolution* solution, int n_leaves, int contractor_iter);


        static void remove_generators(Eigen::SparseMatrix<zono_float>& G, Eigen::SparseMatrix<zono_float>& A, const std::set<int>& idx_to_remove);
        static std::set<int> find_unused_generators(const Eigen::SparseMatrix<zono_float>& G, const Eigen::SparseMatrix<zono_float>& A);
        OptSolution mi_opt(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
            zono_float c, const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b,
            const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr) const;
        std::vector<OptSolution> mi_opt_multisol(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
            zono_float c, const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b, int n_sols,
            const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr) const;

    private:

        void make_G_A();
        void set_Ac_Ab_from_A();
        std::vector<Eigen::Vector<zono_float, -1>> get_bin_leaves(const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr,
            int n_leaves = std::numeric_limits<int>::max()) const;
};

// forward delcarations
 /**
 * @brief Returns affine map R*Z + s of set Z
 *
 * @param Z zonotopic set
 * @param R affine map matrix
 * @param s vector offset
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> affine_map(const HybZono& Z,
    const Eigen::SparseMatrix<zono_float>& R, const Eigen::Vector<zono_float, -1>& s = Eigen::Vector<zono_float, -1>());

/**
 * @brief Projects set Z onto the dimensions specified in dims.
 *
 * @param Z zonotopic set
 * @param dims vector of dimensions
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> project_onto_dims(const HybZono& Z, const std::vector<int>& dims);

/**
 * @brief Computes Minkowski sum of two sets Z1 and Z2.
 *
 * @param Z1 zonotopic set
 * @param Z2 zonotopic set
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> minkowski_sum(const HybZono& Z1, HybZono& Z2);

/**
 * @brief Computes the Pontryagin difference Z1 - Z2
 *
 * @param Z1 minuend
 * @param Z2 subtrahend
 * @param exact require output to be exact, otherwise inner approximation will be returned
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 *
 * For inner approximations (exact = false), the algorithm from Vinod et. al. 2025 is used.
 * Note that this algorithm is exact when the minuend is a constrained zonotope and the matrix [G;A] is invertible.
 * Exact Pontryagin difference can only be computed when the subtrahend is a zonotope.
 * If subtrahend is a constrained zonotope, it will first be over-approximated as a zonotope.
 * If subtrahend is a hybrid zonotope, a get_leaves operation will first be performed to produce
 * a union of constrained zonotopes.
 * If the minuend is a hybrid zonotope and exact is false, a get_leaves operation will be performed for 
 * Z1 to reduce the number of leaves in the resulting set.
 */
std::unique_ptr<HybZono> pontry_diff(HybZono& Z1, HybZono& Z2, bool exact=false);

/**
 * @brief Computes the generalized intersection of sets Z1 and Z2 over the matrix R.
 *
 * @param Z1 zonotopic set
 * @param Z2 zonotopic set
 * @param R affine map matrix
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> intersection(const HybZono& Z1, HybZono& Z2,
    const Eigen::SparseMatrix<zono_float>& R=Eigen::SparseMatrix<zono_float>());

/**
 * @brief Computes the generalized intersection of sets Z1 and Z2 over the specified dimensions.
 *
 * @param Z1 zonotopic set
 * @param Z2 zonotopic set
 * @param dims vector of dimensions
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> intersection_over_dims(const HybZono& Z1, HybZono& Z2,
    const std::vector<int>& dims);

/**
 * @brief Computes the intersection generalized intersection of set Z with halfspace H*x <= f over matrix R.
 *
 * @param Z zonotopic set
 * @param H halfspace matrix
 * @param f halfspace vector
 * @param R affine map matrix
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> halfspace_intersection(HybZono& Z, const Eigen::SparseMatrix<zono_float>& H,
    const Eigen::Vector<zono_float, -1>& f, const Eigen::SparseMatrix<zono_float>& R=Eigen::SparseMatrix<zono_float>());

/**
 * @brief Computes union of several sets
 *
 * @param Zs Sets to be unioned.
 * @param preserve_sharpness Flag to preserve sharpness of the union at expense of complexity.
 * @param expose_indicators Flag to append indicator set to the union.
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 *
 * Computes union of sets {Z0, Z1, ..., Zn}. If expose_indicators is true, returns union({Z0, ..., Zn}) x I where I is the indicator set for the union.
 * Specifically, each dimension of I corresponds to one of the Zi in the union. So for union_of_many({Z0, Z1, Z2}, true) with Z0, Z1, Z2 not intersecting,
 * if a vector [z, i] is in union({Z0, Z1, Z2}) x I, then i = [1, 0, 0] if z is in Z0, etc.
 */
std::unique_ptr<HybZono> union_of_many(const std::vector<HybZono*>& Zs, bool preserve_sharpness=false, bool expose_indicators=false);

/**
 * @brief Computes the Cartesian product of two sets Z1 and Z2.
 *
 * @param Z1 zonotopic set
 * @param Z2 zonotopic set
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> cartesian_product(const HybZono& Z1, HybZono& Z2);

/**
 * @brief Applies inequalities to set.
 *
 * @param Z Set for inequalities to be applied to.
 * @param ineqs Vector of inequalities.
 * @param R For generalized intersection with the inequalities. Defaults to identity.
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 *
 * Constrains set Z by applying the given inequalities to the set.
 * For example, given a set Z with states z0, z1, z2, the constraint z0 + z1 - z2 <= 2 could be added via an inequality object.
 * R is used for generalized intersection-like operations. For instance, when all the inequalities are <= inequalities,
 * this function returns Z int_R (Hx<=f) where H is the halfspace represented by the inequalities.
 */
std::unique_ptr<HybZono> constrain(HybZono& Z, const std::vector<Inequality> &ineqs, const Eigen::SparseMatrix<zono_float>& R=Eigen::SparseMatrix<zono_float>());

/**
 * @brief Set difference Z1 \ Z2
 *
 * @param Z1 zonotopic set
 * @param Z2 zonotopic set
 * @param delta_m parameter defining range of complement
 * @param remove_redundancy remove redundant constraints and unused generators in get_leaves function call
 * @param settings optimization settings for get_leaves function call
 * @param solution optimization solution for get_leaves function call
 * @param n_leaves maximum number of leaves to return in get_leaves function call
 * @param contractor_iter number of interval contractor iterations to run if using remove_redundancy
 * @return zonotopic set
 * @ingroup ZonoOpt_SetOperations
 */
std::unique_ptr<HybZono> set_diff(const HybZono& Z1, HybZono& Z2, zono_float delta_m = 100, bool remove_redundancy=true,
    const OptSettings &settings=OptSettings(), OptSolution* solution=nullptr, int n_leaves = std::numeric_limits<int>::max(), int contractor_iter = 100);


/**
 * @brief Computes a hybrid zonotope from a union of vertex representation polytopes.
 *
 * @param Vpolys V-rep polytopes to be unioned.
 * @param expose_indicators Flag to append indicator set to the union.
 * @return zonotopic set
 * @ingroup ZonoOpt_SetupFunctions
 *
 * Vpolys is a vector of matrices, where each matrix represents a polytope in vertex representation.
 * Each row in each polytope matrix is a vertex of the polytope, and each column corresponds to a dimension.
 * The function constructs a hybrid zonotope in [0,1] form that represents the union of these polytopes.
 * This function computes union of sets {V0, V1, ..., Vn}. If expose_indicators is true, returns union({V0, ..., Vn}) x I where I is the indicator set for the union.
 * Specifically, each dimension of I corresponds to one of the Vi in the union. So for vrep_2_hybzono({V0, V1, V2}, true) with V0, V1, V2 not intersecting,
 * if a vector [z, i] is in union({V0, V1, V2}) x I, then i = [1, 0, 0] if z is in V0, etc.
 */
std::unique_ptr<HybZono> vrep_2_hybzono(const std::vector<Eigen::Matrix<zono_float, -1, -1>> &Vpolys, bool expose_indicators=false);


/**
* @brief Computes a hybrid zonotope from a union of zonotopes.
*
* @param Zs A vector of zonotopes to be unioned.
* @param expose_indicators Flag to append indicator set to the union.
* @return zonotopic set
* @ingroup ZonoOpt_SetupFunctions
*
* This function computes union of sets {Z0, Z1, ..., Zn}. This can be more efficient than union_of_many if all sets are zonotopes because generators can be reused.
* If expose_indicators is true, returns union({Z0, ..., Zn}) x I where I is the indicator set for the union.
* Specifically, each dimension of I corresponds to one of the Zi in the union. So for zono_union_2_hybzono({Z0, Z1, Z2}, true) with Z0, Z1, VZ2 not intersecting,
* if a vector [z, i] is in union({Z0, Z1, Z2}) x I, then i = [1, 0, 0] if z is in Z0, etc.
*/
std::unique_ptr<HybZono> zono_union_2_hybzono(std::vector<Zono> &Zs, bool expose_indicators=false);


// implementation
inline void HybZono::set(const Eigen::SparseMatrix<zono_float>& Gc, const Eigen::SparseMatrix<zono_float>& Gb, const Eigen::Vector<zono_float, -1>& c,
    const Eigen::SparseMatrix<zono_float>& Ac, const Eigen::SparseMatrix<zono_float>& Ab, const Eigen::Vector<zono_float, -1>& b,
    const bool zero_one_form, const bool sharp)
{
    // check dimensions
    if (Gc.rows() != c.size() || Gb.rows() != c.size() || Gc.cols() != Ac.cols() 
    || Gb.cols() != Ab.cols() || Ac.rows() != b.size() || Ab.rows() != b.size())
    {
        throw std::invalid_argument("HybZono: inconsistent dimensions.");
    }

    this->Gc = Gc;
    this->Gb = Gb;
    this->Ac = Ac;
    this->Ab = Ab;
    this->c = c;
    this->b = b;
    this->nGc = static_cast<int>(Gc.cols());
    this->nGb = static_cast<int>(Gb.cols());
    this->nC = static_cast<int>(Ac.rows());
    this->n = static_cast<int>(Gc.rows());
    this->zero_one_form = zero_one_form;

    make_G_A();

    this->sharp = sharp;
}

inline void HybZono::convert_form()
{
    Eigen::Vector<zono_float, -1> c, b;
    Eigen::SparseMatrix<zono_float> Gb, Ab, Ac, Gc;

    if (!this->zero_one_form) // convert to [0,1] generators
    {
        c = this->c - this->G*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        b = this->b + this->A*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        Gb = 2.0*this->Gb;
        Ab = 2.0*this->Ab;
        Gc = 2.0*this->Gc;
        Ac = 2.0*this->Ac;

        set(Gc, Gb, c, Ac, Ab, b, true);
    }
    else // convert to [-1,1] generators
    {
        c = this->c + 0.5*this->G*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        b = this->b - 0.5*this->A*Eigen::Vector<zono_float, -1>::Ones(this->nG);
        Gb = 0.5*this->Gb;
        Ab = 0.5*this->Ab;
        Gc = 0.5*this->Gc;
        Ac = 0.5*this->Ac;

        set(Gc, Gb, c, Ac, Ab, b, false);
    }
}

inline void HybZono::remove_redundancy(const int contractor_iter)
{
    // declare vars
    std::set<int> idx_c_to_remove, idx_b_to_remove;

    // lambda to remove generators
    auto remove_all_generators = [this](const std::set<int>& idx_c, const std::set<int>& idx_b) -> void
    {
        // remove generators
        if (!idx_c.empty()) {
            remove_generators(this->Gc, this->Ac, idx_c);
        }
        if (!idx_b.empty()) {
            remove_generators(this->Gb, this->Ab, idx_b);
        }

        // update number of generators (needs to happen before call to make_G_A())
        this->nG = static_cast<int>(this->G.cols());
        this->nGc = static_cast<int>(this->Gc.cols());
        this->nGb = static_cast<int>(this->Gb.cols());

        // update equivalent matrices
        make_G_A();
    };

    // apply interval contractor
    Eigen::Vector<zono_float, -1> x_l (this->nG);
    Eigen::Vector<zono_float, -1> x_u (this->nG);
    if (this->zero_one_form)
    {
        x_l.setZero();
    }
    else
    {
        x_l.setConstant(-1);
    }
    x_u.setOnes();
    Box box (x_l, x_u);
    box.contract(this->A, this->b, contractor_iter);

    // find any variables whose values are fixed
    std::vector<std::pair<int, zono_float>> fixed_vars;
    for (int i=0; i<this->nG; ++i)
    {
        if (box[i].is_single_valued())
        {
            fixed_vars.emplace_back(i, box[i].get_y_max());
            if (i < this->nGc)
            {
                idx_c_to_remove.insert(i);
            }
            else
            {
                idx_b_to_remove.insert(i-this->nGc);
            }
        }
    }

    // get updates to c and b
    Eigen::Vector<zono_float, -1> dc (this->n);
    Eigen::Vector<zono_float, -1> db (this->nC);
    Eigen::Vector<zono_float, -1> dc_k (this->n);
    Eigen::Vector<zono_float, -1> db_k (this->nC);
    dc.setZero();
    db.setZero();
    for (auto& [k, val] : fixed_vars)
    {
        dc_k.setZero();
        for (Eigen::SparseMatrix<zono_float>::InnerIterator it(this->G, k); it; ++it)
        {
            dc_k(it.row()) = it.value()*val;
        }
        dc += dc_k;

        db_k.setZero();
        for (Eigen::SparseMatrix<zono_float>::InnerIterator it(this->A, k); it; ++it)
        {
            db_k(it.row()) = it.value()*val;
        }
        db -= db_k;
    }

    // set center and constraint vector
    this->c += dc;
    this->b += db;

    // remove generators
    remove_all_generators(idx_c_to_remove, idx_b_to_remove);

    // remove redundant constraints
    remove_redundant_constraints<zono_float>(this->A, this->b);
    this->nC = static_cast<int>(this->A.rows());

    // update Ac, Ab
    set_Ac_Ab_from_A();

    // identify any unused generators
    idx_c_to_remove = find_unused_generators(this->Gc, this->Ac);
    idx_b_to_remove = find_unused_generators(this->Gb, this->Ab);

    // remove
    remove_all_generators(idx_c_to_remove, idx_b_to_remove);
}

inline std::string HybZono::print() const
{
    std::stringstream ss;
    ss << "HybZono: " << std::endl;
    ss << "n: " << this->n << std::endl;
    ss << "nGc: " << this->nGc << std::endl;
    ss << "nGb: " << this->nGb << std::endl;
    ss << "nC: " << this->nC << std::endl;
    ss << "Gc: " << Eigen::Matrix<zono_float, -1, -1>(this->Gc) << std::endl;
    ss << "Gb: " << Eigen::Matrix<zono_float, -1, -1>(this->Gb) << std::endl;
    ss << "c: " << this->c << std::endl;
    ss << "Ac: " << Eigen::Matrix<zono_float, -1, -1>(this->Ac) << std::endl;
    ss << "Ab: " << Eigen::Matrix<zono_float, -1, -1>(this->Ab) << std::endl;
    ss << "b: " << this->b << std::endl;
    ss << "zero_one_form: " << this->zero_one_form << std::endl;
    ss << "sharp: " << this->sharp;
    return ss.str();
}

inline std::ostream& operator<<(std::ostream& os, const HybZono& Z)
{
    os << Z.print();
    return os;
}

inline Eigen::Vector<zono_float, -1> HybZono::do_optimize_over(
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

    // solve MIQP
    OptSolution sol = this->mi_opt(P_fact, q_fact, c+delta_c, this->A, this->b, settings, solution);
    if (sol.infeasible)
        return Eigen::Vector<zono_float, -1>(this->nG);
    else
        return this->G*sol.z + this->c;
}

inline Eigen::Vector<zono_float, -1> HybZono::do_project_point(const Eigen::Vector<zono_float, -1>& x,
    const OptSettings &settings, OptSolution* solution) const
{
    // check dimensions
    if (this->n != x.size())
    {
        throw std::invalid_argument("Point projection: inconsistent dimensions.");
    }

    // cost
    Eigen::SparseMatrix<zono_float> P = this->G.transpose()*this->G;
    Eigen::Vector<zono_float, -1> q = this->G.transpose()*(this->c-x);
    
    // solve MIQP
    const OptSolution sol = this->mi_opt(P, q, 0, this->A, this->b, settings, solution);
    if (sol.infeasible)
        throw std::runtime_error("Point projection: infeasible");

    return this->G*sol.z + this->c;
}

inline bool HybZono::do_is_empty(const OptSettings &settings, OptSolution* solution) const
{
    // trivial case
    if (this->n == 0)
        return true;

    // optimize over P=I, q=0
    Eigen::SparseMatrix<zono_float> P (this->nG, this->nG);
    P.setIdentity();
    Eigen::Vector<zono_float, -1> q = Eigen::Vector<zono_float, -1>::Zero(this->nG);

    // solve
    const std::vector<OptSolution> sol_vec = this->mi_opt_multisol(P, q, 0, this->A, this->b, 1, settings, solution);
    if (sol_vec.size() > 0)
        return sol_vec[0].infeasible;
    else
        return true;
}

inline bool HybZono::do_contains_point(const Eigen::Vector<zono_float, -1>& x, const OptSettings &settings,
    OptSolution* solution) const
{
    // check dimensions
    if (this->n != x.size())
    {
        throw std::invalid_argument("Contains point: inconsistent dimensions.");
    }

    // cost and constraints
    Eigen::SparseMatrix<zono_float> P (this->nG, this->nG);
    P.setIdentity();
    Eigen::Vector<zono_float, -1> q (this->nG);
    q.setZero(); // zeros
    Eigen::SparseMatrix<zono_float> A = vcat<zono_float>(this->A, this->G);
    Eigen::Vector<zono_float, -1> b (this->nC + this->n);
    b.segment(0, this->nC) = this->b;
    b.segment(this->nC, this->n) = x-this->c;

    // solve MIQP
    const OptSolution sol = this->mi_opt(P, q, 0, A, b, settings, solution);
    return !(sol.infeasible);
}


inline OptSolution HybZono::mi_opt(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
    const zono_float c, const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b,
    const OptSettings &settings, OptSolution* solution) const
{
    // QP data
    Eigen::Vector<zono_float, -1> xi_lb (this->nG);
    if (this->zero_one_form)
        xi_lb.setZero();
    else
        xi_lb.setConstant(-1);
    const Eigen::Vector<zono_float, -1> xi_ub = Eigen::Vector<zono_float, -1>::Ones(this->nG);

    const auto admm_data = std::make_shared<ADMM_data>(P, q, A, b, xi_lb, xi_ub, c, settings);

    // mixed integer data
    MI_data mi_data;
    mi_data.admm_data = admm_data;
    mi_data.idx_b = std::make_pair(this->nGc, this->nGb);
    mi_data.zero_one_form = this->zero_one_form;

    // build MI_ADMM_solver object
    MI_solver mi_solver(mi_data);

    // solve optimization problem
    OptSolution sol = mi_solver.solve();

    if (solution != nullptr)
        *solution = sol;
    return sol;
}

inline std::vector<OptSolution> HybZono::mi_opt_multisol(const Eigen::SparseMatrix<zono_float>& P, const Eigen::Vector<zono_float, -1>& q,
    const zono_float c, const Eigen::SparseMatrix<zono_float>& A, const Eigen::Vector<zono_float, -1>& b, int n_sols,
    const OptSettings &settings, OptSolution* solution) const
{
    // ADMM data
    Eigen::Vector<zono_float, -1> xi_lb (this->nG);
    if (this->zero_one_form)
        xi_lb.setZero();
    else
        xi_lb.setConstant(-1);
    const Eigen::Vector<zono_float, -1> xi_ub = Eigen::Vector<zono_float, -1>::Ones(this->nG);

    const auto admm_data = std::make_shared<ADMM_data>(P, q, A, b, xi_lb, xi_ub, c, settings);

    // mixed integer data
    MI_data mi_data;
    mi_data.admm_data = admm_data;
    mi_data.idx_b = std::make_pair(this->nGc, this->nGb);
    mi_data.zero_one_form = this->zero_one_form;

    // build MI_ADMM_solver object
    MI_solver mi_solver(mi_data);

    // solve optimization problem
    auto [fst, snd] = mi_solver.multi_solve(n_sols);
    if (solution != nullptr)
        *solution = snd;
    return fst;
}


inline void HybZono::remove_generators(Eigen::SparseMatrix<zono_float>& G, Eigen::SparseMatrix<zono_float>& A, const std::set<int>& idx_to_remove)
{
    // declare triplets
    std::vector<Eigen::Triplet<zono_float>> triplets;

    // update G
    int delta_ind = 0;
    for (int k=0; k<G.outerSize(); k++)
    {
        if (idx_to_remove.count(k))
        {
            ++delta_ind;
        }
        else
        {
            for (Eigen::SparseMatrix<zono_float>::InnerIterator it(G, k); it; ++it)
            {
                triplets.emplace_back(static_cast<int>(it.row()), k-delta_ind, it.value());
            }
        }
    }
    G.resize(G.rows(), G.cols() - delta_ind);
    G.setFromTriplets(triplets.begin(), triplets.end());

    // update A
    triplets.clear();
    delta_ind = 0;
    for (int k=0; k<A.outerSize(); k++)
    {
        if (idx_to_remove.count(k))
        {
            ++delta_ind;
        }
        else
        {
            for (Eigen::SparseMatrix<zono_float>::InnerIterator it(A, k); it; ++it)
            {
                triplets.emplace_back(static_cast<int>(it.row()), k-delta_ind, it.value());
            }
        }
    }
    A.resize(A.rows(), A.cols() - delta_ind);
    A.setFromTriplets(triplets.begin(), triplets.end());
}

inline std::set<int> HybZono::find_unused_generators(const Eigen::SparseMatrix<zono_float>& G, const Eigen::SparseMatrix<zono_float>& A)
{
    std::set<int> idx_no_cons;
    for (int k=0; k<A.outerSize(); k++)
    {
        bool is_unused = true;
        for (Eigen::SparseMatrix<zono_float>::InnerIterator it(A, k); it; ++it)
        {
            if (std::abs(it.value()) > zono_eps)
            {
                is_unused = false;
                break;
            }
        }

        if (is_unused)
        {
            idx_no_cons.insert(k);
        }
    }

    // check if any of idx_no_cons multiply only zeros
    std::set<int> idx_to_remove;
    for (int idx_no_con : idx_no_cons)
    {
        bool is_zero = true;
        for (Eigen::SparseMatrix<zono_float>::InnerIterator it(G, idx_no_con); it; ++it)
        {
            if (std::abs(it.value()) > zono_eps)
            {
                is_zero = false;
                break;
            }
        }

        if (is_zero)
        {
            idx_to_remove.insert(idx_no_con);
        }
    }

    return idx_to_remove;
}

inline void HybZono::make_G_A()
{
    std::vector<Eigen::Triplet<zono_float>> tripvec;
    get_triplets_offset<zono_float>(this->Gc, tripvec, 0, 0);
    get_triplets_offset<zono_float>(this->Gb, tripvec, 0, this->nGc);
    this->G.resize(this->n, this->nGc + this->nGb);
    this->G.setFromTriplets(tripvec.begin(), tripvec.end());

    tripvec.clear();
    get_triplets_offset<zono_float>(this->Ac, tripvec, 0, 0);
    get_triplets_offset<zono_float>(this->Ab, tripvec, 0, this->nGc);
    this->A.resize(this->nC, this->nGc + this->nGb);
    this->A.setFromTriplets(tripvec.begin(), tripvec.end());

    this->nG = this->nGc + this->nGb;
}

inline void HybZono::set_Ac_Ab_from_A()
{
    std::vector<Eigen::Triplet<zono_float>> triplets_Ac, triplets_Ab;

    // iterate over A
    for (int k=0; k<this->A.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<zono_float>::InnerIterator it(this->A, k); it; ++it)
        {
            if (it.col() < this->nGc)
            {
                triplets_Ac.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
            }
            else
            {
                triplets_Ab.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col())-this->nGc, it.value());
            }
        }
    }

    // set Ac, Ab
    this->Ac.resize(this->nC, this->nGc);
    this->Ac.setFromTriplets(triplets_Ac.begin(), triplets_Ac.end());
    this->Ab.resize(this->nC, this->nGb);
    this->Ab.setFromTriplets(triplets_Ab.begin(), triplets_Ab.end());
}

inline std::vector<Eigen::Vector<zono_float, -1>> HybZono::get_bin_leaves(const OptSettings &settings, OptSolution* solution, const int n_leaves) const
{
    // optimize over P=I, q=0
    Eigen::SparseMatrix<zono_float> P (this->nG, this->nG);
    P.setIdentity();
    const Eigen::Vector<zono_float, -1> q = Eigen::Vector<zono_float, -1>::Zero(this->nG);

    // solve
    std::vector<OptSolution> sol_vec = this->mi_opt_multisol(P, q, 0, this->A, this->b, n_leaves, settings, solution);
    
    // get leaves as conzonos
    std::vector<Eigen::Vector<zono_float, -1>> bin_leaves;
    for (auto &sol : sol_vec)
    {
        bin_leaves.emplace_back(sol.z.segment(this->nGc, this->nGb));
    }

    return bin_leaves;
}

} // end namespace ZonoOpt

#endif