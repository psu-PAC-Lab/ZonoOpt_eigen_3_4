#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#define IS_PYTHON_ENV
#include "ZonoOpt.hpp"
using namespace ZonoOpt;

#include <sstream>

PYBIND11_MODULE(_core, handle)
{
    std::stringstream docstring;
    docstring << "Python bindings for the ZonoOpt C++ library" << std::endl
              << "This library provides classes for zonotopes, constrained zonotopes, and hybrid zonotopes." << std::endl
              << "All classes and methods are implemented using sparse linear algebra via the Eigen library." << std::endl
              << "Generators may optionally have range [0,1] instead of [-1,1]." << std::endl
              << "numpy arrays bind to dense Eigen matrices and vectors." << std::endl
              << "scipy csc_matrices bind to sparse Eigen matrices." << std::endl;
           
    handle.doc() = docstring.str();

    // ADMM classes and structs
    py::class_<ADMM_settings<double>>(handle, "ADMM_settings", "settings for ADMM solver")
        .def(py::init())
        .def_readwrite("rho", &ADMM_settings<double>::rho, "ADMM penalty parameter")
        .def_readwrite("t_max", &ADMM_settings<double>::t_max, "max solution time")
        .def_readwrite("k_max", &ADMM_settings<double>::k_max, "max iterations")
        .def_readwrite("eps_dual", &ADMM_settings<double>::eps_dual, "dual residual convergence tolerance")
        .def_readwrite("eps_prim", &ADMM_settings<double>::eps_prim, "primal residual convergence tolerance")
        .def_readwrite("k_inf_check", &ADMM_settings<double>::k_inf_check, "check convergence every k_inf_check iterations")
        .def_readwrite("inf_norm_conv", &ADMM_settings<double>::inf_norm_conv, 
            "use infinity norm for convergence check, else use 2-norm scaled by square root of number of states")
        .def_readwrite("verbose", &ADMM_settings<double>::verbose, "CAUTION: setting this true can significantly increase solution times when called from Python")
        .def_readwrite("verbosity_interval", &ADMM_settings<double>::verbosity_interval, "print status every verbosity_interval iterations")
    ;

    py::class_<ADMM_solution<double>>(handle, "ADMM_solution", "solution struct for ADMM solver")
        .def(py::init())
        .def_readwrite("x", &ADMM_solution<double>::x, "primal solution")
        .def_readwrite("z", &ADMM_solution<double>::z, "primal solutions (x ~ z at optimum)")
        .def_readwrite("u", &ADMM_solution<double>::u, "dual solution")
        .def_readwrite("J", &ADMM_solution<double>::J, "objective value")
        .def_readwrite("primal_residual", &ADMM_solution<double>::primal_residual, "primal residual")
        .def_readwrite("dual_residual", &ADMM_solution<double>::dual_residual, "dual residual")
        .def_readwrite("run_time", &ADMM_solution<double>::run_time, "total solution time")
        .def_readwrite("startup_time", &ADMM_solution<double>::startup_time, "run time prior to first iteration (matrix factorizations)")
        .def_readwrite("k", &ADMM_solution<double>::k, "number of iterations")
        .def_readwrite("converged", &ADMM_solution<double>::converged, "convergence flag")
        .def_readwrite("infeasible", &ADMM_solution<double>::infeasible, "infeasibility certificate found")
    ;

    py::class_<ADMM_solver<double>>(handle, "ADMM_solver", "solves problems of form min_x 0.5*x^T*P*x + q^T*x + c s.t. Ax=b, x_l <= x <= x_u")
        .def(py::init())
        .def("setup", &ADMM_solver<double>::setup, py::arg("P"), py::arg("q"), py::arg("A"), py::arg("b"), 
            py::arg("x_l"), py::arg("x_u"), py::arg("settings")=ADMM_settings<double>(), py::arg("c")=0.0,
            "setup ADMM solver")
        .def("update_P", &ADMM_solver<double>::update_P, py::arg("P"))
        .def("update_q", &ADMM_solver<double>::update_q, py::arg("q"))
        .def("update_A", &ADMM_solver<double>::update_A, py::arg("A"))
        .def("update_b", &ADMM_solver<double>::update_b, py::arg("b"))
        .def("update_bounds", &ADMM_solver<double>::update_bounds, py::arg("x_l"), py::arg("x_u"))
        .def("update_c", &ADMM_solver<double>::update_c, py::arg("c"))
        .def("update_settings", &ADMM_solver<double>::update_settings, py::arg("settings"))
        .def("warmstart", &ADMM_solver<double>::warmstart, py::arg("x"), py::arg("u"), "warm start primal and dual variables")
        .def("factorize", &ADMM_solver<double>::factorize, "optional pre-factorization of matrices")
        .def("solve", &ADMM_solver<double>::solve, "solve optimization problem")
    ;

    // abstract zono class
    py::class_<AbstractZono<double>>(handle, "AbstractZono", 
        "Base class for points, zonotopes, constrained zonotopes, and hybrid zonotopes. Cannot be instantiated. For optimization methods, an ADMM_settings object may be passed as an optional input, and an ADMM_solution object may be passed as an optional output.")
        .def("convert_form", &AbstractZono<double>::convert_form, "converts between forms with generators xi in [-1,1] and [0,1]")
        .def("__repr__", &AbstractZono<double>::print, "print object data to console")
        .def("remove_redundancy", &AbstractZono<double>::remove_redundancy, "removes linearly dependent constraints (assumed consistent) and unused generators")
        .def("optimize_over", &AbstractZono<double>::optimize_over, "optimize over", py::arg("P"), py::arg("q"), py::arg("c")=0,
            py::arg("settings")=ADMM_settings<double>(), py::arg("solution")=nullptr, "solves min_x 0.5*x^T*P*x + q^T*x + c s.t. x in Z using ADMM")
        .def("project_point", &AbstractZono<double>::project_point, py::arg("x"), py::arg("settings")=ADMM_settings<double>(), 
            py::arg("solution")=nullptr, "projects point x onto Z")
        .def("is_empty", &AbstractZono<double>::is_empty, py::arg("settings")=ADMM_settings<double>(), 
            py::arg("solution")=nullptr, "returns true if Z is the empty set")
        .def("support", &AbstractZono<double>::support, py::arg("d"), py::arg("settings")=ADMM_settings<double>(), 
            py::arg("solution")=nullptr, "returns the support function value in direction d")
        .def("contains_point", &AbstractZono<double>::contains_point, py::arg("x"), py::arg("settings")=ADMM_settings<double>(),
            py::arg("solution")=nullptr, "returns true if Z contains point x")
        .def("bounding_box", &AbstractZono<double>::bounding_box, py::arg("settings")=ADMM_settings<double>(),
            py::arg("solution")=nullptr, "returns the bounding box of Z as zonotope or point")
        .def("is_point", &AbstractZono<double>::is_point)
        .def("is_zono", &AbstractZono<double>::is_zono)
        .def("is_conzono", &AbstractZono<double>::is_conzono)
        .def("is_hybzono", &AbstractZono<double>::is_hybzono)
    ;

    // point class
    py::class_<Point<double>, AbstractZono<double> /* parent type */>(handle, "Point", "Point class")
        .def(py::init<const Eigen::Vector<double, -1>&>(), "Point constructor", py::arg("c")=Eigen::Vector<double, -1>())
        .def("set", &Point<double>::set, "set point", py::arg("c"))
        .def("get_n", &Point<double>::get_n)
        .def("get_c", &Point<double>::get_c)
    ;

    // zono class
    py::class_<Zono<double>, AbstractZono<double> /* parent type */>(handle, "Zono", "Zonotope class")
        .def(py::init<const Eigen::SparseMatrix<double>&, const Eigen::Vector<double, -1>&, bool>(), 
            "Zono constructor", py::arg("G")=Eigen::SparseMatrix<double>(), py::arg("c")=Eigen::Vector<double, -1>(), 
            py::arg("zero_one_form")=false)
        .def("set", &Zono<double>::set, "set zonotope", py::arg("G"), py::arg("c"), py::arg("zero_one_form")=false)
        .def("get_n", &Zono<double>::get_n)
        .def("get_nG", &Zono<double>::get_nG)
        .def("get_G", &Zono<double>::get_G)
        .def("get_c", &Zono<double>::get_c)
        .def("is_0_1_form", &Zono<double>::is_0_1_form, "true if xi in [0,1], false if xi in [-1,1]")
    ;

    // conzono class
    py::class_<ConZono<double>, AbstractZono<double> /* parent type */>(handle, "ConZono", "Constrained zonotope class")
        .def(py::init<const Eigen::SparseMatrix<double>&, const Eigen::Vector<double, -1>&,
            const Eigen::SparseMatrix<double>&, const Eigen::Vector<double, -1>&, 
            bool>(), "ConZono constructor", py::arg("G")=Eigen::SparseMatrix<double>(), py::arg("c")=Eigen::Vector<double, -1>(),
            py::arg("A")=Eigen::SparseMatrix<double>(), py::arg("b")=Eigen::Vector<double, -1>(), py::arg("zero_one_form")=false)
        .def("set", &ConZono<double>::set, "set constrained zonotope", py::arg("G"), py::arg("c"), py::arg("A"), py::arg("b"), 
            py::arg("zero_one_form")=false)
        .def("get_G", &ConZono<double>::get_G)
        .def("get_c", &ConZono<double>::get_c)
        .def("get_nG", &ConZono<double>::get_nG)
        .def("get_n", &ConZono<double>::get_n)
        .def("is_0_1_form", &ConZono<double>::is_0_1_form, "true if xi in [0,1], false if xi in [-1,1]")
        .def("get_nC", &ConZono<double>::get_nC)
        .def("get_A", &ConZono<double>::get_A)
        .def("get_b", &ConZono<double>::get_b)
    ;

    // hybzono class
    py::class_<HybZono<double>, AbstractZono<double> /* parent type */>(handle, "HybZono", "Hybrid zonotope class")
        .def(py::init<const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&, const Eigen::Vector<double, -1>&,
            const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&, const Eigen::Vector<double, -1>&,
            bool>(), "HybZono constructor", py::arg("Gc")=Eigen::SparseMatrix<double>(), py::arg("Gb")=Eigen::SparseMatrix<double>(),
            py::arg("c")=Eigen::Vector<double, -1>(), py::arg("Ac")=Eigen::SparseMatrix<double>(), py::arg("Ab")=Eigen::SparseMatrix<double>(),
            py::arg("b")=Eigen::Vector<double, -1>(), py::arg("zero_one_form")=false)
        .def("set", &HybZono<double>::set, "set hybrid zonotope", py::arg("Gc"), py::arg("Gb"), py::arg("c"), py::arg("Ac"), py::arg("Ab"), 
            py::arg("b"), py::arg("zero_one_form")=false)
        .def("get_n", &HybZono<double>::get_n)
        .def("get_nC", &HybZono<double>::get_nC)
        .def("get_nG", &HybZono<double>::get_nG)
        .def("is_0_1_form", &HybZono<double>::is_0_1_form, "true if xi in [0,1], false if xi in [-1,1]")
        .def("get_G", &HybZono<double>::get_G)
        .def("get_c", &HybZono<double>::get_c)
        .def("get_A", &HybZono<double>::get_A)
        .def("get_b", &HybZono<double>::get_b)
        .def("get_nGc", &HybZono<double>::get_nGc)
        .def("get_nGb", &HybZono<double>::get_nGb)
        .def("get_Gc", &HybZono<double>::get_Gc)
        .def("get_Gb", &HybZono<double>::get_Gb)
        .def("get_Ac", &HybZono<double>::get_Ac)
        .def("get_Ab", &HybZono<double>::get_Ab)
        .def("convex_relaxation", &HybZono<double>::convex_relaxation, "returns convex relaxation of hybrid zonotope as constrained zonotope")
    ;

    // set operations
    handle.def("affine_map", &affine_map<double>, py::arg("Z"), py::arg("R"), py::arg("s")=Eigen::Vector<double, -1>(),
        "returns Z_o = R*Z + s");
    handle.def("project_onto_dims", &project_onto_dims<double>, py::arg("Z"), py::arg("dims"),
        "equivalent to affine map with dims vector selecting dimensions");
    handle.def("minkowski_sum", &minkowski_sum<double>, py::arg("Z1"), py::arg("Z2"),
        "returns Z = Z1 + Z2");
    handle.def("pontry_diff", &pontry_diff<double>, py::arg("Z1"), py::arg("Z2"),
        "returns Z = Z1 - Z2");
    handle.def("intersection", &intersection<double>, py::arg("Z1"), py::arg("Z2"), py::arg("R")=Eigen::SparseMatrix<double>(),
        "returns generalized intersection over R of Z1 and Z2");
    handle.def("intersection_over_dims", &intersection_over_dims<double>, py::arg("Z1"), py::arg("Z2"), py::arg("dims"),
        "performs generalized intersection over selected dimensions of Z1");
    handle.def("halfspace_intersection", &halfspace_intersection<double>, py::arg("Z"), py::arg("H"), py::arg("f"), py::arg("R")=Eigen::SparseMatrix<double>(),
        "performs generalized intersection over R of Z with halfspace Hx <= f");
    handle.def("union_of_many", &union_of_many<double>, py::arg("Z_list"),
        "returns union of Z1, Z2, ...");
    handle.def("cartesian_product", &cartesian_product<double>, py::arg("Z1"), py::arg("Z2"),
        "returns Z = Z1 x Z2");

    // global setup functions
    handle.def("vrep_2_conzono", &vrep_2_conzono<double>, py::arg("Vpoly"), "Vpoly is nV x n numpy array of vertices where n is polytope dimension");
    handle.def("vrep_2_hybzono", &vrep_2_hybzono<double>, py::arg("Vpolys"), "Vpolys is list of nV x n numpy arrays of vertices where n is polytope dimension");
    handle.def("zono_union_2_hybzono", &zono_union_2_hybzono<double>, py::arg("Zs"), "Zs is list of zonotopes");
    handle.def("make_regular_zono_2D", &make_regular_zono_2D<double>, py::arg("radius"), py::arg("n_sides"), py::arg("outer_approx")=false, py::arg("c")=Eigen::Vector<double, 2>::Zero(),
        "makes a regular zonotope with specified radius and number of sides centered at c. If outer_approx is true, then Z_o contains the circle of specified radius centered at c. Otherwise, the circle contains Z_o.");
    handle.def("interval_2_zono", &interval_2_zono<double>, py::arg("a"), py::arg("b"), "converts the interval [a,b] to a zonotope");
}