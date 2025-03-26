#include "ZonoOpt.hpp"
#include <iostream>
#include <exception>

using namespace ZonoOpt;

int main()
{
    // create conzono
    Eigen::SparseMatrix<float> G(2, 2);
    G.insert(0, 0) = 1;
    G.insert(1, 1) = 1;

    Eigen::Vector<float, -1> c(2);

    Eigen::SparseMatrix<float> A(1, 2); 
    A.insert(0, 0) = 1;

    Eigen::Vector<float, -1> b(1);
    b(0) = 1;

    ConZono<float> Z1 (G, c, A, b, true);
    Zono<float> Z2 (G, c, false);

    // test polymorphism
    std::cout << "Z1 is conzono?: " << Z1.is_conzono() << std::endl;
    std::cout << "Z2 is conzono?: " << Z2.is_conzono() << std::endl;

    // try minkowski sum
    ZonoPtrF Z_sum = minkowski_sum(Z1, Z2);
    std::cout << "Z_sum: " << *Z_sum << std::endl;

    // union
    std::vector<AbstractZono<float>*> Zs;
    Zs.push_back(&Z1);
    Zs.push_back(&Z2);

    ZonoPtrF U = union_of_many(Zs);
    std::cout << "U: " << *U << std::endl;
    try
    {
        std::cout << "U is empty? " << U->is_empty() << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    // convex relaxation
    ConZono<float> C = dynamic_cast<HybZono<float>*>(U.get())->convex_relaxation();
    std::cout << "C: " << C << std::endl;

    // project point
    Eigen::VectorXf x (C.get_n());
    x.setOnes();
    ADMM_settings<float> settings;
    settings.verbose = true;

    auto x_proj = C.project_point(x, settings);
    std::cout << x_proj << std::endl;


    // build hybzono with redundant constraints
    Eigen::SparseMatrix<float> Gc_h(2, 2);
    Gc_h.setIdentity();
    Eigen::SparseMatrix<float> Gb_h(2, 2);
    Gb_h.setIdentity();
    Eigen::Vector<float, 2> c_h;
    c_h.setZero();

    Eigen::MatrixXf Acd_h = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf Abd_h = Eigen::MatrixXf::Ones(2, 2);
    Eigen::SparseMatrix<float> Ac_h = Acd_h.sparseView();
    Eigen::SparseMatrix<float> Ab_h = Abd_h.sparseView();
    Eigen::Vector<float, 2> b_h;
    b_h << 1, 1;

    HybZono<float> Zh (Gc_h, Gb_h, c_h, Ac_h, Ab_h, b_h, true);
    Zh.remove_redundancy();
    
    std::cout << "Zh: " << Zh << std::endl;

    return 0;
}