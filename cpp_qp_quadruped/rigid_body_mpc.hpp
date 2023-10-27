#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <ros/ros.h>
#include <iostream>
#include <parameters.hpp>
#include <OsqpEigen/OsqpEigen.h>

using Eigen::Dynamic;
using Eigen::NoChange;
using namespace Eigen;
using namespace std;

/*

xhat = Ac * x[k] + Bc * u[k]

x[k+1] = Ad * x[k] + Bd * u[k]

x = [
    roll,
    pitch,
    yaw,

    x,
    y,
    z,

    roll_vel,
    pitch_vel,
    yaw_vel,

    x_vel,
    y_vel,
    z_vel,

    g
]

u = [
    force1_x,
    force1_y,
    force1_z,

    force2_x,
    force2_y,
    force2_z,

    force3_x,
    force3_y,
    force3_z,

    force4_x,
    force4_y,
    force4_z,
]

*/

class RigidBodyMPC
{
private:
    double mass = 0.0;
    double mu = 0.0;
    double max_f = 0.0;
    Matrix<double, 3, 3> I;

    double *feet_pos;
    bool *gait;

    Matrix<double, 13, 1> state_weight;

    Matrix<double, 3, 3> I_world;

    Matrix<double, 5, 3> friction;

    Matrix<double, 13, 13> A_con;
    Matrix<double, 13, 12> B_con;

    Matrix<double, Dynamic, 1> X_ref;
    Matrix<double, 13, 1> current_X;

    Matrix<double, 3, 3> R_z;

    Matrix<double, 25, 25> AB_con;
    Matrix<double, 25, 25> exp_AB_con;

    Matrix<double, 13, 13> A_des;
    Matrix<double, 13, 12> B_des;

    Matrix<double, Dynamic, 13> A_qp;
    Matrix<double, Dynamic, Dynamic> B_qp;
    Matrix<double, Dynamic, Dynamic> L_qp;
    Matrix<double, Dynamic, Dynamic> K_qp;
    Matrix<double, Dynamic, Dynamic> C_qp;
    Matrix<double, Dynamic, Dynamic> H_qp;
    Matrix<double, Dynamic, 1> g_qp;
    Matrix<double, Dynamic, 1> c_u;
    Matrix<double, Dynamic, 1> c_l;

    Matrix<double, Dynamic, 1> U;
    Matrix<double, 12, 1> U_body;
    Matrix<double, Dynamic, 1> stateTrajectory;
    double *input;

    SparseMatrix<double, Eigen::ColMajor> H_qp_spare;
    SparseMatrix<double, Eigen::ColMajor> C_qp_spare;

    OsqpEigen::Solver qpSolver;

    double dt = 0.0;
    int horizonLen = 0;

    void buildContinuousSSM();
    void convertToDiscreteSSM();
    void convertToQPForm();

    int a = 0;

public:
    RigidBodyMPC();
    ~RigidBodyMPC();

    void setup(int predictionHorizon, double dt, double mass, double mu, double max_force, double ixx, double iyy, double izz);
    void run(double *current_state, double *ref_state, bool *gait, double *feet_pos);
    void getResults(double *&input_forces, double *&input_forces_body);
    void getPredictedTrajectory(double *&inputTrajectory, double *&stateTrajectory);
};
