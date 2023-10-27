#include <rigid_body_mpc.hpp>

RigidBodyMPC::RigidBodyMPC()
{
    double weight[] = MPC_STATE_ERROR_WEIGHT;

    for (int i = 0; i < 12; i++)
    {
        state_weight(i) = weight[i];
    }
    state_weight(12) = 0.0;

    qpSolver.settings()->setVerbosity(false);
}

RigidBodyMPC::~RigidBodyMPC()
{
}

void RigidBodyMPC::setup(int predictionHorizon, double dt, double mass, double mu, double max_force, double ixx, double iyy, double izz)
{
    this->dt = dt;
    this->horizonLen = predictionHorizon;

    this->mass = mass;
    this->mu = mu;
    this->max_f = max_force;

    this->I.setZero();
    this->I(0, 0) = ixx;
    this->I(1, 1) = iyy;
    this->I(2, 2) = izz;

    X_ref.resize(13 * horizonLen, NoChange);

    C_qp.resize(20 * horizonLen, 12 * horizonLen);
    L_qp.resize(13 * horizonLen, 13 * horizonLen);
    K_qp.resize(12 * horizonLen, 12 * horizonLen);
    A_qp.resize(13 * horizonLen, NoChange);
    B_qp.resize(13 * horizonLen, 12 * horizonLen);
    g_qp.resize(12 * horizonLen, NoChange);
    H_qp.resize(12 * horizonLen, 12 * horizonLen);
    c_l.resize(20 * horizonLen, NoChange);
    c_u.resize(20 * horizonLen, NoChange);
    stateTrajectory.resize(12 * horizonLen, NoChange);

    c_l.setZero();

    friction << 1.0 / mu, 0.0, 1.0,
        -1.0 / mu, 0.0, 1.0,
        0.0, 1.0 / mu, 1.f,
        0.0, -1.0 / mu, 1.0,
        0.0, 0.0, 1.0;

    for (int i = 0; i < horizonLen * 4; i++)
    {
        C_qp.block(5 * i, 3 * i, 5, 3) = friction;
    }

    C_qp_spare = C_qp.sparseView();

    L_qp.setZero();
    L_qp.diagonal() = state_weight.replicate(horizonLen, 1);

    K_qp.setIdentity();
    K_qp *= MPC_INPUT_WEIGHT;
}

void RigidBodyMPC::buildContinuousSSM()
{
    A_con.setZero();
    B_con.setZero();

    A_con.block(3, 9, 3, 3) = Matrix<double, 3, 3>::Identity();
    A_con(11, 12) = 1.0;

    A_con.block(0, 6, 3, 3) = R_z.transpose();

    Matrix<double, 3, 3> I_world_inv = I_world.inverse();

    for (int i = 0; i < 4; i++)
    {
        Matrix<double, 3, 3> cross_m;
        double *foot = &feet_pos[i * 3];

        cross_m << 0.0, -foot[2], foot[1],
            foot[2], 0.0, -foot[0],
            -foot[1], foot[0], 0.0;

        B_con.block(6, i * 3, 3, 3) = I_world_inv * cross_m;
        B_con(9, i * 3 + 0) = 1.0 / mass;
        B_con(10, i * 3 + 1) = 1.0 / mass;
        B_con(11, i * 3 + 2) = 1.0 / mass;
    }
}

void RigidBodyMPC::convertToDiscreteSSM()
{
    AB_con.setZero();

    AB_con.block(0, 0, 13, 13) = A_con;
    AB_con.block(0, 13, 13, 12) = B_con;

    AB_con *= dt;

    exp_AB_con = AB_con.exp();

    A_des = exp_AB_con.block(0, 0, 13, 13);
    B_des = exp_AB_con.block(0, 13, 13, 12);
}

void RigidBodyMPC::convertToQPForm()
{
    //TODO: Optimize this function using DP
    for (int row = 0; row < horizonLen; row++)
    {
        A_qp.block(13 * row, 0, 13, 13) = A_des.pow(row + 1);

        for (int col = 0; col < horizonLen; col++)
        {
            if (row >= col)
            {
                B_qp.block(13 * row, 12 * col, 13, 12) = A_des.pow(row - col) * B_des;
            }
        }
    }

    H_qp = 2.0 * (B_qp.transpose() * L_qp * B_qp + K_qp);
    g_qp = 2.0 * B_qp.transpose() * L_qp * (A_qp * current_X - X_ref);

    H_qp_spare = H_qp.sparseView();

    int k = 0;
    for (int i = 0; i < horizonLen; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            c_u(5 * k + 0) = numeric_limits<double>::infinity();
            c_u(5 * k + 1) = numeric_limits<double>::infinity();
            c_u(5 * k + 2) = numeric_limits<double>::infinity();
            c_u(5 * k + 3) = numeric_limits<double>::infinity();
            c_u(5 * k + 4) = max_f * (double)gait[i * 4 + j];
            k++;
        }
    }
}

void RigidBodyMPC::run(double *current_state, double *ref_state, bool *gait, double *feet_pos)
{
    this->feet_pos = feet_pos;
    this->gait = gait;

    for (int i = 0; i < 12; i++)
    {
        current_X(i) = current_state[i];
    }
    current_X(12) = (double)SSM_GRAVITY;

    for (int i = 0; i < horizonLen; i++)
    {
        for (int j = 0; j < 12; j++)
        {
            X_ref(i * 13 + j) = ref_state[i * 12 + j];
        }
        X_ref(i * 13 + 12) = (double)SSM_GRAVITY;
    }

    double yaw = current_state[2];
    double c_yaw = cos(yaw);
    double s_yaw = sin(yaw);

    R_z << c_yaw, -s_yaw, 0.0,
        s_yaw, c_yaw, 0.0,
        0.0, 0.0, 1.0;

    I_world = R_z * I * R_z.transpose();

    buildContinuousSSM();
    convertToDiscreteSSM();
    convertToQPForm();

    qpSolver.data()->setNumberOfVariables(12 * horizonLen);
    qpSolver.data()->setNumberOfConstraints(20 * horizonLen);
    qpSolver.data()->setLinearConstraintsMatrix(C_qp_spare);
    qpSolver.data()->setHessianMatrix(H_qp_spare);
    qpSolver.data()->setGradient(g_qp);
    qpSolver.data()->setLowerBound(c_l);
    qpSolver.data()->setUpperBound(c_u);

    qpSolver.initSolver();
    qpSolver.solve();

    U = qpSolver.getSolution();

    U_body = U;
    input = U.data();

    for (int i = 0; i < 4; i++)
    {
        Matrix<double, 3, 1> body_f;
        Matrix<double, 3, 1> world_f;
        body_f(0) = input[i * 3 + 0];
        body_f(1) = input[i * 3 + 1];
        body_f(2) = input[i * 3 + 2];

        world_f = -R_z.transpose() * body_f;

        input[i * 3 + 0] = world_f(0);
        input[i * 3 + 1] = world_f(1);
        input[i * 3 + 2] = world_f(2);
    }

    qpSolver.data()->clearLinearConstraintsMatrix();
    qpSolver.data()->clearHessianMatrix();
    qpSolver.clearSolver();
}

void RigidBodyMPC::getResults(double *&input_forces_world, double *&input_forces_body)
{
    input_forces_world = this->input;
    input_forces_body = this->U_body.data();
}

void RigidBodyMPC::getPredictedTrajectory(double *&inputTrajectory, double *&stateTrajectory)
{
    inputTrajectory = this->input;

    auto X = A_qp * current_X + B_qp * U;

    for (int i = 0; i < horizonLen; i++)
    {
        for (int j = 0; j < 12; j++)
        {
            this->stateTrajectory(i * 12 + j) = X(i * 13 + j);
        }
    }

    stateTrajectory = this->stateTrajectory.data();
}