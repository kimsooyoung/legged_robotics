/*
Rigid body state space model parameters
*/
#define SSM_GRAVITY -9.81

/*
MPC parameters
*/
#define MPC_STATE_ERROR_WEIGHT {10.0, 10.0, 10.0, 50.0, 50.0, 100.0, 0.0, 0.0, 0.5, 3.0, 3.0, 3.0}
#define MPC_INPUT_WEIGHT 1e-6