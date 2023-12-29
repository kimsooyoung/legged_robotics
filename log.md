
unitree_quadruped 
> quadruped_example.py 
> quadruped_example_extension.py
> QuadrupedExampleExtension class 
> quadruped_example.py
  from omni.isaac.quadruped.robots import Unitree

omni.isaac.quadruped > omni\isaac\quadruped
> robots
> Unitree

이제 여기서부터 분석 시작

==========================================
Unitree - Constructor

A1State: base_frame(FrameState)/joint_pos/joint_vel
    FrameState
    - name
    - pos
    - quat
    - lin_vel
    - ang_vel
    - pose
A1Measurement: state/foot_forces/base_lin_acc/base_ang_vel
A1Command: desired_joint_torque


ContactSensor * 4 - omni.isaac.sensor
[] foot_filter? 
    self._foot_filters = [deque(), deque(), deque(), deque()]

IMUSensor - omni.isaac.sensor


A1QPController

set_state 
- self.set_world_pose
- self.set_linear_velocity
- self.set_angular_velocity
- self.set_joint_positions
- self.set_joint_velocities
- self.set_joint_efforts
update_contact_sensor_data
- self.foot_force
update_imu_sensor_data 
- self.base_lin 
- self.ang_vel
update
- update robot sensor variables, state variables in A1Measurement
advance
- compute desired torque and set articulation effort to robot joints
- self._command.desired_joint_torque = self._qp_controller.advance(...)
> self._qp_controller.advance를 봐야 한다.
> self._qp_controller = A1QPController()
> from omni.isaac.quadruped.controllers import A1QPController

사용되는 함수 
_qp_controller.reset
_qp_controller.setup
_qp_controller.set_target_command
_qp_controller.advance
==========================================
controllers\qp_controller.py - A1QPController

Constructor
self._ctrl_params = A1CtrlParams()
- _kp_foot
- _kd_foot
- _km_foot
- _kp_linear
- _kd_linear
- _kp_angular
- _kd_angular
- _torque_gravity
self._ctrl_states = A1CtrlStates()
- 뭐가 엄청 많다.
self._desired_states = A1DesiredStates()
- _root_pos_d : desired body position
- _root_lin_vel_d : desired body velocity
- _euler_d : desired body orientation
- _root_ang_vel_d : desired body angular velocity
self._root_control = A1RobotControl()
> a1_robot_control.py
self._sys_model = A1SysModel()
- kinematics
# toggle standing/moving mode

def ctrl_state_reset
- _ctrl_params._kp_linear : foot force position
- _ctrl_params._kd_linear : foot force velocity
- _ctrl_params._kp_angular : foot force orientation
- _ctrl_params._kd_angular : foot force orientation
- _ctrl_params._kp_foot : swing foot position
- _ctrl_params._kd_foot : swing foot velocity
- _ctrl_params._km_foot : swing foot force amplitude
- _ctrl_params._robot_mass : mass of the robot
- _ctrl_params._foot_force_low : low threshold of foot contact force
- _ctrl_states._counter
- _ctrl_states._gait_counter
- _ctrl_states._exp_time

def update
    Fill measurement into _ctrl_states
- self._ctrl_states._euler
- self._ctrl_states._rot_mat
- self._ctrl_states._root_ang_vel
- self._ctrl_states._rot_mat_z
- self._ctrl_states._joint_vel
- self._ctrl_states._joint_pos
- self._ctrl_states._foot_pos_rel > kinematics 
- self._ctrl_states._j_foot > jacobian
- self._ctrl_states._foot_pos_abs
- self._ctrl_states._foot_forces

def set_target_command
    Set target base velocity command from joystick

def advance
    Perform torque command generation.


==============================


generate_ctrl 
> _compute_grf
> _get_qp_params