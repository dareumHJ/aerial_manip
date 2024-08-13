# Do Inverse Kinematics once then Control...

import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt

xml = "mujoco_menagerie/skydio_x2/scene.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

#Video Setup
DURATION = 20 #(seconds)
FRAMERATE = 60 #(Hz)
goal_freq = 10
goal_t = 1
frames = []

#Reset state and time.
mujoco.mj_resetData(model, data)

#Init position.
pi = np.pi
qinit = [0, pi/2, -pi/2]
# data.qpos =  qinit #ENABLE if you want test circle

#Init parameters
M = np.zeros((model.nv, model.nv))
jacp = np.zeros((3, model.nv)) #translation jacobian
jacr = np.zeros((3, model.nv)) #rotational jacobian
dt = 0.005
gain = 20
tol = 0.001
damping = 0.0
Kp = 200.0
Kd = 20.0


#Get error.
end_effector_id = model.body('E_E').id
current_pose = data.body(end_effector_id).xpos.copy()

#Goal Position set
goal = [0.011, 0.01, 0.7] #Desire position
goal2 = [0.011, 0.01, 0.37]

#Error initialization
error = np.subtract(goal, current_pose) #Init Error
prev_error = 0

#Function Definitions

def compute_com(model, data):
    total_mass = np.sum(model.body_mass)

    mass = (model.body_mass)
    xpos = (data.xipos)

    com = np.sum(xpos.T * mass, axis=1) / total_mass
    return com

def check_joint_limits(q):
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

def inv_kinematics(model, data, jacp, jacr, goal, end_effector_id, error):
    #Calculate jacobian
    mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
    n = jacp.shape[1]
    I = np.identity(n)
    product = jacp.T @ jacp + damping * I
    if np.isclose(np.linalg.det(product), 0):
        j_inv = np.linalg.pinv(product) @ jacp.T
    else:
        j_inv = np.linalg.inv(product) @ jacp.T
    delta_q = j_inv @ error
    return delta_q

def get_coeff(qk, qk1, dqk, dqk1, hk):
    ak0 = qk
    ak1 = dqk
    ak2 = - 2/hk*dqk - 1/hk*dqk1 - 3/pow(hk, 2)*qk + 3/pow(hk, 2)*qk1
    ak3 = 1/pow(hk, 2)*dqk + 1/pow(hk, 2)*dqk1 + 2/pow(hk, 3)*qk - 2/pow(hk, 3)*qk1
    return ak0, ak1, ak2, ak3

def cal_q_goal(ak0, ak1, ak2, ak3, t):
    return ak0 + ak1 * t + ak2 * pow(t, 2) + ak3 * pow(t, 3)

# Get goal qpos using iterative inverse kinematics
while (np.linalg.norm(error) > tol):
    dq = inv_kinematics(model, data, jacp, jacr, goal2, end_effector_id, error)
    q = data.qpos.copy()
    q += dq * 0.05
    data.qpos = q
    mujoco.mj_forward(model, data)
    error = goal2 - data.body(end_effector_id).xpos
    print(data.body(end_effector_id).xpos)
    
print("ITERATION CHECK")
print(data.body(end_effector_id).xpos)
print(np.linalg.norm(error))
q_goal = data.qpos.copy()

# reset
mujoco.mj_resetData(model, data)
data.qpos = [0, pi/2, -pi/2]

# new jacobian
# _jacp = np.zeros((3, model.nv))
# _jacr = np.zeros((3, model.nv))

# new initialize
# _error = 0
# _prev_error = 0

a10, ak11, ak12, ak13 = get_coeff(qinit[0], q_goal[0], 0, 0, goal_t)
a20, ak21, ak22, ak23 = get_coeff(qinit[1], q_goal[1], 0, 0, goal_t)
a30, ak31, ak32, ak33 = get_coeff(qinit[2], q_goal[2], 0, 0, goal_t)

print("Q goal and COEEFICIENTS")
print(q_goal)
print(a20, ak21, ak22, ak23)
print(cal_q_goal(a20, ak21, ak22, ak23, DURATION))
print("---------------------------")

# Iteration count
iter_cnt = 0

#Simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:

        # Calculate_MCG
        mujoco.mj_fullM(model, M, data.qM)
        cg = data.qfrc_bias
        print(cg)

        com = compute_com(model, data)

        # calculate error and error_d
        if(((iter_cnt % goal_freq) == 0) and (data.time + 1/goal_freq) <= goal_t):
            q_goal[0] = cal_q_goal(a10, ak11, ak12, ak13, data.time + 1/goal_freq)
            q_goal[1] = cal_q_goal(a20, ak21, ak22, ak23, data.time + 1/goal_freq)
            q_goal[2] = cal_q_goal(a30, ak31, ak32, ak33, data.time + 1/goal_freq)
        

        error = np.subtract(q_goal, data.qpos)
        error_d = (error - prev_error)/dt

        print(np.linalg.norm(error))
        
        # new error
        # _error = np.subtract(goal, data.body(end_effector_id).xpos)
        # _error_d = (_error - _prev_error)/dt
        #Check limits
        # check_joint_limits(data.qpos)
        
        ctrlval = M @ (Kp * error + Kd * error_d)

        # mujoco.mj_jacBody(model, data, _jacp, _jacr, end_efFector_id)
        # ctrlval = _jacp.T @ (-Kp * _error - Kd * _error_d)
        #Set control signal

        data.ctrl = np.append(np.zeros(4), np.array((cg + ctrlval))) 

        #Step the simulation.
        mujoco.mj_step(model, data)
        #Update previous error
        prev_error = error
        # _prev_error - _error

        # viewer.user_scn.ngeom = 0
        # mujoco.mjv_initGeom(
        #     viewer.user_scn.geoms[0],
        #     type=mujoco.mjtGeom.mjGEOM_SPHERE,    
        #     size=[0.03, 0.03, 0],
        #     pos=(com),
        #     mat=np.eye(3).flatten(),
        #     rgba=np.array([1, 0, 0, 1])
        # )
        # viewer.user_scn.ngeom += 1

        #Render and save frames.
        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

            model.vis.scale.contactwidth = 0.05
            model.vis.scale.contactheight = 0.015
            model.vis.scale.forcewidth = 0.025
            model.vis.scale.com = 0.7
            model.vis.map.force = 0.15

        viewer.sync()
        iter_cnt += 1
        
#Display video.
media.show_video(frames, fps=FRAMERATE)