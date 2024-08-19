# Do Inverse Kinematics once then Control...

import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media
import math
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

#Init parameters
M = np.zeros((model.nv, model.nv))
jacp = np.zeros((3, model.nv)) #translation jacobian
jacr = np.zeros((3, model.nv)) #rotational jacobian
dt = 0.005
gain = 20
tol = 0.001
damping = 0.0
Kp_f = 100.0
Kd_f = 10.0
Kp_M = 8.0
Kd_M = 2.0

d_body_pos = [0, 0, 0.5]
d_body_vel = [0, 0, 0]

d_body_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
print(d_body_rot)
d_body_avel = [0, 0, 0]

e1 = np.array([1.0, 0, 0])
e2 = np.array([0, 1.0, 0])
e3 = np.array([0, 0, 1.0])

tmat_d = 0.228035
tmat_ctf = 0.0008
thrust_M = np.array([[1, 1, 1, 1], [0, -tmat_d, 0, tmat_d], [tmat_d, 0, -tmat_d, 0], [-tmat_ctf, tmat_ctf, -tmat_ctf, tmat_ctf]])


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

def vee_op(sk_mat):
    a = sk_mat[2][1]
    b = sk_mat[0][2]
    c = sk_mat[1][0]
    return np.array([a, b, c])

def hat_op(vec):
    a = vec[0]
    b = vec[1]
    c = vec[2]
    mat = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    return mat
# reset
mujoco.mj_resetData(model, data)

# Iteration count
iter_cnt = 0

#Simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:

        # Calculate_MCG
        mujoco.mj_fullM(model, M, data.qM)
        cg = data.qfrc_bias

        com = compute_com(model, data)
        m = M[0][0]
        g = 9.81

        # *********************************************************
        # Position Error
        body_pos = data.body(model.body('x2').id).xpos.copy()
        body_mat = data.body(model.body('x2').id).xmat.copy().reshape(3, 3)
        body_vel = (data.body(model.body('x2').id).cvel.copy())[3:]

        pos_err = body_pos - d_body_pos
        vel_err = body_vel - d_body_vel

        # *********************************************************

        # *********************************************************
        # Orientation Error
        body_rot = np.linalg.inv(body_mat)
        body_avel = (data.body(model.body('x2').id).cvel.copy())[:3]

        rot_err = vee_op(d_body_rot.T @ body_rot - body_rot.T @ d_body_rot)/2
        print(rot_err)
        avel_err = body_avel - body_rot.T @ d_body_rot @ d_body_avel
        avel_err = np.zeros(3)

        # *********************************************************

        # *********************************************************
        # Control Inputs
        J = np.diag(model.body_inertia[1])
        # print(J)
        ctrl_f = np.dot(-(-Kp_f*pos_err - Kd_f*vel_err - m*g*e3), (body_rot @ e3))
        ctrl_f = 0
        # ctrl_f = np.dot(-(-m*g*e3), (body_rot @ e3))
        # ctrl_M = np.cross(body_avel, J@body_avel) - J@(hat_op(body_avel)@body_rot.T@d_body_rot@d_body_avel)
        # print(ctrl_M)
        ctrl_M = -Kp_M*rot_err - Kd_M*avel_err + np.cross(body_avel, J@body_avel) - J@(hat_op(body_avel)@body_rot.T@d_body_rot@d_body_avel)
        # print(ctrl_M)
        # ctrl_M = -Kp_M*rot_err - Kd_M*avel_err
        # print(ctrl_M)
        # ctrl_M = np.zeros(3)
        
        # *********************************************************

        # *********************************************************
        # Convert Control inputs to Thrust
        thrust = np.linalg.inv(thrust_M) @ np.append(-ctrl_f, ctrl_M)

        # *********************************************************


        # data.ctrl = np.zeros(4)
        data.ctrl = thrust
        # data.ctrl = np.array([1, 1, 1, 1]) * (m * g / 4)

        #Step the simulation.
        mujoco.mj_step(model, data)

        # *********************************************************
        # Print Error
        # print("**** PRINT ERROR ****")
        # print("POS error: ")
        # print(pos_err)
        # print("VEL error: ")
        # print(vel_err)
        # print("Attitude error: ")
        # print(rot_err)
        # print("Angular VEL error: ")
        # print(avel_err)

        # *********************************************************



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