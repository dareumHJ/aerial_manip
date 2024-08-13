import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media
import math
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Choose a model
xml = "test_leg.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

#Video Setup
DURATION = 6 #(seconds)
FRAMERATE = 100 #(Hz)
frames = []

step_size = 0.005
tol = 0.01
alpha = 0.5
damping = 0.001
dt = 0.005

# Figure plot setup
time_data = []
fx_fromAcc = []
fy_fromAcc = []
fz_fromAcc = []

fx_fromGRF = []
fy_fromGRF = []
fz_fromGRF = []

ext_fx = []
ext_fy = []
ext_fz = []

ext_f_total = []

def compute_com(model, data):
    total_mass = np.sum(model.body_mass)

    mass = (model.body_mass)
    xpos = (data.xipos)

    com = np.sum(xpos.T * mass, axis=1) / total_mass
    return com

def compute_com_vel(model, data):
    total_mass = np.sum(model.body_mass)

    mass = (model.body_mass)
    cvel = (data.cvel[:, 3:])

    com_vel = np.sum(cvel.T * mass, axis=1) / total_mass
    return com_vel

def compute_com_acc(cur_vel, prev_vel):
    com_acc = (cur_vel - prev_vel) / dt
    return com_acc

def glb_to_foot(model, data, c1, c2, c3):
    foot_pos = data.body(model.body('L_FOOT').id).xpos.copy()
    foot_mat = data.body(model.body('L_FOOT').id).xmat.copy().reshape(3, 3)
    local_c1 = np.linalg.inv(foot_mat) @ (c1 - foot_pos)
    local_c2 = np.linalg.inv(foot_mat) @ (c2 - foot_pos)
    local_c3 = np.linalg.inv(foot_mat) @ (c3 - foot_pos)
    return local_c1, local_c2, local_c3

def foot_to_glb(model, data, v):
    foot_pos = data.body(model.body('L_FOOT').id).xpos.copy()
    foot_mat = data.body(model.body('L_FOOT').id).xmat.copy().reshape(3, 3)
    global_v = foot_pos + (foot_mat @ v)
    return global_v

def cal_cop(model, data, f1, f2, f3, c1, c2, c3): # in foot frame!
    # f1, f2, f3 are the global frame force
    # need to convert
    foot_mat = data.body(model.body('L_FOOT').id).xmat.copy().reshape(3, 3) # foot to global
    foot_mat = np.linalg.inv(foot_mat) # global to foot
    f1, f2, f3 = foot_mat @ (f1, f2, f3) # in foot frame

    temp_sum = np.cross(c1, f1) + np.cross(c2, f2) + np.cross(c3, f3)
    f_sum = f1 + f2 + f3
    ref_z = -(data.body(model.body('L_FOOT').id).xpos.copy())[2]

    px = (ref_z*f_sum[0] - temp_sum[1])/f_sum[2]
    py = (ref_z*f_sum[1] + temp_sum[0])/f_sum[2]
    pz = ref_z
    return px, py, pz

def get_ext_force(acc, m, grf): # in global frame
    ddx = acc[0]
    ddy = acc[1]
    ddz = acc[2]
    g = 9.81

    Fx = m*ddx - grf[0]
    Fy = m*ddy - grf[1]
    Fz = m*(ddz + g) - grf[2]

    return Fx, Fy, Fz

def get_zmp(model, data, com, acc, m, grf, cop): # in local frame
    # cop is in the foot frame
    # need to convert it
    foot_mat = data.body(model.body('L_FOOT').id).xmat.copy().reshape(3, 3) # foot to global
    cop = foot_mat @ cop # in foot frame
    ref_z = -(data.body(model.body('L_FOOT').id).xpos.copy())[2]

    tau1 = data.ctrl[2] # in foot frame
    tau2 = data.ctrl[1] # in foot frame (need to be modified)
    ext_f = get_ext_force(acc, m, grf) # in foot frame
    ext_fx.append(ext_f[0])
    ext_fy.append(ext_f[1])
    ext_fz.append(ext_f[2])
    ext_f = np.array(ext_f)
    ext_f_total.append(np.linalg.norm(ext_f))
    g = 9.81 # in foot frame

    print((grf[2] + ext_f[2] - m*g))

    zmp_x = (-tau2 - (com[2] - ref_z)*ext_f[0] + grf[2]*cop[0] + (ext_f[2] - m*g) * com[0]) / (grf[2] + ext_f[2] - m*g)
    zmp_y = (tau1 - (com[2] - ref_z)*ext_f[1] + grf[2]*cop[1] + (ext_f[2] - m*g) * com[1])/ (grf[2] + ext_f[2] - m*g)
    zmp_z = -(data.body(model.body('L_FOOT').id).xpos.copy())[2]

    return zmp_x, zmp_y, zmp_z

def is_point_in_triangle(A, B, C, P):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(P, A, B)
    d2 = sign(P, B, C)
    d3 = sign(P, C, A)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

mujoco.mj_resetData(model, data)

# 70 ~ 90
des_qpos1 = 80*np.pi/180
des_qpos2 = 90*np.pi/180

# des_qpos1 = 0*np.pi/180
# des_qpos2 = 0*np.pi/180

unit_vec1 = np.array([1, 0, 0])
unit_vec2 = np.array([0, 1, 0])
unit_vec3 = np.array([0, 0, 1])
norm_vec1 = np.array([1, 0, 1])
norm_vec1 = norm_vec1 / np.linalg.norm(norm_vec1)

Ks = 100.0
Bs = 10.0
Ks2 = 100.0
Bs2 = 10.0

plant_idx = (data.qpos.size) - 1
inver_idx = (data.qpos.size) - 2

# dq = np.zeros((1, 3))

# vars for explicit velocity calculation
cur_vel = compute_com_vel(model, data)
prev_vel = compute_com_vel(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    if(data.time <= 0.005):
        # vars for explicit velocity calculation
        cur_vel = compute_com_vel(model, data)
        prev_vel = compute_com_vel(model, data)
    while data.time < DURATION:
        # mujoco.mj_differentiatePos(model, dq, step_size, qpos0, data.qpos)

        # ROTATION MATRIX (Global -> Local)
        mat1 = (data.body(model.body('L_FOOT').id).xmat.copy()).reshape(3, 3)
        mat2 = (data.body(model.body('L_ANKLE').id).xmat.copy()).reshape(3, 3)


        joint1_pos = data.body(model.body('L_FOOT').id).xpos.copy() + mat1 @ np.array([-0.15552, 0.17526, -0.9975])
        joint2_pos = data.body(model.body('L_ANKLE').id).xpos.copy() + mat2 @ np.array([-0.06603, 0.10476, -0.90801])
        com = compute_com(model, data)
        det_vec1 = com - joint1_pos
        det_vec2 = com - joint2_pos


        proj_vec1 = np.linalg.inv(mat1) @ det_vec1
        proj_vec1 = proj_vec1 - (np.dot(proj_vec1, unit_vec2)) * unit_vec2

        proj_vec2 = np.linalg.inv(mat2) @ det_vec2
        proj_vec2 = proj_vec2 - (np.dot(proj_vec2, norm_vec1)) * norm_vec1
        
        cur_pos1 = math.acos(np.dot((proj_vec1), unit_vec1) / np.linalg.norm(proj_vec1))
        cur_pos2 = math.acos(np.dot((proj_vec2), unit_vec2) / np.linalg.norm(proj_vec2))
        err1 = -(des_qpos1 - cur_pos1)
        # err1 = des_qpos1 - (data.qpos.copy())[plant_idx]
        err2 = -(des_qpos2 - cur_pos2)
        # err2 = des_qpos2 - (data.qpos.copy())[inver_idx]

        errdiff1 = 0.0 - (data.qvel.copy())[plant_idx - 1]
        errdiff2 = 0.0 - (data.qvel.copy())[inver_idx - 1]

        #Set control signal
        ctrlval1 = (Ks * err1 + Bs * errdiff1)
        ctrlval2 = (Ks2 * err2 + Bs2 * errdiff2)
        data.ctrl = [0, ctrlval2, ctrlval1]
        # data.ctrl = [0, 0, 0]

        if (data.time > 2 and data.time < 4):
            data.xfrc_applied[2] = [10, 0, 20, 0, 0, 0]
        else:
            data.xfrc_applied[2] = [0, 0, 0, 0, 0, 0]
        # Step the simulation.
        mujoco.mj_step(model, data)


        # ↑ Control and Update mujoco PF
        # ------------------------------------------------------------------------------------------
        # ↓ Get and Determine ZMP & CoP


        # Get Acceleration of the CoM
        prev_vel = cur_vel
        cur_vel = compute_com_vel(model, data)
        cur_acc = compute_com_acc(cur_vel, prev_vel)

        # Define variables for contact forces & contact points
        res1 = np.zeros(6)
        res2 = np.zeros(6)
        res3 = np.zeros(6)
        contact_points = []

        # Case: Contact is stable
        if len(data.contact.pos) > 2:
            for i in range(data.ncon):
                contact = data.contact[i]
                geom_id1 = contact.geom1
                geom_id2 = contact.geom2
                pos = contact.pos
                contact_points.append([pos[0], pos[1], pos[2]])
            
            # Get contact points (in local frame)
            contact_points = np.array(contact_points)
            c1 = contact_points[0]
            c2 = contact_points[1]
            c3 = contact_points[2]
            c1, c2, c3 = glb_to_foot(model, data, c1, c2, c3)


            # Get contact forces (in local frame)
            mujoco.mj_contactForce(model, data, 0, res1) 
            mujoco.mj_contactForce(model, data, 1, res2)
            mujoco.mj_contactForce(model, data, 2, res3)

            totmass = np.sum(model.body_mass)

            # Data update for plotting
            time_data.append(data.time)
            fx_fromAcc.append(totmass * cur_acc[0])
            fy_fromAcc.append(totmass * cur_acc[1])
            fz_fromAcc.append(totmass * cur_acc[2])

            # Convert Forces from contact frame to global frame
            mat4 = np.array((data.contact.frame.copy())[0].reshape(3, 3)) # 90' wrt y-axis rotation, global frame to contact frame
            mat4 = np.linalg.inv(mat4) # contact frame to global frame
            res1 = mat4 @ (res1[:3])
            res2 = mat4 @ (res2[:3])
            res3 = mat4 @ (res3[:3])
            print("-----GRF-----")
            print(res1, res2, res3)
            print("-------------")
            
            res = res1[:3] + res2[:3] + res3[:3]
            fx_fromGRF.append(res[0])
            fy_fromGRF.append(res[1])
            fz_fromGRF.append(res[2])

            # calculate ZMP
            cop = cal_cop(model, data, res1, res2, res3, c1, c2, c3) # CLEAR!!!!!!!!!!
            zmp = get_zmp(model, data, com, cur_acc, totmass, res, cop)
            # ----------------------------------
            print(zmp)

            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,    
                size=[0.03, 0.03, 0],
                pos=(foot_to_glb(model, data, np.array(zmp))),
                # pos=(foot_to_glb(model, data, np.array(c1))),
                # pos = compute_com(model, data),
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 1])
            )
            viewer.user_scn.ngeom += 1
        
        # mujoco.mjv_initGeom(
        #     viewer.user_scn.geoms[1],
        #     type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #     size=[0.03, 0.03, 0],
        #     pos = [0, 0, 0],
        #     # pos=com,
        #     mat=np.eye(3).flatten(),
        #     rgba=np.array([1, 0, 0, 1])
        # )
        # viewer.user_scn.ngeom += 1

        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        viewer.sync()

        with viewer.lock():
            model.vis.scale.com = 0.1
            # model.vis.scale.frames = 0.1

# Convert collected data to numpy arrays for easier handling

st = 100
ft = 600

time_data = np.array(time_data)[st:ft]
fx_fromAcc = np.array(fx_fromAcc)[st:ft]
fy_fromAcc = np.array(fy_fromAcc)[st:ft]
fz_fromAcc = np.array(fz_fromAcc)[st:ft]

fx_fromGRF = np.array(fx_fromGRF)[st:ft]
fy_fromGRF = np.array(fy_fromGRF)[st:ft]
fz_fromGRF = np.array(fz_fromGRF)[st:ft]

ext_fx = np.array(ext_fx)[st:ft]
ext_fy = np.array(ext_fy)[st:ft]
ext_fz = np.array(ext_fz)[st:ft]

ext_f_total = np.array(ext_f_total)[st:ft]

# Plot the results after the simulation
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 10))

ax1.plot(time_data, fx_fromAcc, label = 'fx_acc')
ax1.plot(time_data, fx_fromGRF, label = 'grf_x')
ax1.plot(time_data, ext_fx, label = 'ext_fx')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Force (N)')
ax1.set_title('External Force on the body (X-axis)')
ax1.legend()
ax1.grid(True)

ax2.plot(time_data, fy_fromAcc, label = 'fy_acc')
ax2.plot(time_data, fy_fromGRF, label = 'grf_y')
ax2.plot(time_data, ext_fy, label = 'ext_fy')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Force (N)')
ax2.set_title('External Force on the body (Y-axis)')
ax2.legend()
ax2.grid(True)

ax3.plot(time_data, fz_fromAcc, label = 'fz_acc')
ax3.plot(time_data, fz_fromGRF, label = 'grf_z')
ax3.plot(time_data, ext_fz, label = 'ext_fz')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Force (N)')
ax3.set_title('External Force on the body (Z-axis)')
ax3.legend()
ax3.grid(True)

ax4.plot(time_data, ext_f_total, label = 'force')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Force (N)')
ax4.set_title('External Force on the body')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show(block=True)

media.show_video(frames, fps=FRAMERATE)