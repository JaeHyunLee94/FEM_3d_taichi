# This code is an implementation of the
# SIGGRAPH 2012 Course "FEM Simulation of 3D Deformable Solids" by Eftychios Sifakis and Jernej Barbič
# Please check https://viterbi-web.usc.edu/~jbarbic/femdefo/ for more details

import tetgen
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, device_memory_fraction=0.9)

# cube mesh data vertices and faces
v = np.array([[0, 0, 0], [1, 0, 0],
              [1, 1, 0], [0, 1, 0],
              [0, 0, 1], [1, 0, 1],
              [1, 1, 1], [0, 1, 1], ])
f = np.vstack([[0, 1, 2], [2, 3, 0],
               [0, 1, 5], [5, 4, 0],
               [1, 2, 6], [6, 5, 1],
               [2, 3, 7], [7, 6, 2],
               [3, 0, 4], [4, 7, 3],
               [4, 5, 6], [6, 7, 4]])

# tetrahedralize
tgen = tetgen.TetGen(v, f)
_nodes, _elem = tgen.tetrahedralize()

ti_raw_3d_pos = ti.field(dtype=ti.f32, shape=_nodes.shape)
ti_raw_elem = ti.field(dtype=int, shape=_elem.shape)

ti_raw_3d_pos.from_numpy(_nodes)
ti_raw_elem.from_numpy(_elem)


# Simulation data
dim = 3
vert_num = _nodes.shape[0]
elem_num = _elem.shape[0]

ti_pos = ti.Vector.field(dim, dtype=ti.f32, shape=vert_num)
ti_vel = ti.Vector.field(dim, dtype=ti.f32, shape=vert_num)
ti_force = ti.Vector.field(dim, dtype=ti.f32, shape=vert_num)
ti_mass = ti.field(dtype=ti.f32, shape=vert_num)

ti_elem = ti.Vector.field(4, dtype=int, shape=elem_num)
ti_Dm_inv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=elem_num)
ti_W = ti.field(dtype=ti.f32, shape=elem_num)

ti_face = ti.field(dtype=int, shape=elem_num * 12)
ti_lines = ti.field(dtype=int, shape=elem_num * 12)

# Parameters
dt = 1e-3
gravity = ti.Vector([0.0, 0.0, 0.0])
mu = 1200
lamb = 1200

# for rendering
particle_color = (0, 0.5, 1)
particle_radius = 0.01


# Singular Value Decomposition so that U and V are rotation matrices
@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)): U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)): V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def get_deformation_grad(i):
    """
    calculate the deformation gradient of i-th element
    return: deformation gradient
    """
    c1 = ti_pos[ti_elem[i][0]] - ti_pos[ti_elem[i][3]]
    c2 = ti_pos[ti_elem[i][1]] - ti_pos[ti_elem[i][3]]
    c3 = ti_pos[ti_elem[i][2]] - ti_pos[ti_elem[i][3]]
    Ds = ti.Matrix.cols([c1, c2, c3])
    return Ds @ ti_Dm_inv[i]


@ti.func
def get_piola_corot(F):
    """
    calculate the first Piola-Kirchhoff stress tensor for co-rotational model
    return: Piola stress tensor
    """
    U, sig, V = ssvd(F)
    R = U @ V.transpose()
    return 2 * mu * (F - R) + lamb * ((R.transpose() @ F).trace() - 3) * R


@ti.kernel
def init():
    for v in range(vert_num):
        ti_pos[v] = ti.Vector([ti_raw_3d_pos[v, 0], ti_raw_3d_pos[v, 1], ti_raw_3d_pos[v, 2]])
        ti_vel[v] = ti.Vector([0.0, 0.0, 0.0])  # 0.01 * (ti.Vector([0.5, 0.5, 0.5]) - ti_pos[v])
        ti_force[v] = ti.Vector([0.0, 0.0, 0.0])
        ti_mass[v] = 1

    for e in range(elem_num):
        ti_elem[e] = ti.Vector([ti_raw_elem[e, 0], ti_raw_elem[e, 1], ti_raw_elem[e, 2], ti_raw_elem[e, 3]])
        c1 = ti_pos[ti_elem[e][0]] - ti_pos[ti_elem[e][3]]
        c2 = ti_pos[ti_elem[e][1]] - ti_pos[ti_elem[e][3]]
        c3 = ti_pos[ti_elem[e][2]] - ti_pos[ti_elem[e][3]]
        Dm = ti.Matrix.cols([c1, c2, c3])
        ti_W[e] = ti.abs((1 / 6) * Dm.determinant())
        ti_Dm_inv[e] = Dm.inverse()
        # for rendering
        ti_face[e * 12 + 0] = ti_elem[e][0]
        ti_face[e * 12 + 1] = ti_elem[e][1]
        ti_face[e * 12 + 2] = ti_elem[e][2]

        ti_face[e * 12 + 3] = ti_elem[e][0]
        ti_face[e * 12 + 4] = ti_elem[e][1]
        ti_face[e * 12 + 5] = ti_elem[e][3]

        ti_face[e * 12 + 6] = ti_elem[e][0]
        ti_face[e * 12 + 7] = ti_elem[e][2]
        ti_face[e * 12 + 8] = ti_elem[e][3]

        ti_face[e * 12 + 9] = ti_elem[e][1]
        ti_face[e * 12 + 10] = ti_elem[e][2]
        ti_face[e * 12 + 11] = ti_elem[e][3]

        ti_lines[e * 12 + 0] = ti_elem[e][0]
        ti_lines[e * 12 + 1] = ti_elem[e][1]

        ti_lines[e * 12 + 2] = ti_elem[e][0]
        ti_lines[e * 12 + 3] = ti_elem[e][2]

        ti_lines[e * 12 + 4] = ti_elem[e][0]
        ti_lines[e * 12 + 5] = ti_elem[e][3]

        ti_lines[e * 12 + 6] = ti_elem[e][1]
        ti_lines[e * 12 + 7] = ti_elem[e][2]

        ti_lines[e * 12 + 8] = ti_elem[e][1]
        ti_lines[e * 12 + 9] = ti_elem[e][3]

        ti_lines[e * 12 + 10] = ti_elem[e][2]
        ti_lines[e * 12 + 11] = ti_elem[e][3]

    # randomize position
    for v in range(vert_num):
        ti_pos[v] += ti.Vector([ti.random() * 0.8 - 0.4, ti.random() * 0.8 - 0.4, ti.random() * 0.8 - 0.4])


@ti.kernel
def step():
    # reset force vector
    for i in range(vert_num):
        ti_force[i] = ti.Vector([0.0, 0.0, 0.0])

    # calculate elastic force
    for e in range(elem_num):
        F = get_deformation_grad(e)

        P = get_piola_corot(F)

        # calculate H matrix in Course note page 29
        H = -ti_W[e] * P @ ti_Dm_inv[e].transpose()

        for i in ti.static(range(3)):
            ti_force[ti_elem[e][i]] += ti.Vector([H[0, i], H[1, i], H[2, i]])
            ti_force[ti_elem[e][3]] += -ti.Vector([H[0, i], H[1, i], H[2, i]])  # for momentum conservation

    for v in range(vert_num):
        # calculate external force (gravity)
        ti_force[v] += gravity * ti_mass[v]
        # update position in semi-explicit way
        ti_vel[v] += dt * ti_force[v] / ti_mass[v]
        ti_pos[v] += dt * ti_vel[v]

        # damp
        ti_vel[v] *= 0.998


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.point_light((3, 3, 3), (1, 1, 1))

    scene.mesh(ti_pos, ti_face, color=(0.5, 0.8, 0.8))
    scene.lines(ti_pos, 5, indices=ti_lines, color=(0, 0, 0))
    canvas.scene(scene)


if __name__ == '__main__':

    init()
    window = ti.ui.Window('Window Title', (1280, 720))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    camera.position(4, 2, 2)
    camera.lookat(1, 0.2, 0)
    camera.up(0, 1, 0)
    camera.fov(55)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)

    while window.running:
        step()
        render()
        window.show()
