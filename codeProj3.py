#! /usr/bin/python3
from math import pi

import numpy as np
na = np.newaxis
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix, block_diag #Added block_diag for Tj_matrix
import mpi4py.MPI as MPI

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import time as t
import argparse

import os
import shutil

def mesh(nx,ny,Lx,Ly):
   i = np.arange(0,nx)[na,:] * np.ones((ny,1), np.int64)
   j = np.arange(0,ny)[:,na] * np.ones((1,nx), np.int64)
   p = np.zeros((2,ny-1,nx-1,3), np.int64)
   q = i+nx*j
   p[:,:,:,0] = q[:-1,:-1]
   p[0,:,:,1] = q[1: ,1: ]
   p[0,:,:,2] = q[1: ,:-1]
   p[1,:,:,1] = q[:-1,1: ]
   p[1,:,:,2] = q[1: ,1: ]
   v = np.concatenate(((Lx/(nx-1)*i)[:,:,na], (Ly/(ny-1)*j)[:,:,na]), axis=2)
   vtx = np.reshape(v, (nx*ny,2)) # The vertices are ordered
   elt = np.reshape(p, (2*(nx-1)*(ny-1),3)) # The triangles are stored s.t. the "low triangles" in a square come all before the "high triangles" 
   return vtx, elt 

def boundary(nx, ny):
    bottom = np.hstack((np.arange(0,nx-1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*(ny-1),nx*ny-1,1)[:,na],
                        np.arange(nx*(ny-1)+1,nx*ny,1)[:,na]))
    left   = np.hstack((np.arange(0,nx*(ny-1),nx)[:,na],
                        np.arange(nx,nx*ny,nx)[:,na]))
    right  = np.hstack((np.arange(nx-1,nx*(ny-1),nx)[:,na],
                        np.arange(2*nx-1,nx*ny,nx)[:,na]))
    return np.vstack((bottom, top, left, right))

# If i give triangles as second input, it computes the area of each triangle and returns a vector of areas
# If i am giving the boundary as second input, it computes the length of each boundary segment and returns a vector of lengths
def get_area(vtx, elt):

    d = np.size(elt, 1)
    if d == 2:
        e = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        areas = la.norm(e, axis=1)
    else:
        e1 = vtx[elt[:, 1], :] - vtx[elt[:, 0], :]
        e2 = vtx[elt[:, 2], :] - vtx[elt[:, 0], :]
        areas = 0.5 * np.abs(e1[:,0] * e2[:,1] - e1[:,1] * e2[:,0])
    return areas

def mass(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    M = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = areas * (1 + (j == k)) / (d*(d+1))
           M += csr_matrix((val, (row, col)), shape=(nv, nv))
    return M

def stiffness(vtx, elt):
    nv = np.size(vtx, 0)
    d = np.size(elt, 1)
    areas = get_area(vtx, elt)
    ne, d = np.shape(elt)
    E = np.empty((ne, d, d-1), dtype=np.float64)
    E[:,0,:] = 0.5 * (vtx[elt[:,1],0:2] - vtx[elt[:,2],0:2])
    E[:,1,:] = 0.5 * (vtx[elt[:,2],0:2] - vtx[elt[:,0],0:2])
    E[:,2,:] = 0.5 * (vtx[elt[:,0],0:2] - vtx[elt[:,1],0:2])
    K = csr_matrix((nv, nv), dtype=np.float64)
    for j in range(d):
        for k in range(d):
           row = elt[:,j]
           col = elt[:,k]
           val = np.sum(E[:,j,:] * E[:,k,:], axis=1) / areas
           K += csr_matrix((val, (row, col)), shape=(nv, nv))
    return K

def point_source(sp, k):    
    def ps(x):
        v = np.zeros(np.size(x,0), float)
        for s in sp:
            v += s[2]*np.exp(-10*k/(2.0*pi) * la.norm(x - s[na,0:2], axis=1))
        return v
    return ps 

def plot_mesh(vtx, elt, val=None, vmin=None, vmax=None, ax = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    trig = mtri.Triangulation(vtx[:,0], vtx[:,1], elt)
    if val is None:
        ax.triplot(trig, **kwargs)
    else:
        ax.tripcolor(trig, val,
                      shading='gouraud',
                      cmap=cm.jet, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_aspect('equal', adjustable='box')

#############################################################################
##                             Local mesh                                  ##
#############################################################################

def local_mesh(nx, ny, Lx, Ly, j, J):
    """
    Input:
        nx, ny: number of points in the x and y directions
        Lx, Ly: lenghts of the domain in the x and y direction
        j: subdivision to be considered
        J: total number of subdivisions of the domain
    Output: 
        vtxj: vertex of the j-th subdivision
        eltj: connectivity of the triangles in the j-th subdivision
    """
    assert j >= 0 and j < J
    vtx, elt = mesh(nx,ny,Lx,Ly)
    endj = (ny - 1) // J
    vtxj = vtx[endj*j*nx:((j + 1)*endj + 1)*nx,:]
    eltj1 = elt[endj*j*(nx-1):((j + 1)*endj)*(nx - 1),:]
    eltj2 = elt[(nx-1)*(ny-1) + endj*j*(nx-1):(nx-1)*(ny-1) + ((j + 1)*endj)*(nx - 1),:] 
    eltj = np.concatenate((eltj1, eltj2), axis=0)
    eltj = eltj - eltj.min() # renumbering of triangles for local mesh
    return vtxj, eltj

def local_boundary(nx, ny, j, J):
    """
    Input:
        nx, ny: number of points in the x and y directions
        j: subdivision to be considered
        J: total number of subdivisions of the domain
    Output: 
        beltj_phys, beltj_artf: connectivity of the physical and artificial boundaries
    """
    assert j >= 0 and j < J
    endj = (ny - 1) // J
    bottom = np.hstack((np.arange(0,nx - 1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*endj,nx*(endj + 1) -1,1)[:,na],
                        np.arange(nx*endj + 1,nx*(endj + 1),1)[:,na]))
    left   = np.hstack((np.arange(0,nx*endj,nx)[:,na],
                        np.arange(nx,nx*(endj + 1),nx)[:,na]))
    right  = np.hstack((np.arange(nx - 1,nx*endj,nx)[:,na],
                        np.arange(2*nx - 1,nx*(endj + 1),nx)[:,na]))
    if j == 0:
        beltj_phys = np.vstack((bottom, left, right))
        beltj_artf = top
    elif j == J - 1:
        beltj_phys = np.vstack((top, left, right))
        beltj_artf = bottom
    else:
        beltj_phys = np.vstack((left, right))
        beltj_artf = np.vstack((bottom, top)) 
    return beltj_phys, beltj_artf

#############################################################################
##                          Restriction matrices                           ##
#############################################################################

def Rj_matrix(nx, ny, j, J): # shape Rj = (nx * (((ny - 1) // J) + 1), nx * ny)
    """
    Input:
        nx, ny: number of points in the x and y directions
        j: subdivision to be considered
        J: total number of subdivisions of the domain
    Output: 
        crs_matrix: the local volume restiction matrix R_j
    """
    assert j >= 0 and j < J
    endj = (ny - 1) // J
    cols = np.arange(endj*j*nx,(endj*(j + 1) + 1)*nx)
    rows = np.arange(len(cols))
    data = np.ones_like(cols)
    return csr_matrix((data, (rows, cols)), shape=(len(rows),nx * ny))

def Bj_matrix(nx, ny, belt_artf, J): # shape Bj = (depends on j, nx * (((ny - 1) // J) + 1))
    """
    Input:
        nx, ny: number of points in the x and y directions
        belt_artf: connectivity of the artificial boundaries
        J: total number of subdivisions of the domain
    Output: 
        crs_matrix: the local boundary restiction matrix B_j
    """
    cols = belt_artf[:,0]
    aux = belt_artf[nx - 2::(nx - 1),1] # Aux takes every value starting from position nx - 2
                                        # in the position multiple of nx - 1
    cols = np.append(cols,aux)
    cols = np.sort(cols)
    rows = np.arange(len(cols))
    data = np.ones_like(cols)
    return csr_matrix((data, (rows,cols)), shape=(len(rows), nx*(((ny - 1) // J) + 1)))

# S has dimention 2*nx*(J-1) since every artificial surface has 2*nx points a part from the first and last
# that have nx points
# Also, ny shouldn't be needed since we are not looking at the y direction expicitly
def Cj_matrix(nx, ny, j ,J): # shape Cj = (depends on j, 2*nx*(J-1))
    """
    Input:
        nx, ny: number of points in the x and y directions
        belt_artf: connectivity of the artificial boundaries
        J: total number of subdivisions of the domain
    Output: 
        crs_matrix: the local boundary restiction matrix C_j
    """
    assert j >= 0 and j < J
    if j == 0:
        cols = np.arange(0, nx)
    elif j == J - 1:
        cols = np.arange((2*j - 1)*nx, 2*j*nx)
    else:
        start = 2*j - 1
        cols = np.arange(start * nx, (start + 2) * nx)
    rows = np.arange(len(cols))
    data = np.ones_like(cols)
    return csr_matrix((data, (rows,cols)), shape=(len(rows), 2*(J - 1)*nx))

#############################################################################
##                           Local matrices                                ##
#############################################################################

def Aj_matrix(vtxj, eltj, beltj_phys, k):
    """
    Input:
        vtxj: vertex of the j-th subdivision
        eltj: connectivity of the triangles in the j-th subdivision
        belt_phys: connectivity of the physical boundaries
        k: wavenumber 
    Output: 
        crs_matrix: the local problem matrix A_j
    """
    Mj = mass(vtxj, eltj) # mass matrix related to Omega_j
    Mbj = mass(vtxj, beltj_phys) # mass matrix related to the physical boundary of Omega_j (does not include artificial boundary)
    Kj = stiffness(vtxj, eltj)
    Aj = csr_matrix(Kj - k**2 * Mj - 1j*k*Mbj)
    return Aj

def Tj_matrix(vtxj, beltj_artf, Bj, k):
    """
    Input:
        vtxj: vertex of the j-th subdivision
        belt_phys: connectivity of the physical boundaries
        Bj: boundary restriction matrix
        k: wavenumber 
    Output: 
        crs_matrix: the local transmssion matrix T_j
    """
    Mbj = mass(vtxj, beltj_artf)
    Tj = csr_matrix(k * (Bj @ Mbj @ Bj.T))
    return Tj

def Sj_factorization(Aj, Tj, Bj):
    """
    Input:
        Aj: local problem matrix 
        Tj: local trasmission matrix
        Bj: local boundary restricition matrix
    Output: 
        spla.splu: LU factorisation of Aj - i Bj^T Tj Bj
    """
    # CSC format is more efficient for direct solvers
    Aj_csc = csc_matrix(Aj)
    Tj_csc = csc_matrix(Tj)
    Bj_csc = csc_matrix(Bj)
    return spla.splu(Aj_csc - 1j * (Bj_csc.T @ Tj_csc @ Bj_csc))


# don't understand why ps, I put sp
def bj_vector(vtxj, eltj, sp, k): # has dimention Omega_j
    """
    Input:
        vtxj: vertex of the j-th subdivision
        eltj: connectivity of the triangles in the j-th subdomain
        sp: ?
        k: wavenumber 
    Output: 
        bj: local right-hand side of the interface problem
    """
    Mj = mass(vtxj, eltj)
    return Mj @ point_source(sp, k)(vtxj)

#############################################################################
##                             Global operators                            ##
#############################################################################


def S_operator(nx, J, global_x, Bj_list, Sj_list, Cj_list, Tj_list, n_values = None, values_displacement = 0, local_js = None):
    """
    Input:
        nx, ny: number of points in the x and y directions
        Lx, Ly: lenghts of the domain in the x and y direction
        j: subdivision to be considered
        J: total number of subdivisions of the domain 
        x: interface unknown 
    Output: 
        y: action of the operator S on x 
    """
    # y = Sx
    if n_values is None:
        n_values = 2*nx*(J-1)
    if local_js is None:
        local_js = J
    
    x = np.zeros(2*nx*(J-1), dtype = np.complex128)
    x[values_displacement:values_displacement + n_values] = global_x.copy()

    y = np.zeros_like(x)
    
    for j in range(local_js): 
        xj = Cj_list[j] @ x
        y_local = 2j * Bj_list[j] @ Sj_list[j].solve(Bj_list[j].T @ Tj_list[j] @ xj)
        y += Cj_list[j].T @ y_local
        y += Cj_list[j].T @ xj
    
    final_y = y[values_displacement:values_displacement + n_values].copy()
    return final_y

def Pi_operator(nx, J, x): # Swaps the artificial boundaries between neighbours, thus x has dimention 2*nx*(J-1)
    """
    Input:
        nx: number of points in the x direction
        J: total number of subdivisions of the domain
        x: interface unknown 
    Output: 
        x: action of the operator Pi on x itself
    """
    x1 = x.copy()
    for i in range (0, (2*J - 1)*nx,2*nx):
        x1[i: i + nx] = x[i + nx: i + 2*nx].copy()
        x1[i + nx: i + 2*nx] = x[i: i + nx].copy()
    return x1
    
def g_vector(nx, J, Sj_list, Cj_list, Bj_list, bj_list, local_js = None):
    """
    Input:
        nx, ny: number of points in the x and y directions
        Lx, Ly: lenghts of the domain in the x and y direction
        sp: source points
        k: wavenumber
        J: total number of subdivisions of the domain
    Output: 
        g: global right-hand side of the skeleton problem
    """
    if local_js is None:
        local_js = J
    dimg = 2*(J-1)*nx
    g = np.zeros(dimg, dtype = np.complex128)

    for j in range(local_js):
        g += Cj_list[j].T @ (Bj_list[j] @ Sj_list[j].solve(bj_list[j]))

    g = -2j * Pi_operator(nx, J, g.copy())
    return g


#############################################################################
##                            Helper functions                             ##
#############################################################################

def create_matrices(nx, ny, Lx, Ly, sp, k, J, displacement = 0, n_rows = None):
    if n_rows is None:
        n_rows = J
    vtxj_list = []
    eltj = []
    Aj_list = []
    Tj_list = []
    Bj_list = []
    Cj_list = []
    Sj_list = []
    Rj_list = []
    bj_list = []
    
    for j in range(displacement, n_rows + displacement):
        vtxj, eltj = local_mesh(nx, ny, Lx, Ly, j, J)
        beltj_phys, beltj_artf = local_boundary(nx, ny, j, J)
        Bj = Bj_matrix(nx, ny, beltj_artf, J)
        Aj = Aj_matrix(vtxj, eltj, beltj_phys, k)
        Tj = Tj_matrix(vtxj, beltj_artf, Bj, k)
        Sj = Sj_factorization(Aj, Tj, Bj)
        Cj = Cj_matrix(nx, ny, j, J)
        Rj = Rj_matrix(nx, ny, j, J)
        bj = bj_vector(vtxj, eltj, sp, k)
        vtxj_list.append(vtxj)
        Aj_list.append(Aj)
        Tj_list.append(Tj)
        Bj_list.append(Bj)
        Cj_list.append(Cj)
        Sj_list.append(Sj)
        Rj_list.append(Rj)
        bj_list.append(bj)

    return vtxj_list, eltj, Aj_list, Tj_list, Bj_list, Cj_list, Sj_list, Rj_list, bj_list

def print_matrices(Tj_list, Bj_list, Cj_list):
    """
    Function to print or visualize the matrices in the provided lists.
    """
    for i, Tj in enumerate(Tj_list):
        print(f"Tj[{i}]:\n{Tj.toarray()}\n")
        plt.figure(figsize=(8, 6))
        plt.spy(Tj, markersize=5)
        plt.title(f'Sparsity Pattern of Tj[{i}]')
        plt.show()

    for i, Bj in enumerate(Bj_list):
        print(f"Bj[{i}]:\n{Bj.toarray()}\n")
        plt.figure(figsize=(8, 6))
        plt.spy(Bj, markersize=5)
        plt.title(f'Sparsity Pattern of Bj[{i}]')
        plt.show()

    for i, Cj in enumerate(Cj_list):
        print(f"Cj[{i}]:\n{Cj.toarray()}\n")
        plt.figure(figsize=(8, 6))
        plt.spy(Cj, markersize=5)
        plt.title(f'Sparsity Pattern of Cj[{i}]')
        plt.show()

def arguments():
    parser = argparse.ArgumentParser(description='Domain decomposition for Helmholtz problem')
    parser.add_argument('--Lx', type=float, default=2, help='Length in x direction of the rectangle')
    parser.add_argument('--Ly', type=float, default=4, help='Length in y direction of the rectangle')
    parser.add_argument('--loc_nx', type=int, default=16, help='Number of vertices for each unitary segment of the rectangle in the x direction')
    parser.add_argument('--loc_ny', type=int, default=16, help='Number of vertices for each unitary segment of the rectangle in the y direction')
    parser.add_argument('--k', type=float, default=16, help='Wavenumber of the problem')
    parser.add_argument('--ns', type=int, default=8, help='Number of point sources')
    parser.add_argument('--J', type=int, default=4, help='Number of subdomains in the y direction (be aware that ny - 1 has to be a multiple of J)')
    parser.add_argument('--tol', type=float, default=1e-12, help='Tolerance for the iterative methods')
    parser.add_argument('--iter_max', type=int, default=10000, help='Maximum number of iterations for the iterative methods')
    parser.add_argument('--w', type=float, default=0.5, help='Relaxation parameter for the fixed point method')
    parser.add_argument('--both', type=bool, default=False, help='Run both the sequential and parallel fixed point methods')
    parser.add_argument('--plot', type=bool, default=False, help='Plot the results')
    parser.add_argument('--save', type=bool, default=True, help='Save the results')
    args = parser.parse_args()
    return args

#############################################################################
##                      Fixed Point Method                                 ##
#############################################################################

def fixed_point(nx, ny, Lx, Ly, sp, k, J, p0, w, tol = 1e-12, iter_max = 100000):
    """
    Input:
        nx, ny: number of points in the x and y directions
        Lx, Ly: lenghts of the domain in the x and y direction
        J: total number of subdivisions of the domain
        p0: initial estimate
        w: relaxation paramter 
        tol: tollerance (automatically fixed to 1e-6 if not specified)
        iter_max: maximum number of iterations (automatically fixed to 1000 if not specified)
    Output: 
        p_next: final iteration
        iter: number of computed iterations
        err: error at the final iteration
    """
    assert w > 0 and w < 1
    residuals = []
    iter = 0


    _, _, _, Tj_list, Bj_list, Cj_list, Sj_list, _, bj_list = create_matrices(nx, ny, Lx, Ly, sp, k, J)
    g = g_vector(nx, J, Sj_list, Cj_list, Bj_list, bj_list)
    assert len(p0) == len(g)
    p_next = np.zeros_like(p0)
    p_0 = p0.copy()
    while (iter < iter_max):
        y = S_operator(nx, J, p_0, Bj_list, Sj_list, Cj_list, Tj_list)
        PiSp0 = Pi_operator(nx,J,y)
        p_next = (1 - w)*p_0 - w*PiSp0  + w*g
        
        residual = np.linalg.norm(p_0 + Pi_operator(nx,J,
        S_operator(nx, J, p_0, Bj_list, Sj_list, Cj_list, Tj_list)) - g, ord=2)
        residuals = np.append(residuals,residual)
        p_0 = p_next.copy()
        iter += 1
        if residual < tol:
            break

    return p_next, iter, residuals

def par_fixed_point(nx, ny, Lx, Ly, sp, k, J, p0, w, tol = 1e-12, iter_max = 100000):
    """
    Input:
        nx, ny: number of points in the x and y directions
        Lx, Ly: lenghts of the domain in the x and y direction
        J: total number of subdivisions of the domain
        p0: initial estimate
        w: relaxation paramter 
        tol: tollerance (automatically fixed to 1e-6 if not specified)
        iter_max: maximum number of iterations (automatically fixed to 1000 if not specified)
    Output: 
        p_next: final iteration
        iter: number of computed iterations
        err: error at the final iteration
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    assert w > 0 and w < 1

    residual = 0
    residuals = []
    iter = 0
    n_p0 = 0

    if rank == 0:
        n_p0 = len(p0)
    n_p0 = comm.bcast(n_p0, root=0)

    j_per_process = [(J // size + (1 if i < J % size else 0)) for i in range(size)] 
    j_displacements = [sum(j_per_process[:i]) for i in range(size)]
    values_per_process = [2*j_per_process[i]*nx - (nx if i == 0 else 0) - (nx if i == size - 1 else 0)  for i in range(size)] 
    values_displacements = [sum(values_per_process[:i]) for i in range(size)]
    
    _, _, _, Tj_list, Bj_list, Cj_list, Sj_list, _, bj_list = create_matrices(nx, ny, Lx, Ly, sp, k, J, j_displacements[rank], j_per_process[rank])
    
    local_g = np.empty(values_per_process[rank], dtype=np.complex128)
    local_p0 = np.empty(values_per_process[rank], dtype=np.complex128)
    local_PiSp0 = np.empty(values_per_process[rank], dtype=np.complex128)
    local_p_next = np.empty(values_per_process[rank], dtype=np.complex128)

    if rank == 0:
        _, _, _, T_list, B_list, C_list, S_list, _, b_list = create_matrices(nx, ny, Lx, Ly, sp, k, J)
        y = np.empty(n_p0,dtype=np.complex128)
        PiSp0 = np.empty(n_p0,dtype=np.complex128)
        p_next = np.empty(n_p0,dtype=np.complex128)
        g = np.empty(n_p0, dtype=np.complex128)
        g = g_vector(nx, J, S_list, C_list, B_list, b_list)
        del _, T_list, B_list, C_list, S_list, b_list
    
    comm.Scatterv([p0 if rank == 0 else None,[2*p for p in values_per_process],[2*v for v in values_displacements],MPI.COMPLEX], local_p0, root=0)
    comm.Scatterv([g if rank == 0 else None, [2*p for p in values_per_process],[2*v for v in values_displacements],MPI.COMPLEX], local_g, root=0)
    
    comm.Gatherv(local_p0, [p0 if rank == 0 else None,[2*p for p in values_per_process],[2*v for v in values_displacements],MPI.COMPLEX], root=0)

    while (iter < iter_max):
        

        local_y = S_operator(nx, J, local_p0, Bj_list, Sj_list, Cj_list, Tj_list, values_per_process[rank], values_displacements[rank], j_per_process[rank])
        #local_y = S_operator(nx, J, local_p0, Bj_list[j_displacements[rank]:j_displacements[rank] + j_per_process[rank]], Sj_list[j_displacements[rank]:j_displacements[rank] + j_per_process[rank]], Cj_list[j_displacements[rank]:j_displacements[rank] + j_per_process[rank]], Tj_list[j_displacements[rank]:j_displacements[rank] + j_per_process[rank]], values_per_process[rank], values_displacements[rank], j_per_process[rank])

        comm.Gatherv([local_y, MPI.COMPLEX], [y if rank == 0 else None ,([2*p for p in values_per_process],[2*v for v in values_displacements]),MPI.COMPLEX])
        
        if rank == 0:
            PiSp0 = Pi_operator(nx,J,y)
        
        comm.Scatterv([PiSp0 if rank == 0 else None,[2*p for p in values_per_process],[2*v for v in values_displacements],MPI.COMPLEX], local_PiSp0, root=0)
        
        local_p_next = (1 - w)*local_p0 - w*local_PiSp0  + w*local_g
        #print("iter:",iter, "rank:",rank,"local_pnext:",local_p_next)
        # We compute the residual at iteration iter - 1 since we already have the value of PiSp0
        local_residual = np.square(np.linalg.norm(local_p0 + local_PiSp0 - local_g, ord=2))
        residual = np.empty(1,np.float64)
        comm.Allreduce(local_residual, residual, op=MPI.SUM)
        residual = np.sqrt(residual)
        if rank == 0:
            residuals = np.append(residuals,residual)
        local_p0 = local_p_next.copy()
        iter += 1
        if residual < tol:
            break
    
    comm.Gatherv(local_p_next,[p_next if rank == 0 else None,[2*p for p in values_per_process],[2*v for v in values_displacements],MPI.COMPLEX],root=0)
    return p_next if rank == 0 else None, iter, residuals
#############################################################################
##                          GMRES Method                                   ##
#############################################################################

def MyGmres(nx, ny, Lx, Ly, sp, k, J, tol = 1e-12):
    """
    Input:
        nx, ny: number of points in the x and y directions
        Lx, Ly: lenghts of the domain in the x and y direction
        J: total number of subdivisions of the domain
        p0: initial estimate
        tol: tollerance (automatically fixed to 1e-12 if not specified)
    Output: 
        y:  solution od the interface problem using GMRES
        Myresiduals: residuals
    """
    def linear_op(nx, J, x, Bj_list, Sj_list, Cj_list, Tj_list):
        y = S_operator(nx, J, x, Bj_list, Sj_list, Cj_list, Tj_list)
        return x + Pi_operator(nx, J, y) 
   

    _, _, _, Tj_list, Bj_list, Cj_list, Sj_list, _, bj_list = create_matrices(nx, ny, Lx, Ly, sp, k, J, 0, J)
    A = spla.LinearOperator((2*nx*(J-1), 2*nx*(J-1)), matvec =lambda x: linear_op(nx, J ,x, Bj_list, Sj_list, Cj_list, Tj_list), dtype = np.complex128)
    g = g_vector(nx, J, Sj_list, Cj_list, Bj_list, bj_list)
    Myresiduals = []

    def callback2(x):
        Myresiduals.append(x)

    y, _ = spla.gmres(A, g, rtol = tol, callback=callback2, callback_type='pr_norm')
    return y, Myresiduals

#############################################################################
##                         Local Solutions                                 ##
#############################################################################

def uj_solution(nx, ny, Lx, Ly, j, J, sp, k, x): # x is the solution of the linear system
    """
    Input:
        nx, ny: number of points in the x and y directions
        Lx, Ly: lenghts of the domain in the x and y direction
        j: subdivision to be considered
        J: total number of subdivisions of the domain
        sp: ?
        k: wavenumber
        x: interface unknown
    Output: 
        u: solution of the system Sj u = (bj + Bj.T @ Tj @ xj)
    """
    vtxj, eltj = local_mesh(nx, ny, Lx, Ly, j, J)
    beltj_phys, beltj_artf = local_boundary(nx, ny, j, J)
    Bj = Bj_matrix(nx, ny, beltj_artf, J)
    Aj = Aj_matrix(vtxj, eltj, beltj_phys, k)
    Tj = Tj_matrix(vtxj, beltj_artf, Bj, k)
    Sj = Sj_factorization(Aj, Tj, Bj)
    Cj = Cj_matrix(nx, ny, j, J)
    bj = bj_vector(vtxj, eltj, sp, k)
    xj = Cj @ x
    return Sj.solve(bj + Bj.T @ Tj @ xj)

#############################################################################
##                                Plots                                    ##
#############################################################################

def plot_residuals(residuals, method_names):
    """
    Input: 
        fixed_point_residuals, gmres_residuals: residuals obtained using the two methods
    Output:
        plt: plot of the concergence of the residuals
    """
    assert len(residuals) == len(method_names)
    
    plt.figure(figsize=(10, 6))
    
    for res, name in zip(residuals, method_names):
        if len(res) == 0:
            raise ValueError("The residuals cannot be of length 0.")
        iterations = np.arange(1, len(res) + 1)
        plt.semilogy(iterations, res, label=name, linestyle='-', linewidth=1.5, marker='x', markersize=4)
    
    plt.xlabel("Number of iterations")
    plt.ylabel("Residual (log scale)")
    plt.title("Convergence of residuals")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

def plot_solutions(vtx, elt, arrays, titles, fig_title = None):
    """
    Plots multiple solutions in a single figure with subplots.

    Parameters:
    vtx: array-like, shape (n_vertices, 2)
        The vertices of the mesh.
    elt: array-like, shape (n_elements, 3)
        The elements of the mesh.
    arrays: list of array-like
        The solutions to plot.
    titles: list of str
        The titles for each subplot.
    """
    if len(arrays) != len(titles):
        raise ValueError("The number of arrays and titles must be the same.")
    
    # Determine the common color range
    vmin = min(np.min(np.real(arr)) for arr in arrays)
    vmax = max(np.max(np.real(arr)) for arr in arrays)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(arrays), figsize=(18, 8), sharex=True, sharey=True)
    fig.suptitle(fig_title, fontsize=16)

    for i, (arr, title) in enumerate(zip(arrays, titles)):
        plot_mesh(vtx, elt, arr, vmin, vmax, ax=axes[i])
        axes[i].set_title(title)
        fig.colorbar(axes[i].collections[0], ax=axes[i])

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_times(method_names, times):
    """
    Plots the times for each method in a bar plot.

    Parameters:
    method_names: list of str
        The names of the methods.
    times: list of float
        The times for each method.
    """
    if len(method_names) != len(times):
        raise ValueError("The number of method names and times must be the same.")
    
    plt.figure(figsize=(10, 6))
    plt.bar(method_names, times, color='skyblue')
    plt.ylabel("Time (seconds)")
    plt.title("Computation Time for Each Method")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels to prevent overlap
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()

def plot_uj_solutions(nx, ny, Lx, Ly, J, uj_solutions):
    """
    Plots each uj solution in a grid of subplots.

    Parameters:
    nx, ny: int
        Number of vertices in the x and y directions.
    Lx, Ly: float
        Length of the domain in the x and y directions.
    sp: list
        List of source points.
    k: float
        Wavenumber.
    J: int
        Number of subdomains in the y direction.
    y_gmres: array-like
        Solution obtained from GMRES.
    """
    fig, axes = plt.subplots(J, 1, figsize=(8, 18), sharex=True, sharey=True)
    fig.suptitle("uj Solutions")

    for j in range(J):
        
        vtxj, eltj = local_mesh(nx, ny, Lx, Ly, j, J)
        uj = uj_solutions[j]
        plot_mesh(vtxj, eltj, uj, ax=axes[j])
        axes[J-1-j].set_title(f"Subdomain {j+1}")
        fig.colorbar(axes[J-1-j].collections[0], ax=axes[J-1-j])

    plt.tight_layout()
    plt.show()
#############################################################################
##                           Save functions                                ##
#############################################################################

def save_plots_and_values(folder_name, vtx, elt, solutions, method_names, times, residuals, values, nx, ny, Lx, Ly, J, ujs):
    """
    Saves all plots and printed values to a specified folder.

    Parameters:
    folder_name: str
        The name of the folder to save the plots and values.
    vtx: array-like
        The vertices of the mesh.
    elt: array-like
        The elements of the mesh.
    solutions: list of array-like
        The solutions to plot.
    method_names: list of str
        The names of the methods.
    times: list of float
        The times for each method.
    residuals: list of array-like
        The residuals for each method.
    values: list of str
        The printed values to save in the file.
    """
    # Check if the folder exists, delete it if it does, and create a new folder
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    # Save the residuals plot
    plt.figure(figsize=(10, 6))
    for res, name in zip(residuals, method_names):
        iterations = np.arange(1, len(res) + 1)
        plt.semilogy(iterations, res, label=name, linestyle='-', linewidth=1.5, marker='x', markersize=4)
    plt.xlabel("Number of iterations")
    plt.ylabel("Residual (log scale)")
    plt.title("Convergence of residuals")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "residuals.png"))
    plt.close()

    # Save the solution plots
    def save_solution_plot(solutions, titles, filename):
        
        vmin = min(np.min(arr) for arr in solutions)
        vmax = max(np.max(arr) for arr in solutions)
        fig, axes = plt.subplots(1, len(solutions), figsize=(18, 8), sharex=True, sharey=True)
        fig.suptitle(filename)
        for i, (arr, title) in enumerate(zip(solutions, titles)):
            plot_mesh(vtx, elt, arr, vmin, vmax, ax=axes[i])
            axes[i].set_title(title)
            fig.colorbar(axes[i].collections[0], ax=axes[i])
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, filename + ".png"))
        plt.close()

    save_solution_plot(np.real(solutions), method_names, "Real part of the solution u")
    save_solution_plot(np.imag(solutions), method_names, "Imaginary part of the solution u")
    save_solution_plot(np.abs(solutions), method_names, "Absolute value of the solution u")

    method_names = np.append("Direct solver", method_names)

    # Save the times plot
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.bar(method_names, times, color='skyblue')
    plt.ylabel("Time (seconds)")
    plt.title("Computation Time for Each Method")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "times.png"))
    plt.close()

    def save_uj_solutions(ujs, filename):
        vmin = min(np.min(arr) for arr in solutions)
        vmax = max(np.max(arr) for arr in solutions)

        fig, axes = plt.subplots(J, 1, figsize=(8, 18), sharex=True, sharey=True)
        fig.suptitle(filename)
        for j in range(J):
            vtxj, eltj = local_mesh(nx, ny, Lx, Ly, j, J)
            uj = ujs[j]
            plot_mesh(vtxj, eltj, uj, ax=axes[j])
            axes[J-1-j].set_title(f"Subdomain {j+1}")
            fig.colorbar(axes[j].collections[0], ax=axes[j])
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, f"{filename}.png"))
        plt.close()
    
    save_uj_solutions(np.real(ujs), "Real part of uj Solutions")
    save_uj_solutions(np.imag(ujs), "Imaginary part of uj Solutions")
    save_uj_solutions(np.abs(ujs), "Absolute value of uj Solutions")

    # Save the printed values to a file
    with open(os.path.join(folder_name, "values.txt"), "w") as f:
        for value in values:
            f.write(value + "\n")    

## Example resolution of model problem

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = arguments()

    both = args.both
    tol = args.tol
    iter_max = args.iter_max
    w = args.w
    

    Lx = args.Lx
    Ly = args.Ly
    nx = 1 + Lx * args.loc_nx
    ny = 1 + Ly * args.loc_ny
    k = args.k
    ns = args.ns
    J = args.J

    plot = args.plot
    save = args.save
    
    assert (ny - 1) % J == 0

    sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]

    if rank == 0:
        vtx, elt = mesh(nx, ny, Lx, Ly)
        belt = boundary(nx, ny)
        M = mass(vtx, elt)
        Mb = mass(vtx, belt) 
        K = stiffness(vtx, elt)
        A = K - k**2 * M - 1j*k*Mb      # matrix of linear system
        b = M @ point_source(sp,k)(vtx) # linear system RHS (source term)
        start_time = t.time()
        x = spla.spsolve(A, b)          # solution of linear system via direct solver
        direct_time = t.time() 
        print("Direct solver time = ", direct_time - start_time)
        # GMRES
        residuals = [] # storage of GMRES residual history

        def callback(x):
            residuals.append(x)

        #y, _ = spla.gmres(A, b, tol=1e-12, callback=callback, callback_type='pr_norm')   
        y, _ = spla.gmres(A, b, rtol=tol, callback=callback, callback_type='pr_norm')
        full_GMRES_time = t.time() 
        print("GMRES time for the full problem = ", full_GMRES_time - direct_time)
        print("Total number of GMRES iterations = ", len(residuals))
        print("Direct vs GMRES error            = ", la.norm(y - x))

        ## Plots
        # plot_mesh(vtx, elt) # slow for fine meshes
        # plt.show()

    initial_guess = None
    if rank == 0:
        initial_guess = np.ones(2*nx*(J-1), dtype=np.complex128)
    if both:
        y_par_fixed, iter_par_fixed, res_par_fixed = par_fixed_point(nx, ny, Lx, Ly, sp, k, J, initial_guess, w, tol, iter_max)
        if rank == 0:
            par_fixed_time = t.time()
            y_seq_fixed, iter_seq_fixed, res_seq_fixed = fixed_point(nx, ny, Lx, Ly, sp, k, J, initial_guess, w, tol, iter_max)
            fixed_time = t.time() 
            print("Parallel fixed point time = ", par_fixed_time - full_GMRES_time)
            print("Total number of parallel fixed point iterations = ", iter_par_fixed)
            print("Sequential fixed point time = ", fixed_time - par_fixed_time)
            print("Total number of sequential fixed point iterations = ", iter_seq_fixed)
    else:
        if size == 1:
            y_fixed, iter, res_fixed = fixed_point(nx, ny, Lx, Ly, sp, k, J, initial_guess, w, tol, iter_max)
            fixed_time = t.time() 
            print("Sequential fixed point time = ", fixed_time - full_GMRES_time)
            print("Total number of sequential fixed point iterations = ", iter)
        else:
            y_fixed, iter, res_fixed = par_fixed_point(nx, ny, Lx, Ly, sp, k, J, initial_guess, w, tol, iter_max)
            if rank == 0:
                fixed_time = t.time() 
                print("Parallel fixed point time = ", fixed_time - full_GMRES_time)
                print("Total number of parallel fixed point iterations = ", iter)

    comm.Barrier()

    if rank == 0:
        y_gmres, My_residuals = MyGmres(nx, ny, Lx, Ly, sp, k, J, tol)
        local_GMRES_time = t.time() 

        print("GMRES time for the local problems = ", local_GMRES_time - fixed_time)
        print("Total number of GMRES iterations for the local problems = ", len(My_residuals))

        if both:
            x_par_fixed = []
            x_seq_fixed = []
            x_gmres = []
            uj_solutions = []
            for j in range(J):
                uj = uj_solution(nx, ny, Lx, Ly, j, J, sp, k, y_gmres)
                x_gmres = np.append(x_gmres,uj)
                x_seq_fixed = np.append(x_seq_fixed,uj_solution(nx, ny, Lx, Ly, j, J, sp, k, y_seq_fixed))
                x_par_fixed = np.append(x_par_fixed,uj_solution(nx, ny, Lx, Ly, j, J, sp, k, y_par_fixed))
                uj_solutions.append(uj)
                if j != J - 1:
                    # take out the last nx points of the previous solution
                    x_gmres = x_gmres[:-nx]
                    x_seq_fixed = x_seq_fixed[:-nx]
                    x_par_fixed = x_par_fixed[:-nx]
                
            print("Direct vs DD_GMRES error for subproblem       = ", la.norm(x_gmres - x))
            print("Sequential Fixed Point vs GMRES error for subproblem = ", la.norm(y_seq_fixed - y_gmres))
            print("Parallel Fixed Point vs GMRES error for subproblem   = ", la.norm(y_par_fixed - y_gmres))
        
            all_residuals = [residuals, My_residuals, res_seq_fixed, res_par_fixed]
            all_methods = ["GMRES solver (global problem)", "GMRES solver", "Sequential Fixed Point solver", "Parallel Fixed Point solver"]
            method_names = all_methods
            solutions = [x, x_gmres, x_seq_fixed, x_par_fixed]
            times = [direct_time - start_time, full_GMRES_time - direct_time, par_fixed_time - full_GMRES_time, fixed_time - par_fixed_time, local_GMRES_time - fixed_time]
            if plot:
                plot_residuals(all_residuals, all_methods)
                plot_solutions(vtx, elt, np.real(solutions), all_methods, "Real part of the solution u")
                plot_solutions(vtx, elt, np.imag(solutions), all_methods, "Imaginary part of the solution u")
                plot_solutions(vtx, elt, np.abs(solutions), all_methods, "Absolute value of the solution u")
                plot_uj_solutions(nx, ny, Lx, Ly, sp, k, J, y_gmres)
                all_methods = np.append("Direct solver", all_methods)
                plot_times(all_methods, times)
            if save:
                values = [
                    f"Direct solver time = {direct_time - start_time}",
                    f"GMRES time for the full problem = {full_GMRES_time - direct_time}",
                    f"Parallel fixed point time = {par_fixed_time - full_GMRES_time}",
                    f"Sequential fixed point time = {fixed_time - par_fixed_time}",
                    f"Total number of parallel fixed point iterations = {iter_par_fixed}",
                    f"GMRES time for the local problems = {local_GMRES_time - fixed_time}",
                    f"Total number of GMRES iterations = {len(residuals)}",
                    f"Total number of parallel fixed point iterations = {iter_par_fixed}",
                    f"Total number of sequential fixed point iterations = {iter_seq_fixed}",
                    f"Total number of GMRES iterations for the local problems = {len(My_residuals)}",
                ]
                title = f"both(Lx,Ly,nx,ny,k,J,size) = ({Lx},{Ly},{nx},{ny},{k},{J},{size})"
                save_plots_and_values(title, vtx, elt, solutions, method_names, times, all_residuals, values, nx, ny, Lx, Ly, J, uj_solutions)

        else:
            x_fixed = []
            x_gmres = []
            uj_solutions = []
            for j in range(J):
                uj = uj_solution(nx, ny, Lx, Ly, j, J, sp, k, y_gmres)
                x_fixed = np.append(x_fixed,uj_solution(nx, ny, Lx, Ly, j, J, sp, k, y_fixed))
                x_gmres = np.append(x_gmres,uj)
                uj_solutions.append(uj)
                if j != J - 1:
                    # take out the last nx points of the previous solution
                    x_fixed = x_fixed[:-nx]
                    x_gmres = x_gmres[:-nx]
            print("Direct vs Fixed Point error for subproblem = ", la.norm(x_fixed - x))

            all_residuals = [residuals, My_residuals, res_fixed]
            if size == 1:
                all_methods = ["GMRES solver (global problem)", "GMRES solver", "Sequential Fixed Point solver"]
            else:
                all_methods = ["GMRES solver (global problem)", "GMRES solver", "Parallel Fixed Point solver"]
            solutions = [x, x_gmres, x_fixed]
            method_names = all_methods
            times = [direct_time - start_time, full_GMRES_time - direct_time, fixed_time - full_GMRES_time, local_GMRES_time - fixed_time]
            if plot:
                plot_residuals(all_residuals, all_methods)
                plot_solutions(vtx, elt, np.real(solutions), all_methods, "Real part of the solution u")
                plot_solutions(vtx, elt, np.imag(solutions), all_methods, "Imaginary part of the solution u")
                plot_solutions(vtx, elt, np.abs(solutions), all_methods, "Absolute value of the solution u")

                all_methods = np.append("Direct solver", all_methods)
                plot_times(all_methods, times)
            if save:
                values = [
                    f"Direct solver time = {direct_time - start_time}",
                    f"GMRES time for the full problem = {full_GMRES_time - direct_time}",
                    f"Sequential fixed point time = {fixed_time - full_GMRES_time}",
                    f"Total number of sequential fixed point iterations = {iter}",
                    f"GMRES time for the local problems = {local_GMRES_time - fixed_time}",
                    f"Total number of GMRES iterations = {len(residuals)}",
                    f"Total number of sequential fixed point iterations = {iter}",
                    f"Total number of GMRES iterations for the local problems = {len(My_residuals)}",
                ]
                title = f"(Lx,Ly,nx,ny,k,J,size) = ({Lx},{Ly},{nx},{ny},{k},{J},{size})"
                save_plots_and_values(title, vtx, elt, solutions, method_names, times, all_residuals, values, nx, ny, Lx, Ly, J, uj_solutions)


if __name__ == "__main__":
    main()