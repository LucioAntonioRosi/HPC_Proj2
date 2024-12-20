#! /usr/bin/python3
from math import pi

import numpy as np
na = np.newaxis
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix, block_diag #Added block_diag for Tj_matrix

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

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

def plot_mesh(vtx, elt, val=None, **kwargs):
    trig = mtri.Triangulation(vtx[:,0], vtx[:,1], elt)
    if val is None:
        plt.triplot(trig, **kwargs)
    else:
        plt.tripcolor(trig, val,
                      shading='gouraud',
                      cmap=cm.jet, **kwargs)
    plt.axis('equal')

#############################################################################
##                             Local mesh                                  ##
#############################################################################

def local_mesh(nx, ny, Lx, Ly, j, J):
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
    # I am keeping the arrays with the 2 in the name because I am not sure if I need them,
    # but probably the right ones are the ones without 2
    assert j >= 0 and j < J
    endj = (ny - 1) // J
    bottom2 = np.hstack((np.arange(endj*j*nx,j*endj*nx + nx - 1,1)[:,na],
                        np.arange(endj*j*nx + 1,endj*j*nx + nx,1)[:,na]))
    bottom = np.hstack((np.arange(0,nx - 1,1)[:,na],
                        np.arange(1,nx,1)[:,na]))
    top2    = np.hstack((np.arange(nx*(j + 1)*endj,nx*(j + 1)*endj + nx - 1 ,1)[:,na],
                        np.arange(nx*(j + 1)*endj + 1,nx*(j + 1)*endj + nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*endj,nx*(endj + 1) -1,1)[:,na],
                        np.arange(nx*endj + 1,nx*(endj + 1),1)[:,na]))
    left2   = np.hstack((np.arange(endj*j*nx,endj*(j + 1)*nx,nx)[:,na],
                        np.arange(endj*j*nx + nx,endj*(j + 1)*nx + nx,nx)[:,na]))
    left   = np.hstack((np.arange(0,nx*endj,nx)[:,na],
                        np.arange(nx,nx*(endj + 1),nx)[:,na]))
    right2  = np.hstack((np.arange(endj*j*nx + nx - 1,endj*(j + 1)*nx,nx)[:,na],
                        np.arange(endj*j*nx + 2*nx - 1,(endj*(j + 1) + 1)*nx,nx)[:,na]))
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
    assert j >= 0 and j < J
    endj = (ny - 1) // J
    cols = np.arange(endj*j*nx,(endj*(j + 1) + 1)*nx)
    rows = np.arange(len(cols))
    data = np.ones_like(cols)
    return csr_matrix((data, (rows, cols)), shape=(len(rows),nx * ny))

def Bj_matrix(nx, ny, belt_artf, J): # shape Bj = (depends on j, nx * (((ny - 1) // J) + 1))
    cols = belt_artf[:,0]
    aux = belt_artf[nx - 2::(nx - 1),1] # Aux takes every value starting from position nx - 2
                                        # in the position multiple of nx - 1
    cols = np.append(cols,aux)
    cols = np.sort(cols)
    rows = np.arange(len(cols))
    data = np.ones_like(cols)
    return csr_matrix((data, (rows,cols)), shape=(len(rows),nx * (((ny - 1) // J) + 1)))

# S has dimention 2*nx*(J-1) since every artificial surface has 2*nx points a part from the first and last
# that have nx points
# Also, ny shouldn't be needed since we are not looking at the y direction expicitly
def Cj_matrix(nx, ny, j ,J): # shape Cj = (depends on j, 2*nx*(J-1))
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
    Mj = mass(vtxj, eltj) # mass matrix related to Omega_j
    Mbj = mass(vtxj, beltj_phys) # mass matrix related to the physical boundary of Omega_j (does not include artificial boundary)
    Kj = stiffness(vtxj, eltj)
    Aj = csr_matrix(Kj - k**2 * Mj - 1j*k*Mbj)
    return Aj

# This should be the correct implementation of Tj_matrix, since Tj is defined on the skeleton of the mesh
# and by defining it as Bj.T @ Mb @ Bj we are giving it dimention of (Omega_j)^2
def Tj_matrix(vtxj, beltj_artf, Bj, k):
    Mb = mass(vtxj, beltj_artf)
    Tj = csr_matrix(k * (Bj @ Mb @ Bj.T))
    return Tj

# I added j and J because the implementation of the matrix is different for the first and last domain
def Tj_matrix_probably_wrong(vtxj, beltj_artf, Bj, k, j, J):
    dim1 = np.size(Bj, 0)
    dim2 = np.size(Bj, 1)
    Tj = csr_matrix((dim2, dim2), dtype=np.float64)
    Mb = mass(vtxj, beltj_artf) # mass matrix related to the artificial boundary of Omega_j

    T = Bj @ Mb @ Bj.T
    if j == 0:
        indexes = np.arange(dim2 - dim1, dim2)
        Mb = csr_matrix(Mb[indexes, :][:, indexes])
    elif j == J - 1:
        indexes = np.arange(dim1)
        Mb = csr_matrix(Mb[indexes, :][:, indexes])
    else:
        dim1 = dim1 // 2
        indexes1 = np.arange(dim1)
        indexes2 = np.arange(dim2 - dim1, dim2)
        Mb1 = csr_matrix(Mb[indexes1, :][:, indexes1])
        Mb2 = csr_matrix(Mb[indexes2, :][:, indexes2])
        Mb = block_diag((Mb1, Mb2), format='csr')

    Tj = csr_matrix(k * (Bj.T @ Mb @ Bj))
    return Tj

def Sj_factorization(Aj, Tj, Bj):
    # CSC format is more efficient for direct solvers
    Aj_csc = csc_matrix(Aj)
    Tj_csc = csc_matrix(Tj)
    Bj_csc = csc_matrix(Bj)
    return spla.splu(Aj_csc - 1j * (Bj_csc.T @ Tj_csc @ Bj_csc))

# don't understand why ps, I put sp
def bj_vector(vtxj, eltj, sp, k):
    Mj = mass(vtxj, eltj)
    return Mj @ point_source(sp, k)(vtxj)

#############################################################################
##                             Global operators                            ##
#############################################################################

def S_operator(nx, ny, Lx, Ly, J, x):
    # y = Sx
    dimS = 2*nx*(J-1)
    S = csr_matrix((dimS,dimS), dtype=np.complex128)
    y = x.copy() # by doing this I am already considering the identity in S
    for j in range(J):
        vtxj, eltj = local_mesh(nx, ny, Lx, Ly, j, J)
        beltj_phys, beltj_artf = local_boundary(nx, ny, j, J)
        Bj = Bj_matrix(nx, ny, beltj_artf, J)
        Aj = Aj_matrix(vtxj, eltj, beltj_phys, k)
        Tj = Tj_matrix(vtxj, beltj_artf, Bj, k)
        Sj = Sj_factorization(Aj, Tj, Bj)
        Cj = Cj_matrix(nx, ny, j, J)
        xj = Cj @ x
        y += Cj.T @  (2j * Bj @ Sj.solve(Bj.T @ Tj @ xj))

    return S

def Pi_operator(nx, J, x):
    for i in range (nx, (2*J - 3)*nx,2*nx):
        aux = x[i: i + nx]
        x[i: i + nx] = x[i + nx: i + 2*nx]
        x[i + nx: i + 2*nx] = aux
    return x

def Pi_operator(nx, J): # This isn't needed probably
    cols = np.arange(0, 2*(J-1)*nx)
    for i in range (nx, (2*J - 3)*nx,2*nx):
        aux = cols[i: i + nx]
        cols[i: i + nx] = cols[i + nx: i + 2*nx]
        cols[i + nx: i + 2*nx] = aux
    rows = np.arange(0, 2*(J-1)*nx)
    data = np.ones_like(cols)
    return csr_matrix((data, (rows, cols)), shape=(2*(J-1)*nx, 2*(J-1)*nx))
    
#############################################################################

## Ly has to be a multiple of J

## Example resolution of model problem
Lx = 4           # Length in x direction
Ly = 4           # Length in y direction
nx = 1 + Lx * 2 # Number of points in x direction
ny = 1 + Ly * 5 # Number of points in y direction
k = 16           # Wavenumber of the problem
ns = 8           # Number of point sources + random position and weight below
j = 1
J = 4
sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]
x  = np.ones(2*nx*(J-1), dtype=np.complex128)
S = S_operator(nx, ny, Lx, Ly, J, x)
vtx, elt = local_mesh(nx, ny, Lx, Ly,j,J)
belt_phys, belt_artf = local_boundary(nx, ny,j,J)
Aj = Aj_matrix(vtx, elt, belt_phys, k)
Bj = Bj_matrix(nx, ny, belt_artf, J)
Cj = Cj_matrix(nx, ny, j, J)
Tj = Tj_matrix(vtx, belt_artf, Bj, k)
vtx, elt = mesh(nx, ny, Lx, Ly)
belt = boundary(nx, ny)
M = mass(vtx, elt)
Mb = mass(vtx, belt) # The dimentions of Mb are the same as the ones of M, because of the function mass
K = stiffness(vtx, elt)
A = K - k**2 * M - 1j*k*Mb      # matrix of linear system 
b = M @ point_source(sp,k)(vtx) # linear system RHS (source term)
x = spla.spsolve(A, b)          # solution of linear system via direct solver

# GMRES
residuals = [] # storage of GMRES residual history
def callback(x):
    residuals.append(x)
#y, _ = spla.gmres(A, b, tol=1e-12, callback=callback, callback_type='pr_norm')   
y, _ = spla.gmres(A, b, rtol=1e-12, callback=callback, callback_type='pr_norm')
print("Total number of GMRES iterations = ", len(residuals))
print("Direct vs GMRES error            = ", la.norm(y - x))

## Plots
# plot_mesh(vtx, elt) # slow for fine meshes
# plt.show()
plot_mesh(vtx, elt, np.real(x))
plt.colorbar()
plt.show()
plot_mesh(vtx, elt, np.abs(x))
plt.colorbar()
plt.show()
plt.semilogy(residuals)
plt.show()