#! /usr/bin/python3
from math import pi

import numpy as np
na = np.newaxis
import numpy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, csc_matrix

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

def local_mesh(nx,ny,Lx,Ly,j,J):
    assert j >= 0 and j < J
    vtx, elt = mesh(nx,ny,Lx,Ly)
    endj = (ny - 1) // J
    vtxj = vtx[endj*j*nx:((j + 1)*endj + 1)*nx,:]
    eltj1 = elt[endj*j*(nx-1):((j + 1)*endj)*(nx - 1),:]
    eltj2 = elt[(nx-1)*(ny-1) + endj*j*(nx-1):(nx-1)*(ny-1) + ((j + 1)*endj)*(nx - 1),:] 
    eltj = np.concatenate((eltj1, eltj2), axis=0) 
    return vtxj, eltj

def local_boundary(nx, ny, j, J):
    endj = (ny - 1) // J
    bottom = np.hstack((np.arange(endj*j*nx,j*endj*nx + nx - 1,1)[:,na],
                        np.arange(endj*j*nx + 1,endj*j*nx + nx,1)[:,na]))
    top    = np.hstack((np.arange(nx*(j + 1)*endj,nx*(j + 1)*endj + nx - 1 ,1)[:,na],
                        np.arange(nx*(j + 1)*endj + 1,nx*(j + 1)*endj + nx,1)[:,na]))
    left   = np.hstack((np.arange(endj*j*nx,endj*(j + 1)*nx,nx)[:,na],
                        np.arange(endj*j*nx + nx,endj*(j + 1)*nx + nx,nx)[:,na]))
    right  = np.hstack((np.arange(endj*j*nx + nx - 1,endj*(j + 1)*nx,nx)[:,na],
                        np.arange(endj*j*nx + 2*nx - 1,(endj*(j + 1) + 1)*nx,nx)[:,na]))
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
    endj = (ny - 1) // J
    cols = np.arange(endj*j*nx,(endj*(j + 1) + 1)*nx)
    rows = np.arange(len(cols))
    data = np.ones_like(cols)
    return csr_matrix((data, (rows, cols)), shape=(len(rows),nx * ny))

def Bj_matrix(nx, ny, belt_artf, J): # shape Bj = (depends on j, nx * (((ny - 1) // J) + 1))
    cols = belt_artf[:,0]
    aux = belt_artf[-1::(nx - 1),1]
    np.append(cols,belt_artf[aux,1])
    np.sort(cols)
    rows = np.arange(len(cols))
    data = np.ones_like(cols)
    return csr_matrix((data, (rows,cols)), shape=(len(rows),nx * (((ny - 1) // J) + 1)))

def Cj_matrix(nx, ny, j ,J): # shape Cj = (depends on j, 2*nx*(J-1))
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
    Mj = mass(vtxj, eltj)
    Mbj = mass(vtxj, beltj_phys)
    Kj = stiffness(vtxj, eltj)
    Aj = Kj - k**2 * Mj - 1j*k*Mbj
    return Aj

def Tj_matrix(vtxj, beltj_artf, Bj, k):
    Mb = mass(vtxj, beltj_artf)
    Tj = k* (Bj @ Mb)
    return Tj

def Sj_factorization(Aj, Tj, Bj):
    return spla.splu(Aj - 1j*(Bj.T @ Tj @ Bj))

def bj_vector(vtxj, eltj, sp, k):
    Mj = mass(vtxj, eltj)
    return Mj @ point_source(sp, k)(vtxj)

#############################################################################
##                             Global operators                            ##
#############################################################################

#not sure why it requires Cj
def S_operator(nx, ny, Lx, Ly, J):
    S = csr_matrix(2*nx*(J-1),2*nx*(J-1), dtype=np.complex128)
    for j in range(J):
        vtxj, eltj = local_mesh(nx, ny, Lx, Ly, j, J)
        beltj_phys, beltj_artf = local_boundary(nx, ny, j, J)
        Bj = Bj_matrix(nx, ny, beltj_artf, J)
        Aj = Aj_matrix(vtxj, eltj, beltj_phys, k)
        Tj = Tj_matrix(vtxj, beltj_artf, Bj, k)
        Sj = Sj_factorization(Aj, Tj, Bj)
        Cj = Cj_matrix(nx, ny, j, J)
        
        if j == 0:
            S[0:nx, 0:nx] = 2j * Bj @ Sj @ (Bj.T @ Tj)
        elif j == J - 1:
            S[(2*j - 1)*nx:2*j*nx, (2*j - 1)*nx:2*j*nx] = 2j * Bj @ Sj @ (Bj.T @ Tj)
        else:
            S[(2*j - 1)*nx:(2*j + 1)*nx, (2*j - 1)*nx:(2*j + 1)*nx] = 2j * Bj @ Sj @ (Bj.T @ Tj)
    
    S += np.eye(2*nx*(J-1))
    return S

def Pi_operator(nx, J):
    cols = np.arange(0, 2*(J-1)*nx)
    for i in range (nx, (2*J - 3)*nx,2*nx):
        aux = cols[i: i + nx]
        cols[i: i + nx] = cols[i + nx: i + 2*nx]
        cols[i + nx: i + 2*nx] = aux
    rows = np.arange(0, 2*(J-1)*nx)
    data = np.ones_like(cols)
    return csr_matrix((data, (rows, cols)), shape=(2*(J-1)*nx, 2*(J-1)*nx))

def b_vector():
    

#############################################################################

## Example resolution of model problem
Lx = 3           # Length in x direction
Ly = 12           # Length in y direction
nx = 1 + Lx * 3 # Number of points in x direction
ny = 1 + Ly * 1 # Number of points in y direction
k = 16           # Wavenumber of the problem
ns = 8           # Number of point sources + random position and weight below
sp = [np.random.rand(3) * [Lx, Ly, 50.0] for _ in np.arange(ns)]
vtx, elt = local_mesh(nx, ny, Lx, Ly, 0 ,4)
belt = local_boundary(nx, ny, 1, 4)
vtx, elt = mesh(nx, ny, Lx, Ly)
belt = boundary(nx, ny)
M = mass(vtx, elt)
Mb = mass(vtx, belt)
K = stiffness(vtx, elt)
A = K - k**2 * M - 1j*k*Mb      # matrix of linear system 
b = M @ point_source(sp,k)(vtx) # linear system RHS (source term)
x = spla.spsolve(A, b)          # solution of linear system via direct solver

# GMRES
residuals = [] # storage of GMRES residual history
def callback(x):
    residuals.append(x)
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