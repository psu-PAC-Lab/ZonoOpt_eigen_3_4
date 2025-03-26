import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import time
import warnings

from ._core import *


def find_vertex(Z, d):
    """Get vertex of Z nearest to direction d"""
    
    # maximize dot product
    c = -Z.get_G().transpose().dot(d)
    if Z.is_0_1_form():
        bounds = [(0, 1) for i in range(Z.get_nG())]
    else:
        bounds = [(-1, 1) for i in range(Z.get_nG())]

    if Z.is_zono():
        res = linprog(c, bounds=bounds)
    elif Z.is_conzono():
        res = linprog(c, A_eq=Z.get_A(), b_eq=Z.get_b(), bounds=bounds)
    else:
        raise ValueError('find_vertex unsupported data type')

    return Z.get_G()*res.x + Z.get_c()

def get_conzono_vertices(Z, t_max=60.0):
    """Get vertices of Z"""

    # init time
    t0 = time.time()

    # make sure Z is not empty
    if Z.is_empty():
        warnings.warn('Z is empty, returning empty list of vertices.')
        return []

    # randomly search directions until a simplex is found
    verts = []
    simplex_found = False
    while not simplex_found and ((time.time()-t0) < t_max):
        
        # random direction
        d = np.random.uniform(low=-1, high=1, size=Z.get_n())
        d = d/np.linalg.norm(d)

        # get vertex
        vd = find_vertex(Z, d)

        # check if vertex is new
        if not any(np.allclose(vd, v) for v in verts):
            verts.append(vd)

        # check if simplex is found
        if len(verts) >= Z.get_n()+1:
            try:
                hull = ConvexHull(verts)
                simplex_found = True
            except:
                pass

    # exit if time limit was reached
    if (time.time()-t0) > t_max:
        warnings.warn('get_vertices time limit reached, terminating early.')
        return verts

    # search for additional vertices along the directions of the facet normals
    converged = False
    while not converged and ((time.time()-t0) < t_max):

        # compute convex hull and centroid
        verts_np_arr = np.array(verts)
        hull = ConvexHull(verts_np_arr)
        centroid = np.mean(verts_np_arr, axis=0)

        # get facet normals
        normals = []
        for simplex in hull.simplices:
            
            # get vertices of facet. each row is a vertex
            V = verts_np_arr[simplex]
            
            # get normal
            Vn = V[-1,:] # last element
            A = V[:-1,:] - Vn # subtract last element from each row
            _, _, Vt = np.linalg.svd(A) # singular value decomp to get null space
            n = Vt[-1,:] # last row of Vt is the null space

            # ensure outward normal
            if np.dot(n, Vn - centroid) < 0:
                n = -n

            normals.append(n)

        # search facet normals for additional vertices
        n_new_verts = 0 # init
        for n in normals:

            # get vertex
            vd = find_vertex(Z, n)

            # check if vertex is new
            if not any(np.allclose(vd, v) for v in verts):
                verts.append(vd)
                n_new_verts += 1

        # check for convergence
        if n_new_verts == 0:
            converged = True

    # throw warning if time limit was reached
    if (time.time()-t0) > t_max:
        warnings.warn('get_vertices time limit reached, terminating early.')

    V = np.array(verts)
    hull = ConvexHull(V)
    V = V[hull.vertices,:]

    return V

def get_vertices(Z, t_max=60.0):
    """Wrapper for calls to get vertices of zono object"""
    
    if Z.is_point():
        return Z.get_c().reshape(1,-1)
    elif Z.is_zono() or Z.is_conzono():
        return get_conzono_vertices(Z, t_max=t_max)
    elif Z.is_hybzono():
        raise ValueError('get_vertices not implemented for HybZono')

def plot(Z, ax=None, **kwargs):
    """Plots Point, Zono, or ConZono object. HybZono not yet implemented. Only 2D plotting
    implemented currently. The **kwargs are those of matplotlib.pyplot.fill"""

    if Z.get_n() < 2 or Z.get_n() > 3:
        raise ValueError("Plot only implemented in 2D or 3D")

    V = get_vertices(Z)

    # 2D
    if Z.get_n() == 2:
        
        # get axes
        if ax is None:
            ax = plt.gca()

        # plot
        if Z.is_point():
            return ax.plot(V[0,0], V[0,1], **kwargs)
        else:
            return ax.fill(V[:,0], V[:,1], **kwargs)

    else: # 3D

        # get axes
        if ax is None:
            raise ValueError("3D plotting requires an Axes3D object")
        
        # plot
        if Z.is_point():
            obj = ax.scatter(V[0,0], V[0,1], V[0,2], **kwargs)
        else:
            hull = ConvexHull(V)
            obj = ax.add_collection3d(Poly3DCollection([[V[vertex] for vertex in face] for face in hull.simplices], **kwargs))

        # adjust scaling
        ax.auto_scale_xyz(V[:,0], V[:,1], V[:,2])
        return obj

        
    