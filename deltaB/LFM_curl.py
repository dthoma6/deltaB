#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:40:43 2024

@author: Dean Thomas
"""

import numpy as np
import numba

@numba.njit
def norm(x):
    return np.sqrt( x[0]**2 + x[1]**2 + x[2]**2 )

@numba.njit
def lfm_curl( xyz, b, i, j, k ):
    """ Subroutine to determine curl of B using Stokes theorem. Curl b will be
        determined at the midpoint of the cell with diagonal corners
        i,j,k and i+1,j+1,k+1

    Inputs:
        xyz: x, y, z positions of xyz points with shape (3,nI,nJ,nK)
        
        b: vector values at points xyz with shape (3,nI,nJ,nK), curl B will 
            be determined in this subroutine
        
        i,j,k: index of xyz where curl will be calculated.  

    Returns:
        curlb numpy array with 3 components of curl b
    """
    
    xyzshape = xyz.shape
    assert xyzshape == b.shape
    assert i >= 0 and i < xyzshape[1]
    assert j >= 0 and j < xyzshape[2]
    assert k >= 0 and k < xyzshape[3]
    
    # Useful indices
    x_ = 0
    y_ = 1
    z_ = 2
                            
    # Corners of xy plane on one end of cell
    d10 = xyz[:,i+1,j  ,k  ] - xyz[:,i  ,j  ,k  ]
    d21 = xyz[:,i+1,j+1,k  ] - xyz[:,i+1,j  ,k  ]
    d32 = xyz[:,i  ,j+1,k  ] - xyz[:,i+1,j+1,k  ]
    d03 = xyz[:,i  ,j  ,k  ] - xyz[:,i  ,j+1,k  ]
    
    # Line integral around xy plane
    lxy =    (0.5*( b[x_,i+1,j  ,k  ] + b[x_,i  ,j  ,k  ] ) * d10[x_] \
            + 0.5*( b[x_,i+1,j+1,k  ] + b[x_,i+1,j  ,k  ] ) * d21[x_] \
            + 0.5*( b[x_,i  ,j+1,k  ] + b[x_,i+1,j+1,k  ] ) * d32[x_] \
            + 0.5*( b[x_,i  ,j  ,k  ] + b[x_,i  ,j+1,k  ] ) * d03[x_] ) \
            +(0.5*( b[y_,i+1,j  ,k  ] + b[y_,i  ,j  ,k  ] ) * d10[y_] \
            + 0.5*( b[y_,i+1,j+1,k  ] + b[y_,i+1,j  ,k  ] ) * d21[y_] \
            + 0.5*( b[y_,i  ,j+1,k  ] + b[y_,i+1,j+1,k  ] ) * d32[y_] \
            + 0.5*( b[y_,i  ,j  ,k  ] + b[y_,i  ,j+1,k  ] ) * d03[y_] ) \
            +(0.5*( b[z_,i+1,j  ,k  ] + b[z_,i  ,j  ,k  ] ) * d10[z_] \
            + 0.5*( b[z_,i+1,j+1,k  ] + b[z_,i+1,j  ,k  ] ) * d21[z_] \
            + 0.5*( b[z_,i  ,j+1,k  ] + b[z_,i+1,j+1,k  ] ) * d32[z_] \
            + 0.5*( b[z_,i  ,j  ,k  ] + b[z_,i  ,j+1,k  ] ) * d03[z_] )  
                    
    # Area is 1/2 cross product of diagonals
    p0  = xyz[:,i+1,j+1,k  ] - xyz[:,i  ,j  ,k  ]
    p1  = xyz[:,i  ,j+1,k  ] - xyz[:,i+1,j  ,k  ]
    Axy = 0.5*( np.cross(p0,p1) ) # zhat direction

    # Corners of yz plane
    d10 = xyz[:,i  ,j+1,k  ] - xyz[:,i  ,j  ,k  ]
    d21 = xyz[:,i  ,j+1,k+1] - xyz[:,i  ,j+1,k  ]
    d32 = xyz[:,i  ,j  ,k+1] - xyz[:,i  ,j+1,k+1]
    d03 = xyz[:,i  ,j  ,k  ] - xyz[:,i  ,j  ,k+1]
    
    # Line integral around yz plane
    lyz =    (0.5*( b[x_,i  ,j+1,k  ] + b[x_,i  ,j  ,k  ] ) * d10[x_] \
            + 0.5*( b[x_,i  ,j+1,k+1] + b[x_,i  ,j+1,k  ] ) * d21[x_] \
            + 0.5*( b[x_,i  ,j  ,k+1] + b[x_,i  ,j+1,k+1] ) * d32[x_] \
            + 0.5*( b[x_,i  ,j  ,k  ] + b[x_,i  ,j  ,k+1] ) * d03[x_] ) \
            +(0.5*( b[y_,i  ,j+1,k  ] + b[y_,i  ,j  ,k  ] ) * d10[y_] \
            + 0.5*( b[y_,i  ,j+1,k+1] + b[y_,i  ,j+1,k  ] ) * d21[y_] \
            + 0.5*( b[y_,i  ,j  ,k+1] + b[y_,i  ,j+1,k+1] ) * d32[y_] \
            + 0.5*( b[y_,i  ,j  ,k  ] + b[y_,i  ,j  ,k+1] ) * d03[y_] ) \
            +(0.5*( b[z_,i  ,j+1,k  ] + b[z_,i  ,j  ,k  ] ) * d10[z_] \
            + 0.5*( b[z_,i  ,j+1,k+1] + b[z_,i  ,j+1,k  ] ) * d21[z_] \
            + 0.5*( b[z_,i  ,j  ,k+1] + b[z_,i  ,j+1,k+1] ) * d32[z_] \
            + 0.5*( b[z_,i  ,j  ,k  ] + b[z_,i  ,j  ,k+1] ) * d03[z_] )  
    
    # Area is 1/2 cross product of diagonals
    p0  = xyz[:,i  ,j+1,k+1] - xyz[:,i  ,j  ,k  ]
    p1  = xyz[:,i  ,j  ,k+1] - xyz[:,i  ,j+1,k  ]
    Ayz = 0.5*( np.cross(p0,p1) ) # xhat direction
   
    # Corners of xz plane
    d10 = xyz[:,i+1,j  ,k  ] - xyz[:,i  ,j  ,k  ]
    d21 = xyz[:,i+1,j  ,k+1] - xyz[:,i+1,j  ,k  ]
    d32 = xyz[:,i  ,j  ,k+1] - xyz[:,i+1,j  ,k+1]
    d03 = xyz[:,i  ,j  ,k  ] - xyz[:,i  ,j  ,k+1]
    
    # Line integral around xz plane
    lxz =   ((0.5*( b[x_,i+1,j  ,k  ] + b[x_,i  ,j  ,k  ] ) * d10[x_] \
            + 0.5*( b[x_,i+1,j  ,k+1] + b[x_,i+1,j  ,k  ] ) * d21[x_] \
            + 0.5*( b[x_,i  ,j  ,k+1] + b[x_,i+1,j  ,k+1] ) * d32[x_] \
            + 0.5*( b[x_,i  ,j  ,k  ] + b[x_,i  ,j  ,k+1] ) * d03[x_] ) \
            +(0.5*( b[y_,i+1,j  ,k  ] + b[y_,i  ,j  ,k  ] ) * d10[y_] \
            + 0.5*( b[y_,i+1,j  ,k+1] + b[y_,i+1,j  ,k  ] ) * d21[y_] \
            + 0.5*( b[y_,i  ,j  ,k+1] + b[y_,i+1,j  ,k+1] ) * d32[y_] \
            + 0.5*( b[y_,i  ,j  ,k  ] + b[y_,i  ,j  ,k+1] ) * d03[y_] ) \
            +(0.5*( b[z_,i+1,j  ,k  ] + b[z_,i  ,j  ,k  ] ) * d10[z_] \
            + 0.5*( b[z_,i+1,j  ,k+1] + b[z_,i+1,j  ,k  ] ) * d21[z_] \
            + 0.5*( b[z_,i  ,j  ,k+1] + b[z_,i+1,j  ,k+1] ) * d32[z_] \
            + 0.5*( b[z_,i  ,j  ,k  ] + b[z_,i  ,j  ,k+1] ) * d03[z_] )) 
        
    # Area is 1/2 cross product of diagonals
    p0  = xyz[:,i+1,j  ,k+1] - xyz[:,i  ,j  ,k  ]
    p1  = xyz[:,i+1,j  ,k  ] - xyz[:,i  ,j  ,k+1]              
    Axz = 0.5*( np.cross(p1,p0) ) # y-hat direction, hence p1xp0
    
    # We have the line integrals from the three planes. These give use three
    # values of curl b dot normal for the three planes.  Using Stokes Theorem
    # we can convert these to curl b
    
    # First get the areas for the three line integrals
    nAyz = norm(Ayz)
    nAxz = norm(Axz)
    nAxy = norm(Axy)
 
    # Divide the line integrals by the areas to get curl b dot normal
    cb = np.zeros(3)
    cb[0] = lyz/nAyz
    cb[1] = lxz/nAxz
    cb[2] = lxy/nAxy

    # The normals do not necessarily coincide with xhat, yhat, zhat.  So we 
    # need to change to the xhat, yhat, zhat basis
    
    # First get unit vectors normal to the planes
    unityz = Ayz/nAyz
    unitxz = Axz/nAxz
    unitxy = Axy/nAxy
    units = np.vstack((unityz,unitxz,unitxy))
    
    # units @ curlb = cb.  Invert eqn to get curl b in xhat, yhat, zhat basis, 
    # The inverse of units is its transpose in orthogonal vectors, but not when distorted
    # So we use numpy.linalg.inv
    unitsInv = np.linalg.inv( units )
    curlb = unitsInv @ cb
    
    return curlb

if __name__ == "__main__":

    # Test the curl algorithm using a 10x10x10 cube.  We will look at cases where
    # we randomly rotate the cube and cases where we randomly distort the cube.
    
    grid1 = np.zeros([3,2,2,2]) 
    grid1[:,0,0,0] = ( 0, 0, 0)
    grid1[:,1,0,0] = (10, 0, 0)
    grid1[:,0,1,0] = ( 0,10, 0)
    grid1[:,0,0,1] = ( 0, 0,10)
    grid1[:,1,1,0] = (10,10, 0)
    grid1[:,1,0,1] = (10, 0,10)
    grid1[:,0,1,1] = ( 0,10,10)
    grid1[:,1,1,1] = (10,10,10)
    
    # Matrix multiplication used below to rotate cube
    def matmul(A, B):
        """Matrix multiplication of A (3x3) matrix with B (3) vector to give C (3)
        vector, allows numba accelleration
        """
        C = np.zeros(3)
        C[0] = A[0, 0]*B[0] + A[0, 1]*B[1] + A[0, 2]*B[2]
        C[1] = A[1, 0]*B[0] + A[1, 1]*B[1] + A[1, 2]*B[2]
        C[2] = A[2, 0]*B[0] + A[2, 1]*B[1] + A[2, 2]*B[2]
        return C
    
    A = np.zeros([3,3])
    N = 1000
    curlbN = np.zeros([3,N])
    
    import random
    
    # Test that we get the correct curlB when we randomly rotate the cube of points
    for l in range(N):
        
        anum = random.randint(0,90)
        bnum = random.randint(0,90)
        gnum = random.randint(0,90)
        
        alpha = anum * np.pi/180.
        beta  = bnum * np.pi/180.
        gamma = gnum * np.pi/180.
        
        A[0,0] = np.cos(beta) * np.cos(gamma)
        A[0,1] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
        A[0,2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
        A[1,0] = np.cos(beta) * np.sin(gamma)
        A[1,1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
        A[1,2] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
        A[2,0] = - np.sin(beta)
        A[2,1] = np.sin(alpha) * np.cos(beta)
        A[2,2] = np.cos(alpha) * np.cos(beta)
        
        gridn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    gridn[:,i,j,k] = matmul( A, grid1[:,i,j,k] )
    
        Bn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # B = [y, x, z], with curlB = [-1,-1,-1]
                    Bn[0,i,j,k] = gridn[1,i,j,k]
                    Bn[1,i,j,k] = gridn[2,i,j,k]
                    Bn[2,i,j,k] = gridn[0,i,j,k]
        
        curlbN[:,l] = lfm_curl( gridn, Bn, 0,0,0 )
        
    print( 'Random rotation of cube.  Expect [-1,-1,-1]')
    print( 'Avg,Stdev curlBx: ', np.average(curlbN[0,:]), np.std(curlbN[0,:]))
    print( 'Avg,Stdev curlBy: ', np.average(curlbN[1,:]), np.std(curlbN[1,:]))
    print( 'Avg,Stdev curlBz: ', np.average(curlbN[2,:]), np.std(curlbN[2,:]))
    print()
    
    # Test that we get the correct curlB when we randomly rotate the cube of points
    # with a second B field
    for l in range(N):
        
        anum = random.randint(0,90)
        bnum = random.randint(0,90)
        gnum = random.randint(0,90)
        
        alpha = anum * np.pi/180.
        beta  = bnum * np.pi/180.
        gamma = gnum * np.pi/180.
        
        A[0,0] = np.cos(beta) * np.cos(gamma)
        A[0,1] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
        A[0,2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
        A[1,0] = np.cos(beta) * np.sin(gamma)
        A[1,1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
        A[1,2] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
        A[2,0] = - np.sin(beta)
        A[2,1] = np.sin(alpha) * np.cos(beta)
        A[2,2] = np.cos(alpha) * np.cos(beta)
        
        gridn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    gridn[:,i,j,k] = matmul( A, grid1[:,i,j,k] )
    
        Bn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # B = [z, x, y], with curlB = [1, 1, 1]
                    Bn[0,i,j,k] = gridn[2,i,j,k]
                    Bn[1,i,j,k] = gridn[0,i,j,k]
                    Bn[2,i,j,k] = gridn[1,i,j,k]
        
        curlbN[:,l] = lfm_curl( gridn, Bn, 0,0,0 )
        
    print( 'Random rotation of cube.  Expect [1,1,1]')
    print( 'Avg,Stdev curlBx: ', np.average(curlbN[0,:]), np.std(curlbN[0,:]))
    print( 'Avg,Stdev curlBy: ', np.average(curlbN[1,:]), np.std(curlbN[1,:]))
    print( 'Avg,Stdev curlBz: ', np.average(curlbN[2,:]), np.std(curlbN[2,:]))
    print()
    
    # Test that we get the correct curlB when we randomly rotate the cube of points
    # with a third B field
    for l in range(N):
        
        anum = random.randint(0,90)
        bnum = random.randint(0,90)
        gnum = random.randint(0,90)
        
        alpha = anum * np.pi/180.
        beta  = bnum * np.pi/180.
        gamma = gnum * np.pi/180.
        
        A[0,0] = np.cos(beta) * np.cos(gamma)
        A[0,1] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
        A[0,2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
        A[1,0] = np.cos(beta) * np.sin(gamma)
        A[1,1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
        A[1,2] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
        A[2,0] = - np.sin(beta)
        A[2,1] = np.sin(alpha) * np.cos(beta)
        A[2,2] = np.cos(alpha) * np.cos(beta)
        
        gridn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    gridn[:,i,j,k] = matmul( A, grid1[:,i,j,k] )
    
        Bn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # B = [x, y, z], with curlB = [0, 0, 0]
                    Bn[0,i,j,k] = gridn[0,i,j,k]
                    Bn[1,i,j,k] = gridn[1,i,j,k]
                    Bn[2,i,j,k] = gridn[2,i,j,k]
        
        curlbN[:,l] = lfm_curl( gridn, Bn, 0,0,0 )
        
    print( 'Random rotation of cube.  Expect [0,0,0]')
    print( 'Avg,Stdev curlBx: ', np.average(curlbN[0,:]), np.std(curlbN[0,:]))
    print( 'Avg,Stdev curlBy: ', np.average(curlbN[1,:]), np.std(curlbN[1,:]))
    print( 'Avg,Stdev curlBz: ', np.average(curlbN[2,:]), np.std(curlbN[2,:]))
    print()
    

    # Test that we get the correct curlB if we distort the cube by randomly 
    # diplacing the vertices
    for l in range(N):
        
        gridn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    gridn[0,i,j,k] = grid1[0,i,j,k] + random.randint(-100,100)/50.
                    gridn[1,i,j,k] = grid1[1,i,j,k] + random.randint(-100,100)/50.
                    gridn[2,i,j,k] = grid1[2,i,j,k] + random.randint(-100,100)/50.
        
        Bn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # B = [y, z, x], with curlB = [-1,-1,-1]
                    Bn[0,i,j,k] = gridn[1,i,j,k]
                    Bn[1,i,j,k] = gridn[2,i,j,k]
                    Bn[2,i,j,k] = gridn[0,i,j,k]
        
        curlbN[:,l] = lfm_curl( gridn, Bn, 0,0,0 )

    print( 'Random displacement of cube vertices.  Expect [-1,-1,-1]')        
    print( 'Avg,Stdev curlBx: ', np.average(curlbN[0,:]), np.std(curlbN[0,:]))
    print( 'Avg,Stdev curlBy: ', np.average(curlbN[1,:]), np.std(curlbN[1,:]))
    print( 'Avg,Stdev curlBz: ', np.average(curlbN[2,:]), np.std(curlbN[2,:]))
    print()
    
    # Test that we get the correct curlB if we distort the cube by randomly 
    # diplacing the vertices with a second B field
    for l in range(N):
        
        gridn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    gridn[0,i,j,k] = grid1[0,i,j,k] + random.randint(-100,100)/50.
                    gridn[1,i,j,k] = grid1[1,i,j,k] + random.randint(-100,100)/50.
                    gridn[2,i,j,k] = grid1[2,i,j,k] + random.randint(-100,100)/50.
        
        Bn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # B = [z, x, y], with curlB = [ 1, 1, 1]
                    Bn[0,i,j,k] = gridn[2,i,j,k]
                    Bn[1,i,j,k] = gridn[0,i,j,k]
                    Bn[2,i,j,k] = gridn[1,i,j,k]
        
        curlbN[:,l] = lfm_curl( gridn, Bn, 0,0,0 )

    print( 'Random displacement of cube vertices.  Expect [1,1,1]')        
    print( 'Avg,Stdev curlBx: ', np.average(curlbN[0,:]), np.std(curlbN[0,:]))
    print( 'Avg,Stdev curlBy: ', np.average(curlbN[1,:]), np.std(curlbN[1,:]))
    print( 'Avg,Stdev curlBz: ', np.average(curlbN[2,:]), np.std(curlbN[2,:]))
    print()

    # Test that we get the correct curlB if we distort the cube by randomly 
    # diplacing the vertices with a third B field
    for l in range(N):
        
        gridn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    gridn[0,i,j,k] = grid1[0,i,j,k] + random.randint(-100,100)/50.
                    gridn[1,i,j,k] = grid1[1,i,j,k] + random.randint(-100,100)/50.
                    gridn[2,i,j,k] = grid1[2,i,j,k] + random.randint(-100,100)/50.
        
        Bn = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # B = [x, y, z], with curlB = [ 0, 0, 0]
                    Bn[0,i,j,k] = gridn[0,i,j,k]
                    Bn[1,i,j,k] = gridn[1,i,j,k]
                    Bn[2,i,j,k] = gridn[2,i,j,k]
        
        curlbN[:,l] = lfm_curl( gridn, Bn, 0,0,0 )

    print( 'Random displacement of cube vertices.  Expect [0,0,0]')        
    print( 'Avg,Stdev curlBx: ', np.average(curlbN[0,:]), np.std(curlbN[0,:]))
    print( 'Avg,Stdev curlBy: ', np.average(curlbN[1,:]), np.std(curlbN[1,:]))
    print( 'Avg,Stdev curlBz: ', np.average(curlbN[2,:]), np.std(curlbN[2,:]))
                      