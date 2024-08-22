#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:00:51 2024

@author: Dean Thomas
"""

from numba import njit
# from numba.np.extensions import cross2d
import numpy as np

@njit
def norm(x):
    return np.sqrt( x[0]**2 + x[1]**2 + x[2]**2 )

@njit 
def dot(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit 
def cross(a,b):
    c = np.zeros(3)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

@njit
def calcDivBdV(xyz, b, i, j, k, nI, nJ, nK):
    """ Subroutine for LFM_divBint_b that allows numba accelleration. It  
    calculates divergence of B times dV in the cell with bottom corner point i,j,k  
    using data from lfm file and the Helmholtz decompostion theorem to replace 
    Biot-Savart with a volume integral over the divergence of B.  divB is 
    calculated uing the divergence theorem to convert the volume integral to 
    a surface integral over each cell
    
    Inputs:
        xyz = lfm xyz grid from DataArray
        
        b = b array from DataArray
        
        i,j,k = grid coordinates of point
        
        nI,nJ,nK = number of x,y,z points in grid. Provided to avoid constantly 
            looking them up
        
    Outputs:
        divBdV = divergence of B times dV at cell i,j,k (in GSM coordinates)
    """
    
    assert( i>=0 and i<nI )
    assert( j>=0 and j<nJ )
    assert( k>=0 and k<nK )
    
    # To calculate the surface integral, we determine average b on each of the
    # 6 faces of the cell along with the area of each face.  The surface integral
    # is the sum of dot products (Avg B dot Area).
    
    # Average b on bottom xy face
    bxyb = 0.25*( b[:,i  ,j  ,k  ] + b[:,i+1,j  ,k  ] + b[:,i  ,j+1,k  ] + b[:,i+1,j+1,k  ] )
                    
    # Bottom xy face area is 1/2 cross product of diagonals
    p0 = xyz[:,i+1,j+1,k  ] - xyz[:,i  ,j  ,k  ]
    p1 = xyz[:,i  ,j+1,k  ] - xyz[:,i+1,j  ,k  ]
    Axyb = 0.5*( cross(p1,p0) ) # -zhat
    
    # Average b on bottom yz face
    byzb = 0.25*( b[:,i  ,j  ,k  ] + b[:,i  ,j+1,k  ] + b[:,i  ,j  ,k+1] + b[:,i  ,j+1,k+1]  ) 
    
    # Bottom yz face area is 1/2 cross product of diagonals
    p0 = xyz[:,i  ,j+1,k+1] - xyz[:,i  ,j  ,k  ]
    p1 = xyz[:,i  ,j  ,k+1] - xyz[:,i  ,j+1,k  ]
    Ayzb = 0.5*( cross(p1,p0) ) # -xhat
   
    # Average b on bottom xz face
    bxzb = 0.25*( b[:,i  ,j  ,k  ] + b[:,i+1,j  ,k  ] + b[:,i  ,j  ,k+1] + b[:,i+1,j  ,k+1]  ) 
            
    # Bottom xz face area is 1/2 cross product of diagonals
    p0 = xyz[:,i+1,j  ,k+1] - xyz[:,i  ,j  ,k  ]
    p1 = xyz[:,i+1,j  ,k  ] - xyz[:,i  ,j  ,k+1]
    Axzb = 0.5*( cross(p1,p0) )  # -yhat
    
    # Average b on top xy face
    bxyt = 0.25*( b[:,i  ,j  ,k+1] + b[:,i+1,j  ,k+1] + b[:,i  ,j+1,k+1] + b[:,i+1,j+1,k+1] )

    # Top xy face area is 1/2 cross product of diagonals
    p0 = xyz[:,i+1,j+1,k+1] - xyz[:,i  ,j  ,k+1]
    p1 = xyz[:,i  ,j+1,k+1] - xyz[:,i+1,j  ,k+1]
    Axyt = 0.5*( cross(p0,p1) ) # zhat
    
    # Average b on top yz face
    byzt = 0.25*( b[:,i+1,j  ,k  ] + b[:,i+1,j+1,k  ] + b[:,i+1,j  ,k+1] + b[:,i+1,j+1,k+1]  ) 
    
    # Top yz face area is 1/2 cross product of diagonals
    p0 = xyz[:,i+1,j+1,k+1] - xyz[:,i+1,j  ,k  ]
    p1 = xyz[:,i+1,j  ,k+1] - xyz[:,i+1,j+1,k  ]
    Ayzt = 0.5*( cross(p0,p1) ) # xhat

    # Average b on top xz face
    bxzt = 0.25*( b[:,i  ,j+1,k  ] + b[:,i+1,j+1,k  ] + b[:,i  ,j+1,k+1] + b[:,i+1,j+1,k+1]  ) 
            
    # Top xz face area is 1/2 cross product of diagonals
    p0 = xyz[:,i+1,j+1,k+1] - xyz[:,i  ,j+1,k  ]
    p1 = xyz[:,i+1,j+1,k  ] - xyz[:,i  ,j+1,k+1]
    Axzt = 0.5*( cross(p0,p1) )  # yhat
    
    # Determine divBdV via surface integral     
    divBdV = dot(bxyb, Axyb) + dot(byzb, Ayzb) + dot(bxzb, Axzb) + \
           dot(bxyt, Axyt) + dot(byzt, Ayzt) + dot(bxzt, Axzt)

    return divBdV

@njit
def LFM_divBint_b(XGSM, timeISO, lfm):
    """ Subroutine for calc_ms_surfint_b that allows numba accelleration.  It  
    calculates total B field at point XGSM using data from a lfm file and the
    Helmholtz decompostion theorem to replace Biot-Savart volume integral with  
    a volume integral over divergence of B.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in lfm file
              
        lfm = lfm data
        
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
    """

    assert lfm.DataArray.shape == (len(lfm.varidx), lfm.nI, lfm.nJ, lfm.nK)
    
    # Set up some variables used below
    B = np.zeros(3)
    r = np.zeros(3)
    
    # We use these values throughout the routine, so to avoid muliple lookups
    # we look for them once.
    _x = lfm.varidx['x']
    _y = lfm.varidx['y']
    _z = lfm.varidx['z']
    
    xyz = lfm.DataArray[lfm.varidx[ 'x']:lfm.varidx[ 'z']+1,:,:,:]
    b   = lfm.DataArray[lfm.varidx['bx']:lfm.varidx['bz']+1,:,:,:]
    
    nVar, nI, nJ, nK = lfm.DataArray.shape

    # Iterate thru points in simulation grid, calculating the divergence of B
    # at each point.  Use this in the divB integral from the Helmholtz
    # Decomposition Theorem to determine dB
    for i in range(nI-1):
        for j in range(nJ-1):
            for k in range(nK-1):
                
                # Distance from center of earth to point i,j,k,n
                r0 = np.sqrt(lfm.DataArray[_x,i,j,k]**2 +
                             lfm.DataArray[_y,i,j,k]**2 +
                             lfm.DataArray[_z,i,j,k]**2)
                
                # Only include point if it is outside of rCurrents
                # Data are not valid inside rCurrents
                if r0 >= lfm.rCurrents:
                    # Get divergence of B times dV for integral
                    divBdV = calcDivBdV(xyz, b, i, j, k, nI, nJ, nK)
                    
                    # To calculate the integral, we need the distance from 
                    # point i,j,k to XGSM.  Remember, we're calculating divBdV
                    # for the cell, so our distance is measured from the center
                    # of the cell
                    r[0] = XGSM[0] - 0.5*(lfm.DataArray[_x,i,j,k] + lfm.DataArray[_x,i+1,j+1,k+1])
                    r[1] = XGSM[1] - 0.5*(lfm.DataArray[_y,i,j,k] + lfm.DataArray[_y,i+1,j+1,k+1])
                    r[2] = XGSM[2] - 0.5*(lfm.DataArray[_z,i,j,k] + lfm.DataArray[_z,i+1,j+1,k+1])
                    rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
                                                    
                    ##########################################################
                    # Below we calculate the delta B in each differential volume 
                    # element in the integral.  We want the final result to be 
                    # in nT.
                    # dB = 1/(4pi) divB x r/r^3 dV
                    #    = 1/(4pi) [nT/Re] [Re] / [Re^3] * [Re^3]
                    #    = 1/(4pi) with distances in Re, B in nT
                    #
                    # Note, divBdV replaces divB * measure used with BATSRUS
                    # and OpenGGCM data
                    ##########################################################
                    
                    B = B + divBdV * r / rmag**3 / 4 / np.pi 
      
    return B

if __name__ == "__main__":

    # Test divergence algorithm.  We examine cases where a cube of points
    # is randomly rotated and cases where the cube is randomly distorted.
    # Note, random distortions must be handled carefully.  The volume of a 
    # distorted cube is easy to determine in only limited cases.
    
    from copy import deepcopy
    
    # Default 10x10x10 cube
    grid1 = np.zeros([3,2,2,2]) 
    grid1[:,0,0,0] = (0 , 0, 0)
    grid1[:,1,0,0] = (10, 0, 0)
    grid1[:,0,1,0] = ( 0,10, 0)
    grid1[:,0,0,1] = ( 0, 0,10)
    grid1[:,1,1,0] = (10,10, 0)
    grid1[:,1,0,1] = (10, 0,10)
    grid1[:,0,1,1] = ( 0,10,10)
    grid1[:,1,1,1] = (10,10,10)
    
    # matrix multiplication, used below to rotate grid1 cube    
    def matmul(A, B):
        """Matrix multiplication of A (3x3) matrix with B (3) vector to give C (3)
        vector, allows numba accelleration
        """
        C = np.zeros(3)
        C[0] = A[0, 0]*B[0] + A[0, 1]*B[1] + A[0, 2]*B[2]
        C[1] = A[1, 0]*B[0] + A[1, 1]*B[1] + A[1, 2]*B[2]
        C[2] = A[2, 0]*B[0] + A[2, 1]*B[1] + A[2, 2]*B[2]
        return C
    
    # Data storage
    A = np.zeros([3,3])
    N = 1000
    divBalone = np.zeros(N)
    measures  = np.zeros(N)
    
    import random
    
    # Test that we get the correct divB when we randomly rotate the cube of points
    # Note, calcDivBdV gives us divB * dV, so we divide by the measure to get divB
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
                    # B = [x,y,z] and expect divB = 3
                    Bn[0,i,j,k] = gridn[0,i,j,k]
                    Bn[1,i,j,k] = gridn[1,i,j,k]
                    Bn[2,i,j,k] = gridn[2,i,j,k]
        
        divBalone[l] = calcDivBdV(gridn, Bn, 0,0,0, 2,2,2)/1000 # 10*10*10 cube
        
    print( 'Random rotation of cube.  Expect 3')
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB * dV when we randomly rotate the cube of points
    # with a different B field
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
                    # B = [y,z,x] and expect divB = 0
                    Bn[0,i,j,k] = gridn[1,i,j,k]
                    Bn[1,i,j,k] = gridn[2,i,j,k]
                    Bn[2,i,j,k] = gridn[0,i,j,k]
        
        divBalone[l] = calcDivBdV(gridn, Bn, 0,0,0, 2,2,2)/1000 # 10*10*10 cube
        
    print( 'Random rotation of cube.  Expect 0')
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB * dV when we randomly rotate the cube of points
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
                    # B = [z,x,y] and expect divB = 0
                    Bn[0,i,j,k] = gridn[2,i,j,k]
                    Bn[1,i,j,k] = gridn[0,i,j,k]
                    Bn[2,i,j,k] = gridn[1,i,j,k]
        
        divBalone[l] = calcDivBdV(gridn, Bn, 0,0,0, 2,2,2)/1000 # 10*10*10 cube
        
    print( 'Random rotation of cube.  Expect 0')
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on xy plane
    grid2 = np.zeros([3,2,2,2]) 
    for l in range(N):
        d = random.randint(-100,100)/50.
        
        grid2[:,0,0,0] = (  -d,  -d, 0)
        grid2[:,1,0,0] = (10+d,  -d, 0)
        grid2[:,0,1,0] = (  -d,10+d, 0)
        grid2[:,1,1,0] = (10+d,10+d, 0)
        grid2[:,0,0,1] = ( 0, 0,10)
        grid2[:,1,0,1] = (10, 0,10)
        grid2[:,0,1,1] = ( 0,10,10)
        grid2[:,1,1,1] = (10,10,10)
        
        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[0,i,j,k]
                    B2[1,i,j,k] = grid2[1,i,j,k]
                    B2[2,i,j,k] = grid2[2,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices xy face.  Expect 3')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()

    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on xz plane
    for l in range(N):
        d = random.randint(-100,100)/50.
        
        grid2[:,0,0,0] = (  -d, 0,  -d)
        grid2[:,1,0,0] = (10+d, 0,  -d)
        grid2[:,0,0,1] = (  -d, 0,10+d)
        grid2[:,1,0,1] = (10+d, 0,10+d)
        grid2[:,1,1,0] = (10,10, 0)
        grid2[:,0,1,0] = ( 0,10, 0)
        grid2[:,0,1,1] = ( 0,10,10)
        grid2[:,1,1,1] = (10,10,10)

        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[0,i,j,k]
                    B2[1,i,j,k] = grid2[1,i,j,k]
                    B2[2,i,j,k] = grid2[2,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices xz face.  Expect 3')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on yz plane
    for l in range(N):
        d = random.randint(-100,100)/50.
                
        grid2[:,0,0,0] = ( 0,  -d,  -d)
        grid2[:,0,1,0] = ( 0,10+d,  -d)
        grid2[:,0,0,1] = ( 0,  -d,10+d)
        grid2[:,0,1,1] = ( 0,10+d,10+d)
        grid2[:,1,0,0] = (10, 0, 0)
        grid2[:,1,1,0] = (10,10, 0)
        grid2[:,1,0,1] = (10, 0,10)
        grid2[:,1,1,1] = (10,10,10)
    
        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[0,i,j,k]
                    B2[1,i,j,k] = grid2[1,i,j,k]
                    B2[2,i,j,k] = grid2[2,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices yz face.  Expect 3')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on 2nd xy plane
    for l in range(N):
        d = random.randint(-100,100)/50.
        
        grid2[:,0,0,1] = (  -d,  -d,10)
        grid2[:,1,0,1] = (10+d,  -d,10)
        grid2[:,0,1,1] = (  -d,10+d,10)
        grid2[:,1,1,1] = (10+d,10+d,10)
        grid2[:,0,0,0] = ( 0, 0, 0)
        grid2[:,1,0,0] = (10, 0, 0)
        grid2[:,0,1,0] = ( 0,10, 0)
        grid2[:,1,1,0] = (10,10, 0)
        
        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[0,i,j,k]
                    B2[1,i,j,k] = grid2[1,i,j,k]
                    B2[2,i,j,k] = grid2[2,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices 2nd xy face.  Expect 3')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()

    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on 2nd xz plane
    for l in range(N):
        d = random.randint(-100,100)/50.
        
        grid2[:,0,1,0] = (  -d,10,  -d)
        grid2[:,1,1,0] = (10+d,10,  -d)
        grid2[:,0,1,1] = (  -d,10,10+d)
        grid2[:,1,1,1] = (10+d,10,10+d)
        grid2[:,1,0,0] = (10, 0, 0)
        grid2[:,0,0,0] = ( 0, 0, 0)
        grid2[:,0,0,1] = ( 0, 0,10)
        grid2[:,1,0,1] = (10, 0,10)

        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[0,i,j,k]
                    B2[1,i,j,k] = grid2[1,i,j,k]
                    B2[2,i,j,k] = grid2[2,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices 2nd xz face.  Expect 3')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on 2nd yz plane
    for l in range(N):
        d = random.randint(-100,100)/50.
                
        grid2[:,1,0,0] = (10,  -d,  -d)
        grid2[:,1,1,0] = (10,10+d,  -d)
        grid2[:,1,0,1] = (10,  -d,10+d)
        grid2[:,1,1,1] = (10,10+d,10+d)
        grid2[:,0,0,0] = ( 0, 0, 0)
        grid2[:,0,1,0] = ( 0,10, 0)
        grid2[:,0,0,1] = ( 0, 0,10)
        grid2[:,0,1,1] = ( 0,10,10)
    
        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[0,i,j,k]
                    B2[1,i,j,k] = grid2[1,i,j,k]
                    B2[2,i,j,k] = grid2[2,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices 2nd yz face.  Expect 3')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on xy plane with different B field
    for l in range(N):
        d = random.randint(-100,100)/50.
        
        grid2[:,0,0,0] = (  -d,  -d, 0)
        grid2[:,1,0,0] = (10+d,  -d, 0)
        grid2[:,0,1,0] = (  -d,10+d, 0)
        grid2[:,1,1,0] = (10+d,10+d, 0)
        grid2[:,0,0,1] = ( 0, 0,10)
        grid2[:,1,0,1] = (10, 0,10)
        grid2[:,0,1,1] = ( 0,10,10)
        grid2[:,1,1,1] = (10,10,10)
        
        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[1,i,j,k]
                    B2[1,i,j,k] = grid2[2,i,j,k]
                    B2[2,i,j,k] = grid2[0,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices xy face.  Expect 0')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()

    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on xz plane with different B field
    for l in range(N):
        d = random.randint(-100,100)/50.
        
        grid2[:,0,0,0] = (  -d, 0,  -d)
        grid2[:,1,0,0] = (10+d, 0,  -d)
        grid2[:,0,0,1] = (  -d, 0,10+d)
        grid2[:,1,0,1] = (10+d, 0,10+d)
        grid2[:,1,1,0] = (10,10, 0)
        grid2[:,0,1,0] = ( 0,10, 0)
        grid2[:,0,1,1] = ( 0,10,10)
        grid2[:,1,1,1] = (10,10,10)

        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[2,i,j,k]
                    B2[1,i,j,k] = grid2[0,i,j,k]
                    B2[2,i,j,k] = grid2[1,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices xz face.  Expect 0')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
    # Test that we get the correct divB if we distort the cube by randomly 
    # displacing the vertices on yz plane with different B field
    for l in range(N):
        d = random.randint(-100,100)/50.
                
        grid2[:,0,0,0] = ( 0,  -d,  -d)
        grid2[:,0,1,0] = ( 0,10+d,  -d)
        grid2[:,0,0,1] = ( 0,  -d,10+d)
        grid2[:,0,1,1] = ( 0,10+d,10+d)
        grid2[:,1,0,0] = (10, 0, 0)
        grid2[:,1,1,0] = (10,10, 0)
        grid2[:,1,0,1] = (10, 0,10)
        grid2[:,1,1,1] = (10,10,10)
    
        measure2 = 10/3 * (10**2 + (10+2*d)**2 + 10*(10+2*d))
      
        B2 = np.zeros([3,2,2,2]) 
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    B2[0,i,j,k] = grid2[1,i,j,k]
                    B2[1,i,j,k] = grid2[2,i,j,k]
                    B2[2,i,j,k] = grid2[0,i,j,k]
        
        divBalone[l] = calcDivBdV(grid2, B2, 0,0,0, 2,2,2)/measure2
        
    print( 'Random displacement of cube vertices yz face.  Expect 0')        
    print( 'Avg,Stdev divB: ', np.average(divBalone[:]), np.std(divBalone[:]))
    print()
    
