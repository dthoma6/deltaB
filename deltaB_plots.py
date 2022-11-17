#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:42:06 2022

@author: dean
"""

def nez(time, pos, csys):
  """Unit vectors in geographic north, east, and zenith dirs"""

  from hxform import hxform as hx
  import numpy as np

  # z axis in geographic
  Z = hx.transform(np.array([0, 0, 1]), time, 'GEO', csys, lib='cxform')

  # zenith direction ("up")
  z_geo = pos/np.linalg.norm(pos)

  e_geo = np.cross(z_geo, Z)
  e_geo = e_geo/np.linalg.norm(e_geo)

  n_geo = np.cross(e_geo, z_geo)
  n_geo = n_geo/np.linalg.norm(n_geo)

  return n_geo, e_geo, z_geo


def process_data():
    # Initialize input variables
    filename = "/tmp/3d__var_2_e20190902-041000-000.vtk"
    F_field_str = 'b'
    cell_centers_str = 'cc'
    measure_str = 'measure'
    current_str ='j'
    method_interp = 'nearest'
    rCurrents = 1.8
    (X,Y,Z) = (1.0, 0.0, 0.0) # X,Y,Z point to calculate delta B
    title_suffix = '100'
    
    # Get location of Colaba
    from hxform import hxform as hx
    import numpy as np
    
    # time = '2001-09-02T04:10:00' # Should be 2019
    # pos = (1., 18.907, 72.815)   # Geographic r, lat, long of Colaba
    # from hxform import hxform as hx
    # pos = hx.transform(np.array(pos), time, 'GEO', 'GSM', ctype_in="sph", ctype_out="car", lib='cxform')
    
    # n_geo, e_geo, z_geo = nez(time, pos, "GSM")
    # (X,Y,Z) = n_geo # Point to calculate delta B (dB)
       
    # Verify VTK file exists
    from os.path import exists
    assert exists(filename)
    
    from vtk import vtkUnstructuredGridReader, vtkCellCenters
    from vtk.util import numpy_support as vn
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    
    # Open VTK file which should contain an unstructured cartesian grid
    print('Reading VTK file...')
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    
    # Extract data from VTK file
    data = reader.GetOutput()
    
    # Note, the x,y,z points in the unstructured grid are offset from
    # the center of the cells where the field is defined.
    
    print('Parsing VTK file...')
    
    # The field F at each cell center
    F = vn.vtk_to_numpy(data.GetCellData().GetArray(F_field_str))
    Fx = F[:,0]
    Fy = F[:,1]
    Fz = F[:,2]
    
    # The volume of each cell
    measure = vn.vtk_to_numpy(data.GetCellData().GetArray(measure_str))
    
    # The current at each cell center
    j = vn.vtk_to_numpy(data.GetCellData().GetArray(current_str))
    jx = j[:,0]
    jy = j[:,1]
    jz = j[:,2]
    
    # The density and other variables at each cell center
    rho = vn.vtk_to_numpy(data.GetCellData().GetArray('rho'))
    p = vn.vtk_to_numpy(data.GetCellData().GetArray('p'))
    u = vn.vtk_to_numpy(data.GetCellData().GetArray('u'))
    ux = u[:,0]
    uy = u[:,1]
    uz = u[:,2]
    
    
    # We need the x,y,z locations of the cell centers.
    # If Cell_centers is str, we read them from the file.
    # If Cell_centers is None, we calculate them via VTK.  
    if( isinstance( cell_centers_str, str ) ):
        C = vn.vtk_to_numpy(data.GetCellData().GetArray(cell_centers_str))
        x = C[:,0]
        y = C[:,1]
        z = C[:,2]
    else:
        cellCenters = vtkCellCenters()
        cellCenters.SetInputDataObject(data)
        cellCenters.Update()
        Cpts = vn.vtk_to_numpy(cellCenters.GetOutput().GetPoints().GetData())
        x = Cpts[:,0]
        y = Cpts[:,1]
        z = Cpts[:,2]
        
    # # Create interpolators for unstructured data
    # # Use list(zip(...)) to convert x,y,z's => (x0,y0,z0), (x1,y1,z1), ...
    # xyz = list(zip(x, y, z))
    # if( method_interp == 'linear'):
    #     Fx_interpolate = LinearNDInterpolator( xyz, Fx )
    #     Fy_interpolate = LinearNDInterpolator( xyz, Fy )
    #     Fz_interpolate = LinearNDInterpolator( xyz, Fz )
    # if( method_interp == 'nearest'):
    #     Fx_interpolate = NearestNDInterpolator( xyz, Fx )
    #     Fy_interpolate = NearestNDInterpolator( xyz, Fy )
    #     Fz_interpolate = NearestNDInterpolator( xyz, Fz )
    
    print('Calculating Delta B...')
    
    n = len(x)
    dBx = [None] * n
    dBy = [None] * n
    dBz = [None] * n
    dBmag = [None] * n
    r = [None] * n
    
    for i in range(n):
        r[i] = ((X-x[i])**2+(Y-y[i])**2+(Z-z[i])**2)**(1/2)
        if r[i] >= rCurrents:
            factor = 10000*measure[i]/r[i]**3
            dBx[i] = factor*( jy[i]*(Z-z[i]) - jz[i]*(Y-y[i]) )
            dBy[i] = factor*( jz[i]*(X-x[i]) - jx[i]*(Z-z[i]) )
            dBz[i] = factor*( jx[i]*(Y-y[i]) - jy[i]*(X-x[i]) )
            dBmag[i] = (dBx[i]**2 + dBy[i]**2 + dBz[i]**2)**(1/2)
        else:
            dBx[i]= 0
            dBy[i]= 0
            dBz[i]= 0
            dBmag[i]= 0
            
        if( i%500000 == 0 ): print( i/n )
    
    print('Create dataframe...')
    import pandas as pd
    
    data = list(zip(x,y,z,r,dBx,dBy,dBz,dBmag,jx,jy,jz,Fx,Fy,Fz,rho,p,ux,uy,uz))
    df = pd.DataFrame( data, columns=['x', 'y', 'z', 'r', 'dBx', 'dBy', 'dBz', 'dBmag', 
                                      'jx', 'jy', 'jz', 'Bx', 'By','Bz', 'rho', 'p', 
                                      'ux', 'uy', 'uz'])
    
    # df.to_pickle('dB Colaba rC=15.pkl')
    df.to_pickle('db ' + title_suffix + ' rC=' + str(int(rCurrents*10)) + '.pkl')

def plot_data():
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Set some plot configs
    plt.rcParams["figure.figsize"] = [8,6]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.size'] = 16

    # Read pickle file
    # filename = "dB Colaba rC=18.pkl"
    filename = "dB 100 rc=18.pkl"
    df = pd.read_pickle(filename)
    
    print('Creating 2D plots...')
 
    # Make cuts based on the magnitude of dB in each cell
    df4=df.loc[df['dBmag'] > 0.0001]
    # df3=df.loc[df['dBmag'] > 0.001]
    # df2=df.loc[df['dBmag'] > 0.01]
    # df1=df.loc[df['dBmag'] > 0.1]
    
    # Make a double cut that bounds dB mag between 0.001 and 0.0001
    df34=df4.loc[df4['dBmag'] < 0.001]
    
    # Plot dB mag as a function of range r
    df.plot.scatter(x='r', y='dBmag', logy=True, title='All Points ' + filename)
    # df4.plot.scatter(x='r', y='dBmag', logy=True, title='> 0.0001 ' + filename)
    # df3.plot.scatter(x='r', y='dBmag', logy=True, title='> 0.001 ' + filename)
    # df2.plot.scatter(x='r', y='dBmag', logy=True, title='> 0.01')
    # df1.plot.scatter(x='r', y='dBmag', logy=True, title='> 0.1')
    
    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.  
    # Note, depending on which input file is read, the dB values below
    # the value of rCurrents will be 0
    
    df_r = df.sort_values(by='r', ascending=True)
    df_r['dBx sum'] = df_r['dBx'].cumsum()
    df_r['dBy sum'] = df_r['dBy'].cumsum()
    df_r['dBz sum'] = df_r['dBz'].cumsum()
    df_r['dB sum mag'] = (df_r['dBx sum']**2 + df_r['dBy sum']**2 + df_r['dBz sum']**2)**(1/2)
    
    # Plot the cummulative sum of dB
    df_r.plot.scatter(x='r', y='dB sum mag', xlim=[1,1000], logy=True, 
                        logx=True, title = 'Cumulative dB sum, ' + filename)
    
   
    # Sort the original data by dB magnitude, descending
    # Since the cummulative B stablizes by the time we reach 15 Re
    # keep only those with r < 15 Re.
    
    df_tmp = df.sort_values(by='r', ascending=True)
    df_tmp = df_tmp.loc[df_tmp['r'] <= 15]
    
    df_db = df_tmp
    df_db['dBx sum'] = df_db['dBx'].cumsum()
    df_db['dBy sum'] = df_db['dBy'].cumsum()
    df_db['dBz sum'] = df_db['dBz'].cumsum()
    df_db['dB sum mag'] = (df_db['dBx sum']**2 + df_db['dBy sum']**2 + df_db['dBz sum']**2)**(1/2)
    
    # Plot the cummulative sum of dB
    df_db.plot.scatter(x='r', y='dB sum mag', xlim=[1,15], logy=True, 
                        title = 'Cumulative dB sum, all, ' + filename)
    
    df_db2 = df_tmp.loc[df_tmp['dBmag'] >= 0.01]
    df_db2['dBx sum'] = df_db2['dBx'].cumsum()
    df_db2['dBy sum'] = df_db2['dBy'].cumsum()
    df_db2['dBz sum'] = df_db2['dBz'].cumsum()
    df_db2['dB sum mag'] = (df_db2['dBx sum']**2 + df_db2['dBy sum']**2 + df_db2['dBz sum']**2)**(1/2)
    
    # Plot the cummulative sum of dB
    df_db2.plot.scatter(x='r', y='dB sum mag', xlim=[1,15], logy=True, 
                        title = 'Cumulative dB sum, > 0.01, ' + filename)

    df_db3 = df_tmp.loc[df_tmp['dBmag'] >= 0.001]
    df_db3['dBx sum'] = df_db3['dBx'].cumsum()
    df_db3['dBy sum'] = df_db3['dBy'].cumsum()
    df_db3['dBz sum'] = df_db3['dBz'].cumsum()
    df_db3['dB sum mag'] = (df_db3['dBx sum']**2 + df_db3['dBy sum']**2 + df_db3['dBz sum']**2)**(1/2)
    
    # Plot the cummulative sum of dB
    df_db3.plot.scatter(x='r', y='dB sum mag', xlim=[1,15], logy=True, 
                        title = 'Cumulative dB sum, > 0.001, ' + filename)

    df_db4 = df_tmp.loc[df_tmp['dBmag'] >= 0.0001]
    df_db4['dBx sum'] = df_db4['dBx'].cumsum()
    df_db4['dBy sum'] = df_db4['dBy'].cumsum()
    df_db4['dBz sum'] = df_db4['dBz'].cumsum()
    df_db4['dB sum mag'] = (df_db4['dBx sum']**2 + df_db4['dBy sum']**2 + df_db4['dBz sum']**2)**(1/2)
    
    # Plot the cummulative sum of dB
    df_db4.plot.scatter(x='r', y='dB sum mag', xlim=[1,15], logy=True, 
                        title = 'Cumulative dB sum, > 0.0001, ' + filename)


    df_db5 = df_tmp.loc[df_tmp['dBmag'] >= 0.00001]
    df_db5['dBx sum'] = df_db5['dBx'].cumsum()
    df_db5['dBy sum'] = df_db5['dBy'].cumsum()
    df_db5['dBz sum'] = df_db5['dBz'].cumsum()
    df_db5['dB sum mag'] = (df_db5['dBx sum']**2 + df_db5['dBy sum']**2 + df_db5['dBz sum']**2)**(1/2)
    
    # Plot the cummulativce sum of dB
    df_db5.plot.scatter(x='r', y='dB sum mag', xlim=[1,15], logy=True, 
                        title = 'Cumulative dB sum, > 0.00001, ' + filename)


    df_db23 = df_tmp.loc[df_tmp['dBmag'] >= 0.001]
    df_db23 = df_db23.loc[df_db23['dBmag'] <= 0.01]
    df_db23['dBx sum'] = df_db23['dBx'].cumsum()
    df_db23['dBy sum'] = df_db23['dBy'].cumsum()
    df_db23['dBz sum'] = df_db23['dBz'].cumsum()
    df_db23['dB sum mag'] = (df_db23['dBx sum']**2 + df_db23['dBy sum']**2 + df_db23['dBz sum']**2)**(1/2)
    
    # Plot the cummulativce sum of dB
    df_db23.plot.scatter(x='r', y='dB sum mag', xlim=[1,15], logy=True, 
                        title = 'Cumulative dB sum, > 0.001 & < 0.01, ' + filename)


    print('db : ', df_db['dBmag'].min(), df_db['dB sum mag'].max())
    print('db2: ', df_db2['dBmag'].min(), df_db2['dB sum mag'].max())
    print('db3: ', df_db3['dBmag'].min(), df_db3['dB sum mag'].max())
    print('db4: ', df_db4['dBmag'].min(), df_db4['dB sum mag'].max())
    print('db5: ', df_db5['dBmag'].min(), df_db5['dB sum mag'].max())
    
    # Plot dB cum sum as we increase the smallest values included.
    # That is, start with a line with dB's >=0.01
    # Then with dB's >= 0.001, then >= 0.0001
    # See how many decimal points have to be included to get most of the answer
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes()
    # ax.set_title('> 0.0001 & < 0.001 ' + filename)
    ax.set_xlabel('r')
    ax.set_ylabel('dB sum mag')
    ax.set_xlim(1,15)
    plt.yscale("log")
    
    ax.scatter(df_db['r'], df_db['dB sum mag'], color='blue', marker = ".", label='All')
    ax.scatter(df_db2['r'], df_db2['dB sum mag'], color='red', marker = ".", label='>=0.01')
    ax.scatter(df_db3['r'], df_db3['dB sum mag'], color='green', marker = ".", label='>=0.001')
    ax.scatter(df_db4['r'], df_db4['dB sum mag'], color='black', marker = ".", label='>=0.0001')
    ax.scatter(df_db5['r'], df_db5['dB sum mag'], color='yellow', marker = ".", label='>=0.00001')
    plt.legend()

    # Create 3D plots to show the layers with significant contributions to total
    # B field.
    
    print('Creating 3D plots...')
    
    # Use red/blue to highlight whether the dB component is positive or negative
    colors = {True:'red', False:'blue'}
    
    # Plot dB values > 0.01
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('> 0.01 ' + filename)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)
    ax.set_zlim(-15,15)
    # ax.view_init(azim=135, elev=30)
    
    df_db2['pos'] = df_db2['dBy'] > 0
    ax.scatter(df_db2['x'], df_db2['y'], df_db2['z'], color=df_db2['pos'].map(colors), marker = ".")
    
    # Plot dB values between 0.001 and 0.01
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    ax.set_title('> 0.001 & < 0.01 ' + filename)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)
    ax.set_zlim(-15,15)
    # ax.view_init(azim=135, elev=30)
    
    df_db23['pos'] = df_db23['dBy'] > 0
    ax.scatter(df_db23['x'], df_db23['y'], df_db23['z'], color=df_db23['pos'].map(colors), marker = ".")

def plot_data2():
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Set some plot configs
    plt.rcParams["figure.figsize"] = [8,6]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.size'] = 16

    # Read pickle file
    # filename = "dB Colaba rC=18.pkl"
    filename = "dB 100 rc=18.pkl"
    df = pd.read_pickle(filename)
    
    print('Creating 2D plots...')
     
    # Look at day and night side result
    df_plus =df.loc[df['x'] >= 0]
    df_neg  =df.loc[df['x'] <  0]
    
    # # Plot dB mag as a function of range r
    # df.plot.scatter(x='r', y='dBmag', logy=True, title='All Points ' + filename)
    # df_plus.plot.scatter(x='r', y='dBmag', logy=True, title='x>=0 Points ' + filename)
    # df_neg.plot.scatter(x='r', y='dBmag', logy=True, title='x<0 Points ' + filename)
    
    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.  
    # Note, depending on which input file is read, the dB values below
    # the value of rCurrents will be 0
    
    df_r = df.sort_values(by='r', ascending=True)
    df_r['dBx sum'] = df_r['dBx'].cumsum()
    df_r['dBy sum'] = df_r['dBy'].cumsum()
    df_r['dBz sum'] = df_r['dBz'].cumsum()
    df_r['dB sum mag'] = (df_r['dBx sum']**2 + df_r['dBy sum']**2 + df_r['dBz sum']**2)**(1/2)
    
    df_r_plus = df_plus.sort_values(by='r', ascending=True)
    df_r_plus['dBx sum'] = df_r_plus['dBx'].cumsum()
    df_r_plus['dBy sum'] = df_r_plus['dBy'].cumsum()
    df_r_plus['dBz sum'] = df_r_plus['dBz'].cumsum()
    df_r_plus['dB sum mag'] = (df_r_plus['dBx sum']**2 + df_r_plus['dBy sum']**2 + df_r_plus['dBz sum']**2)**(1/2)
    
    df_r_neg = df_neg.sort_values(by='r', ascending=True)
    df_r_neg['dBx sum'] = df_r_neg['dBx'].cumsum()
    df_r_neg['dBy sum'] = df_r_neg['dBy'].cumsum()
    df_r_neg['dBz sum'] = df_r_neg['dBz'].cumsum()
    df_r_neg['dB sum mag'] = (df_r_neg['dBx sum']**2 + df_r_neg['dBy sum']**2 + df_r_neg['dBz sum']**2)**(1/2)
    
    # # Plot the cummulative sum of dB
    # df_r.plot.scatter(x='r', y='dB sum mag', xlim=[1,1000], logy=True, 
    #                     logx=True, title = 'Cumulative dB sum, ' + filename)
    
    # df_r_plus.plot.scatter(x='r', y='dB sum mag', xlim=[1,1000], logy=True, 
    #                     logx=True, title = 'Cumulative dB sum, x>=0, ' + filename)
    
    # df_r_neg.plot.scatter(x='r', y='dB sum mag', xlim=[1,1000], logy=True, 
    #                     logx=True, title = 'Cumulative dB sum, x<0, ' + filename)
    
   
    # corr = df_r.corr()
    # print(corr)
    
    df_tmp = df.sort_values(by='r', ascending=True)
    df_tmp = df_tmp.loc[df_tmp['r'] <= 15]

    df_tmp['j mag'] = (df_r['jx']**2 + df_r['jy']**2 + df_r['jz']**2)**(1/2)
    df_tmp['u mag'] = (df_r['ux']**2 + df_r['uy']**2 + df_r['uz']**2)**(1/2)
    # df_tmp['rho u'] = df_tmp['rho'] * df_tmp['u mag']

    # corr_tmp = df_tmp.corr()
    # print(corr_tmp)
    
    # import statsmodels.formula.api as sm
    # result = sm.ols(formula="dBmag ~ rho + p", data=df_tmp).fit()
    # print(result.params)
    # print(result.summary())
    
    from pandas import plotting
    plotting.scatter_matrix(df_tmp[['dBmag', 'rho', 'p', 'u mag', 'j mag']]) 
    
if __name__ == "__main__":
    # process_data()
    # plot_data()
    plot_data2()
 