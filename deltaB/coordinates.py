#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:14:47 2023

@author: Dean Thomas
"""

import numpy as np

##############################################################################
# Code borrowed from hxform
##############################################################################

def tpad(time, length=7):
    """Pad list with 3 or more elements with zeros.

    Example:
    --------
    >>> from hxform import hxform as hx
    >>> print(hx.tpad([2000,1,1]))                 # [2000, 1, 1, 0, 0, 0, 0]
    >>> print(hx.tpad([2000,1,1], length=4))       # [2000, 1, 1, 0]
    >>> print(hx.tpad([2000,1,1,2,3,4], length=3)) # [2000, 1, 1]
    """

    in_type = type(time)

    time = np.array(time)

    assert(len(time) > 2)

    if len(time.shape) == 1:
        if len(time) > length:
            time = time[0:length]
        else:
            pad = length - len(time)
            time = np.pad(time, (0,pad), 'constant', constant_values=0)
    else:
        if len(time[0]) > length:
            time = time[:,0:length]
        else:
            pad = length - len(time)
            time = np.pad(time, ((0,0),(0,pad)), 'constant', constant_values=0)



    if in_type == np.ndarray:
        return time
    elif in_type == tuple:
        return tuple(map(tuple,time))
    else:
        return list(map(list,time))

def iso2ints(isostr):
    import re
    tmp = re.split("-|:|T|Z", isostr)
    if len(tmp) > 6:
        tmp = tmp[0:5]

    int_list = []
    for str_int in tmp:
        if str_int != "Z" and str_int != '':
            int_list.append(int(str_int))

    return int_list

# transform code modified to use only spacepy.  Original version also accepted
# cxform and geopack_08_dp

def transform(v, time, csys_in, csys_out, ctype_in='car', ctype_out='car', lib='spacepy'):
    """Transfrom between coordinates systems using SpacePy.

    Parameters
    ----------
    v : array-like

        (Nv, 3) float np.array

        np.array of three floats

        list of three floats

        list containing lists of three floats

        list of 3-element np.arrays

    time : array-like
           list of 3+ ints
           list containing lists of 3+ ints
           np.array of 3+ ints
           (Nt, 3) float np.array, where Nt = 1 or Nt = Nv

           The 3+ ints are [year, month, day, [hours, [minutes, [seconds]]]]
           Zeros are used for any missing optional value.

    csys_in : str
              One of MAG, GEI, GEO, GSE, GSM, SM

    csys_out : str
               One of MAG, GEI, GEO, GSE, GSM, SM

    ctype_in : str
               'car' (default) or 'sph'
               For spherical coordinates, `v` should be in r, latitude, longitude,
               with angles in degrees.

    ctype_out : str
               'car' (default) or 'sph'

    lib : str
          only 'spacepy' implemented

    Returns
    -------
    array-like with dimensions matching either `time` (if `Nt` != 1 and `Nv` = 1) or
    `v` (if `Nv` =! 1 and `Nt` = 1). If `Nv` and `Nt` != 1, dimensions are same as `v`.

    Return type will match that of `v`. Note that if a list of 3-element np.arrays are
    passed, execution time will be larger. Use `np.ndarrays` for `v` and `time` for fastest
    execution time.

    Examples
    --------
    >>> from hxform import hxform as hx
    >>> t1 = [2000, 1, 1, 0, 0, 0] # or np.array([2000, 1, 1, 0, 0, 0])
    >>> v1 = [0, 0, 1]             # or np.array([0, 0, 1])
    >>> # All of the following are equivalent and return a list with three floats
    >>> from hxform import hxform as hx
    >>> hx.transform(v1, time1, 'GSM', 'GSE')
    >>> hx.transform(v1, time1, 'GSM', 'GSE', ctype_in='car')
    >>> hx.transform(v1, time1, 'GSM', 'GSE', ctype_in='car', ctype_out='car')

    The following 3 calls return a list with two lists of 3 elements

    1. Transform two vectors at same time t1

        >>> from hxform import hxform as hx
        >>> hx.transform([v1, v1], t1, 'GSM', 'GSE')

    2. Transform one vector at two different times

        >>> from hxform import hxform as hx
        >>> hx.transform(v1, [t1, t1], 'GSM', 'GSE')

    3. Transform two vectors, each at different times

        >>> from hxform import hxform as hx
        >>> hx.transform([v1, v1], [t1, t1], 'GSM', 'GSE')
    """
    assert(lib in ['spacepy'])

    if isinstance(time, str):
        time = iso2ints(time)

    if csys_in == csys_out:
        return v

    v_outertype = type(v)
    v_innertype = type(v[0])
    v = np.array(v, dtype=np.double)
    time = np.array(time, dtype=np.int32)

    if len(time.shape) == 1 and len(v.shape) == 1:
        toret = transform((v,) , (time,) ,
                csys_in, csys_out, ctype_in=ctype_in, ctype_out=ctype_out, lib=lib)[0]
        if issubclass(v_outertype, np.ndarray):
            return toret
        else:
            return v_outertype(toret)

    elif len(time.shape) == 1:
        return transform(v, [time],
                csys_in, csys_out, ctype_in=ctype_in, ctype_out=ctype_out, lib=lib)
    elif len(v.shape) == 1:
        return transform(np.array([v]), time,
                csys_in, csys_out, ctype_in=ctype_in, ctype_out=ctype_out, lib=lib)

    assert(len(time.shape)==2 and len(v.shape)==2)
    assert(time.shape[0]==v.shape[0] or time.shape[0]==1 or v.shape[0]==1)#, "time and v cannot be different lengths")


    if lib == 'spacepy':
        try:
            # SpacePy is not installed when hxform is installed due to
            # frequent install failures and so the default is to not use it.
            import spacepy.coordinates as sc
            from spacepy.time import Ticktock
            import numpy.matlib
        except ImportError as error:
            print(error.__class__.__name__ + ": " + error.message)
        except Exception as exception:
            print(exception, False)
            print(exception.__class__.__name__ + ": " + exception.message)

        if time.shape[0] == 1 and v.shape[0] > 1:
            time = numpy.matlib.repmat(time, v.shape[0], 1)
        if v.shape[0] == 1 and time.shape[0] > 1:
            v = numpy.matlib.repmat(v, time.shape[0], 1)

        cvals = sc.Coords(v, csys_in, ctype_in)

        if len(time.shape) == 1:
            # SpacePy requires time values to be strings with second precision
            t_str = '%04d-%02d-%02dT%02d:%02d:%02d' % tuple(tpad(time, length=6))
        else:
            t_str = []
            for i in range(time.shape[0]):
                t_str.append('%04d-%02d-%02dT%02d:%02d:%02d' % tuple(tpad(time[i,:], length=6)))
            t_str = np.array(t_str)

        cvals.ticks = Ticktock(t_str, 'ISO')
        newcoord = cvals.convert(csys_out, ctype_out)

        vp = newcoord.data

    if issubclass(v_outertype, np.ndarray):
        return vp
    elif issubclass(v_innertype, np.ndarray):
        return v_outertype(vp)
    else:
        return v_outertype(map(v_innertype,vp))

def get_spherical_vector_components(v_cart, x_cart):
    x = x_cart[:,0]
    y = x_cart[:,1]
    z = x_cart[:,2]
    vx = v_cart[:,0]
    vy = v_cart[:,1]
    vz = v_cart[:,2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    L = np.sqrt(x**2 + y**2)

    v_r = (x*vx + y*vy + z*vz)/r
    v_theta = ((x*vx + y*vy)*z - (L**2)*vz)/(r*L)
    v_phi = (-y*vx + x*vy)/L

    return np.column_stack([v_r, v_theta, v_phi])

def get_NED_vector_components(v_cart, x_cart):
    v_sph = get_spherical_vector_components(v_cart, x_cart)
    v_north = -v_sph[:,1]
    v_east  = +v_sph[:,2]
    v_down  = -v_sph[:,0]
    return np.column_stack([v_north, v_east, v_down])

# Changed from lib='geopack_08_dp' to lib='spacepy'
def get_transform_matrix(time, csys_in, csys_out, lib='spacepy'):
    b1 = transform(np.array([1.,0.,0.]), time, csys_in, csys_out, ctype_in='car', ctype_out='car', lib=lib)
    b2 = transform(np.array([0.,1.,0.]), time, csys_in, csys_out, ctype_in='car', ctype_out='car', lib=lib)
    b3 = transform(np.array([0.,0.,1.]), time, csys_in, csys_out, ctype_in='car', ctype_out='car', lib=lib)
    return np.column_stack([b1,b2,b3])
