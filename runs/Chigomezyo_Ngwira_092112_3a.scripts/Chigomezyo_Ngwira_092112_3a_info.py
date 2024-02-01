#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:31:23 2023

@author: Dean Thomas
"""

import os.path

###############################################
# Based on magnetopost info structure
###############################################

data_dir = r'/Volumes/PhysicsHDv3'

info = {
        "model": "SWMF",
        "run_name": "Chigomezyo_Ngwira_092112_3a",
        "rCurrents": 1.5,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "dir_run": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a"),
        "dir_plots": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a.plots"),
        "dir_derived": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a.derived"),
        "dir_magnetosphere": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a", "GM_CDF"),
        "dir_ionosphere": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a", "IONO-2D_CDF")
}

###################################################
# Code taken from magnetopost.  Modified to handle this situation.  
# We have cdf files, not .out files.  No _GM_cdf_list file. Nor
# ionosphere file list.
###################################################

def setup(info):
    # import magnetopost as mp

    assert os.path.exists(info["dir_run"]), "dir_run = " + info["dir_run"] + " not found"

    dir_steps = os.path.join(info["dir_derived"], "timeseries", "timesteps")

    if not os.path.exists(info["dir_plots"]):
        os.mkdir(info["dir_plots"])
        # mp.logger.info("Created " + info["dir_plots"])

    if not os.path.exists(info["dir_derived"]):
        os.mkdir(info["dir_derived"])
        # mp.logger.info("Created " + info["dir_derived"])
    
    if not os.path.exists(dir_steps):
        os.makedirs(os.path.join(dir_steps))
        # mp.logger.info("Created " + dir_steps)

    info['files'] = {}

    # if info['file_type'] == 'out':
    if info['file_type'] == 'cdf':

        generate_filelist_txts(info)

        info['files']['magnetosphere'] = {}
        with open(os.path.join(info["dir_derived"], 'magnetosphere_files.txt'), 'r') as f:
            for line in f.readlines():
                items = line.split(' ')
                time = tuple([int(ti) for ti in items[:6]])
                info['files']['magnetosphere'][time] = os.path.join(info['dir_run'], items[-1][:-1])

        info['files']['ionosphere'] = {}
        with open(os.path.join(info["dir_derived"], 'ionosphere_files.txt'), 'r') as f:
            for line in f.readlines():
                items = line.split(' ')
                time = tuple([int(ti) for ti in items[:6]])
                info['files']['ionosphere'][time] = os.path.join(info['dir_run'], items[-1][:-1])

def generate_filelist_txts(info):

    import os
    import re
    import json

    # import magnetopost as mp

    dir_run = info["dir_run"]

    fn = os.path.join(info["dir_derived"], 'run.info.py')
    with open(fn, 'w') as outfile:
        outfile.write(json.dumps(info))

    # mp.logger.info("Wrote {}".format(fn))

    if 'dir_magnetosphere' in info:
        dir_data = os.path.join(dir_run, info['dir_magnetosphere'])
    else:
        dir_data = os.path.join(dir_run, 'GM/IO2')

    magnetosphere_outs = sorted(os.listdir(dir_data))

    fn = os.path.join(info["dir_derived"], 'magnetosphere_files.txt')
    # fn =   os.path.join(info['dir_run'], 'GM_CDF', info['run_name'] + '_GM_cdf_list')

    k = 0
    with open(fn,'w') as fl:
        # regex = r"3d__.*\.out$"
        regex = r"3d__.*\.out.cdf$"
        for fname in magnetosphere_outs:
            if re.search(regex, fname):
                k = k + 1
                assert(fname[:4] == '3d__')
                assert(fname[9] == '_')
                if fname[10] == 'e':
                    Y = int(fname[11:15])
                    M = int(fname[15:17])
                    D = int(fname[17:19])
                    assert(fname[19] == '-')
                    h = int(fname[20:22])
                    m = int(fname[22:24])
                    s = int(fname[24:26])
                    assert(fname[26] == '-')
                    mil = int(fname[27:30])
                    # assert(fname[30:] == '.out')
                    assert(fname[30:] == '.out.cdf')
                    fl.write(f'{Y} {M} {D} {h} {m} {s} {mil} {dir_data}/{fname}\n')
                    # entry = fname + '  Date: ' + str(Y).zfill(4) + '/' + str(M).zfill(2) + '/' + \
                    #     str(D).zfill(2) + ' Time: ' + str(h).zfill(2) +':' + str(m).zfill(2) + \
                    #     ':' + str(s).zfill(2) +'\n'
                    # fl.write(entry)

    # mp.logger.info("Wrote {} file names to {}".format(k, fn))

    if 'dir_ionosphere' in info:
        dir_data = os.path.join(dir_run, info['dir_ionosphere'])
    else:
        dir_data = os.path.join(dir_run, 'IE/ionosphere')

    ionosphere_outs = sorted(os.listdir(dir_data))

    fn = os.path.join(info["dir_derived"], 'ionosphere_files.txt')

    k = 0
    with open(fn,'w') as fl:
        regex = r"null.swmf.i_e.*\.cdf$"

        for fname in ionosphere_outs:
            if re.search(regex, fname):
                k = k + 1

                    # null.swmf.i_e20120723-174500-000.cdf
                    # 012345678901234567890123456789012345

                # assert(fname[:2] == 'it')
                assert(fname[:13] == 'null.swmf.i_e')
                if fname[0] == 'n':
                    Y = int(fname[13:17])
                    M = int(fname[17:19])
                    D = int(fname[19:21])
                    assert(fname[21] == '-')
                    h = int(fname[22:24])
                    m = int(fname[24:26])
                    s = int(fname[26:28])
                    assert(fname[28] == '-')
                    mil = int(fname[29:32])
                    assert(fname[32:] == '.cdf')
                    fl.write(f'{Y} {M} {D} {h} {m} {s} {mil} {dir_data}/{fname}\n')

    # mp.logger.info("Wrote {} file names to {}".format(k, fn))

###################################################
###################################################

