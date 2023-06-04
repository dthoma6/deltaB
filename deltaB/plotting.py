#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 07:57:10 2022

@author: Dean Thomas
"""

####################################################
# Create namedtuple used to draw multiple plots
#   df = dataframe with data to be plotted
#   x, y = names of columns in dataframe to be plotted
#   logx, logy = Boolean, use log scale on x,y axes
#   xlabel, ylabel = labels for axes
#   legend = labels for legend (multiy only)
#   xlim, ylim = limits of axes
#   title = plot title
####################################################

from collections import namedtuple

plotargs = namedtuple('plotargs', ['df',
                                   'x', 'y',
                                   'logx', 'logy',
                                   'xlabel', 'ylabel',
                                   'xlim', 'ylim',
                                   'title'])

plotargs_multiy = namedtuple('plotargs', ['df',
                                   'x', 'y',
                                   'logx', 'logy',
                                   'xlabel', 'ylabel',
                                   'legend',
                                   'xlim', 'ylim',
                                   'title'])

####################################################
# Import python modules
####################################################

from vtkmodules.vtkChartsCore import (
    vtkAxis,
    vtkChart,
    vtkChartXY,
    vtkPlotPoints
)
from vtk import vtkPen
from vtkmodules.vtkCommonColor import vtkNamedColors
# from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkTable
from vtkmodules.vtkRenderingContext2D import (
    vtkContextActor,
    vtkContextScene
)
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    # vtkRenderWindowInteractor,
    vtkRenderer
)
from vtk.util import numpy_support as ns
from vtk import vtkRenderLargeImage, vtkPNGWriter
import logging
import pandas as pd
from .util import create_directory
import os.path

####################################################
# plotting routines
####################################################

def plot_NxM(target, base, suffix, plots, cols=4, rows=2, size = 600, plottype='scatter' ):
    """Plot N columns of M rows of plots (NxM plots total).  

    Inputs:
        target = main folder that will contain subdirectory with plots
        
        base = basename of file used to create file name where plots are saVED.  
            base is derived from name of file with BATSRUS data.
        
        suffix = suffix is used to generate file names and subdirectory.
            Plots are saved in target + suffix directory, target is the 
            overarching directory for all plots.  It contains subdirectories
            (suffix) where different types of plots are saved
            
        plots = a list of plotargs namedtuples WITH the plot parameters
        
        cols = number of columns
        
        rows = number of rows
        
        size = x,y dimensions of each plot
        
        plottype = scatter or line
        
    Outputs:
        None - other than the plots that are saved to file
     """
    assert( len(plots) > 0 and len(plots) <= cols*rows )
    assert( plottype == 'scatter' or plottype == 'line' )

    colors = vtkNamedColors()

    renwin = vtkRenderWindow()
    renwin.SetWindowName('MultiplePlots')
    renwin.OffScreenRenderingOn()

    # Setup the viewports
    size_x = size * cols
    size_y = size * rows
    renwin.SetSize(size_x, size_y)

    # Set up viewports, each plot in plots is drawn in a different viewpot
    # The goal is to have a set of side-by-side plots in rows and columns
    viewports = list()
    for row in range(rows):
        for col in range(cols):
            viewports.append([float(col) / cols,
                              float(rows - (row + 1)) / rows,
                              float(col + 1) / cols,
                              float(rows - row) / rows])    

    # Link the renderers to the viewports and create the charts
    for i in  range( len(plots) ):
        if( plots[i] != None ):
            # Create renderer for each chart
            renderers = vtkRenderer()
            renderers.SetBackground( colors.GetColor3d('White') )
            renderers.SetViewport( viewports[i] )
            renwin.AddRenderer( renderers )
    
            # Create chart along with the scene and actor for it.  THe generic 
            # flow is chart -> scene -> actor -> renderer -> renderwindow
            charts = vtkChartXY()
            scenes = vtkContextScene()
            actors = vtkContextActor()
        
            scenes.AddItem(charts)
            actors.SetScene(scenes)
            renderers.AddActor(actors)
            scenes.SetRenderer(renderers)
            
            # Set up characteristics of x and y axes - color, titles, and
            # that we will fix the range.  The actual range limits will
            # be set below.
            x_axes = charts.GetAxis( vtkAxis.BOTTOM )
            x_axes.GetGridPen().SetColor( colors.GetColor4ub( "LightGrey" ) )
            x_axes.SetTitle( plots[i].xlabel )
            x_axes.SetBehavior(vtkAxis.FIXED)
             
            y_axes = charts.GetAxis(vtkAxis.LEFT)
            y_axes.GetGridPen().SetColor( colors.GetColor4ub( "LightGrey" ) )
            y_axes.SetTitle( plots[i].ylabel )
            y_axes.SetBehavior(vtkAxis.FIXED)
            
            # Set title for chart
            charts.SetTitle( plots[i].title )
    
            # Store the data to be plotted in a vtkTable, the data is taken 
            # from the dataframe identified in plots
            tables = vtkTable()
            
            x_arrays = ns.numpy_to_vtk( (plots[i].df)[ plots[i].x ].to_numpy() )
            x_arrays.SetName( plots[i].xlabel )
            tables.AddColumn( x_arrays )
  
            y_arrays = ns.numpy_to_vtk( (plots[i].df)[ plots[i].y ].to_numpy() )
            y_arrays.SetName( plots[i].ylabel )
            tables.AddColumn( y_arrays )
            
            # As appropriate, use log scales and set the range limits (min/max
            # values) for the axes
            if( plots[i].logx ):
                x_axes.LogScaleOn()
            x_axes.SetMinimum( plots[i].xlim[0] )
            x_axes.SetMaximum( plots[i].xlim[1] )

            if( plots[i].logy ):
                y_axes.LogScaleOn()
            y_axes.SetMinimum( plots[i].ylim[0] )
            y_axes.SetMaximum( plots[i].ylim[1] )

            # Either plot a scatter plot or a line graph
            if plottype == 'scatter':
                points = charts.AddPlot( vtkChart.POINTS )
                points.SetMarkerStyle(vtkPlotPoints.CIRCLE)
                points.SetColor(*colors.GetColor4ub('Black'))
                points.SetMarkerSize(0.1)
            else:  
                points = charts.AddPlot( vtkChart.LINE )
                points.GetPen().SetLineType( vtkPen.SOLID_LINE )
                points.SetColor(*colors.GetColor4ub('Black'))
                points.SetWidth(3.0)
            
            # Specify the data to be plotted
            points.SetInputData(tables, 0, 1)
    
    # Now that the charts are set up, render them
    renwin.Render()

    # Store the charts in a file.
    create_directory(target, suffix +'/')
    logging.info(f'Saving {base} {suffix} plot')
    # filename = target + suffix + '/' + base + '.out.' + suffix + '.png'
    name = base + '.' + suffix + '.png'
    filename = os.path.join( target, suffix, name )
    
    renLgeIm = vtkRenderLargeImage()
    imgWriter = vtkPNGWriter()
    renLgeIm.SetInput( renwin.GetRenderers().GetFirstRenderer() )
    renLgeIm.SetMagnification(1)
    imgWriter.SetInputConnection( renLgeIm.GetOutputPort() )
    imgWriter.SetFileName( filename )
    imgWriter.Write()
    
    return

def plot_NxM_multiy(target, base, suffix, plots, cols=4, rows=2, size=600, plottype='scatter', fontsize=18 ):
    """Plot N columns of M rows of plots (NxM plots total).  In this version,
    multuple y variables can be plotted in a single plot

    Inputs:
        target = main folder that will contain subdirectory with plots
        
        base = basename of file used to create file name for plot.  
            base is derived from name of file with BATSRUS data.
        
        suffix = suffix is used to generate file names and subdirectory.
            Plots are saved in target + suffix directory, target is the 
            overarching directory for all plots.  It contains subdirectories
            (suffix) where different types of plots are saved
            
        plots = a list of plotargs namedtuples that have the plot parameters
        
        cols = number of columns
        
        rows = number of rows
        
        size = x,y dimensions of each plot
        
        plottype = scatter or line
        
        fontsize = size of font for titles and labels
    Outputs:
        None - other than the plots that are saved to file
     """
    assert( len(plots) > 0 and len(plots) <= cols*rows )
    assert( plottype == 'scatter' or plottype == 'line' )

    colors = vtkNamedColors()

    renwin = vtkRenderWindow()
    renwin.SetWindowName('MultiplePlots')
    renwin.OffScreenRenderingOn()

    # Setup the viewports
    size_x = size * cols
    size_y = size * rows
    renwin.SetSize(size_x, size_y)

    # Set up viewports, each plot in plots is drawn in a different viewpot
    # The goal is to have a set of side-by-side plots in rows and columns
    viewports = list()
    for row in range(rows):
        for col in range(cols):
            viewports.append([float(col) / cols,
                              float(rows - (row + 1)) / rows,
                              float(col + 1) / cols,
                              float(rows - row) / rows])    

    # Link the renderers to the viewports and create the charts
    for i in  range( len(plots) ):
        if( plots[i] != None ):
            # Create renderer for each chart
            renderers = vtkRenderer() 
            renderers.SetBackground( colors.GetColor3d('White') )
            renderers.SetViewport( viewports[i] )
            renwin.AddRenderer( renderers )
    
            # Create chart along with the scene and actor for it.  THe generic 
            # flow is chart -> scene -> actor -> renderer -> renderwindow
            charts = vtkChartXY()
            scenes = vtkContextScene()
            actors = vtkContextActor()
        
            scenes.AddItem(charts)
            actors.SetScene(scenes)
            renderers.AddActor(actors)
            scenes.SetRenderer(renderers)
            
            # Set up characteristics of x and y axes - color, titles, and
            # that we will fix the range.  The actual range limits will
            # be set below.
            x_axes = charts.GetAxis( vtkAxis.BOTTOM )
            x_axes.GetGridPen().SetColor( colors.GetColor4ub( "LightGrey" ) )
            x_axes.SetTitle( plots[i].xlabel )
            x_axes.GetTitleProperties().SetFontSize( fontsize )
            x_axes.GetLabelProperties().SetFontSize( fontsize )
            x_axes.SetBehavior(vtkAxis.FIXED)
             
            y_axes = charts.GetAxis(vtkAxis.LEFT)
            y_axes.GetGridPen().SetColor( colors.GetColor4ub( "LightGrey" ) )
            y_axes.SetTitle( plots[i].ylabel )
            y_axes.GetTitleProperties().SetFontSize( fontsize )
            y_axes.GetLabelProperties().SetFontSize( fontsize )
            y_axes.SetBehavior(vtkAxis.FIXED)
            
            # Set title and legend for chart
            charts.SetTitle( plots[i].title )
            charts.GetTitleProperties().SetFontSize( fontsize )
            charts.GetLegend().GetLabelProperties().SetFontSize( fontsize )
            charts.SetShowLegend( True )
    
            # Store the data to be plotted in a vtkTable, the data is taken 
            # from the dataframe identified in plots
            tables = vtkTable() 
            
            x_arrays = ns.numpy_to_vtk( (plots[i].df)[ plots[i].x ].to_numpy() )
            x_arrays.SetName( plots[i].xlabel )
            tables.AddColumn( x_arrays )
  
            for j in range( len(plots[i].y) ):
                y_arrays = ns.numpy_to_vtk( (plots[i].df)[ (plots[i].y)[j] ].to_numpy() )
                y_arrays.SetName( (plots[i].legend)[j] )
                tables.AddColumn( y_arrays )
            
            # As appropriate, use log scales and set the range limits (min/max
            # values) for the axes
            if( plots[i].logx ):
                x_axes.LogScaleOn()
            x_axes.SetMinimum( plots[i].xlim[0] )
            x_axes.SetMaximum( plots[i].xlim[1] )

            if( plots[i].logy ):
                y_axes.LogScaleOn()
            y_axes.SetMinimum( plots[i].ylim[0] )
            y_axes.SetMaximum( plots[i].ylim[1] )

            # Either plot a scatter plot or a line graph
            for k in range( len(plots[i].y) ):
                # Select different symbols/line styles for each y
                style = k%5
                
                # Select appropriate parameters for scatter or line plot
                if plottype == 'scatter':
                    points = charts.AddPlot( vtkChart.POINTS )

                    if style == 0:
                        points.SetMarkerStyle(vtkPlotPoints.CIRCLE)
                    elif style == 1:
                        points.SetMarkerStyle(vtkPlotPoints.CROSS)
                    elif style == 2:
                        points.SetMarkerStyle(vtkPlotPoints.PLUS)
                    elif style == 3:
                        points.SetMarkerStyle(vtkPlotPoints.SQUARE)
                    elif style == 4:
                        points.SetMarkerStyle(vtkPlotPoints.DIAMOND) 
                        
                    points.SetColor(*colors.GetColor4ub('Black'))
                    points.SetMarkerSize(-3.0)
                else:  
                    points = charts.AddPlot( vtkChart.LINE )

                    if style == 0:
                        points.GetPen().SetLineType( vtkPen.SOLID_LINE )
                    elif style == 1:
                        points.GetPen().SetLineType( vtkPen.DASH_LINE )
                    elif style == 2:
                        points.GetPen().SetLineType( vtkPen.DASH_DOT_LINE )
                    elif style == 3:
                        points.GetPen().SetLineType( vtkPen.DASH_DOT_DOT_LINE )
                    elif style == 4:
                        points.GetPen().SetLineType( vtkPen.DASH_DOT_LINE )                  

                    points.SetColor(*colors.GetColor4ub('Black'))
                    points.SetWidth(3.0)
                
                # Specify the data to be plotted
                points.SetInputData(tables, 0, 1+k)
                
    # Now that the charts are set up, render them
    # renwin.Render()

    # Store the charts in a file.
    create_directory(target, suffix +'/')
    logging.info(f'Saving {base} {suffix} plot')
    # filename = target + suffix + '/' + base + '.out.' + suffix + '.png'
    name = base + '.' + suffix + '.png'
    filename = os.path.join( target, suffix, name )
    
    renLgeIm = vtkRenderLargeImage()
    imgWriter = vtkPNGWriter()
    renLgeIm.SetInput( renwin.GetRenderers().GetFirstRenderer() )
    renLgeIm.SetMagnification(1)
    imgWriter.SetInputConnection( renLgeIm.GetOutputPort() )
    imgWriter.SetFileName( filename )
    imgWriter.Write()
    
    return


class pointcloud():
    """Class to convert point cloud to VTK format and to provide options to save
    and display VTK point cloud
    """
    def __init__(self, df, xyz, colorvars):
        

        """Initialize pointcloud class
            
        Inputs:
            df = dataframe containing data to be converted
            
            xyz = list of strings specifiying the dataframe column headings for the
                x,y,z point coordinates
                
            colorvars = list of strings specifying the dataframe column headings
                for variables that can be used to color the point cloud
                
        Outputs:
            None
        """
        logging.info('Initializing point cloud class') 

        # Check inputs
        assert( isinstance( df, pd.DataFrame ) )
        assert( isinstance( xyz, list ) )
        assert( isinstance( colorvars, list) )
        
        # Store instance data
        self.df = df
        self.xyz = xyz
        self.colorvars = colorvars
                        
        # Initialize storage for point cloud in VTK polydata
        self.vtk_polydata = None
        return
    
    def convert_to_vtk(self):
        """Convert point cloud to VTK format.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPoints, vtkCellArray, vtkPolyData

        logging.info('Converting point cloud data to VTK') 
 
        # Make sure that we have data to convert
        if( not isinstance( self.df, pd.DataFrame ) ):
            logging.info('Point cloud dataframe must be specified')
            return -1
       
        # We need to loop through all n points
        n = len(self.df)
        
        # Convert xyz points to VTK format
        vtk_points = vtkPoints()
        vtk_cellarray = vtkCellArray()
         
        vtk_points.SetNumberOfPoints(n)
        for i in range(n):
            vtk_points.SetPoint(i, (self.df[self.xyz[0]]).iloc[i], 
                                (self.df[self.xyz[1]]).iloc[i], 
                                (self.df[self.xyz[2]]).iloc[i] )
            vtk_cellarray.InsertNextCell(1)
            vtk_cellarray.InsertCellPoint(i)

        # Put points into polydata
        if( self.vtk_polydata != None ): 
            del self.vtk_polydata
        self.vtk_polydata = vtkPolyData()
        self.vtk_polydata.SetPoints(vtk_points)
        self.vtk_polydata.SetVerts(vtk_cellarray)

        # Add color variables to polydata        
        for j in self.colorvars :
            color_array = ns.numpy_to_vtk( self.df[ j ].to_numpy() )
            color_array.SetName( j )
            self.vtk_polydata.GetPointData().AddArray( color_array )

        self.vtk_polydata.Modified()

        return 0
    

    def write_vtk_to_file(self, target, base, suffix):
        """Write field line data to VTK file.
         
        Inputs:
            target = main folder that will contain subdirectory with plots
            
            base = basename of file used to create file name for plot.  
                base is derived from name of file with BATSRUS data.
            
            suffix = suffix is used to generate file names and subdirectory.
                Plots are saved in target + suffix directory, target is the 
                overarching directory for all plots.  It contains subdirectories
                (suffix) where different types of plots are saved
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPolyDataWriter
        import os
       
        logging.info('Writing point cloud VTK data to file') 
        logging.info(f'Saving {base} {suffix} VTK data')

        # Store the charts in a file.
        create_directory(target, suffix +'/')
        # filename = target + suffix + '/' + base + '.out.' + suffix + '.vtk'
        name = base + '.' + suffix + '.vtk'
        filename = os.path.join( target, suffix, name )

        if( self.vtk_polydata == None ):
            logging.info('Before saving data, use create_to_vtk to create VTK data')
            return -1
        
        if( filename == None ):
            logging.info('Valid filename to store vtk_polydata must be provided')
            return -1
        
        if( not filename.endswith('.vtk') ):
           logging.info('Filename ending in .vtk expected')
           return -1
        
        path = os.path.dirname(filename)
        if( not os.path.isdir(path) ):  
            logging.info('Filename must contain a path to a valid directory')
            return -1
        
        # Everything looks OK, so write data to file
        writer = vtkPolyDataWriter()
        writer.SetInputData(self.vtk_polydata)
        writer.SetFileName(filename)
        writer.Write()
        return 0

    def display_vtk(self, earth = True):
        """Display VTK field lines.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, \
            vtkRenderWindowInteractor, vtkEarthSource, vtkSphereSource
        from vtkmodules.vtkCommonColor import vtkNamedColors
       
        logging.info('Displaying point cloud VTK data') 

        if( self.vtk_polydata == None ):
            logging.info('Before displaying data, use create_to_vtk to create VTK data')
            return -1
        
        # See example code at:
        # https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/CylinderExample/
        
        # The mapper is responsible for pushing the geometry into the graphics
        # library. It may also do color mapping, if scalars or other
        # attributes are defined.
        polydataMapper = vtkPolyDataMapper()
        polydataMapper.SetInputData(self.vtk_polydata)

        # The actor is a grouping mechanism: besides the geometry (mapper), it
        # also has a property, transformation matrix, and/or texture map.
        # Here we set its color and rotate it.
        polydataActor = vtkActor()
        polydataActor.SetMapper(polydataMapper)
        colors = vtkNamedColors()
        polydataActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
        polydataActor.RotateX(-90.0)
        # polydataActor.RotateY(-45.0)

        # Create the graphics structure. The renderer renders into the render
        # window. The render window interactor captures mouse events and will
        # perform appropriate camera or actor manipulation depending on the
        # nature of the events.
        ren = vtkRenderer()
        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # Add the actors to the renderer, set the background and size
        ren.AddActor(polydataActor)
        ren.SetBackground(colors.GetColor3d("BkgColor"))
        renWin.SetSize(1000, 1000)
        renWin.SetWindowName('VTK Field Lines')

        if( earth ):
            # Add earth to image
            # Start with earth source
            earthSource = vtkEarthSource()
            earthSource.OutlineOn()
            earthSource.Update()
            earthSource.SetRadius(1.0)
            
            # Create a sphere to map the earth onto
            sphere = vtkSphereSource()
            sphere.SetThetaResolution(100)
            sphere.SetPhiResolution(100)
            sphere.SetRadius(earthSource.GetRadius())
            
            # Create mappers and actors
            earthMapper = vtkPolyDataMapper()
            earthMapper.SetInputConnection(earthSource.GetOutputPort())
            
            earthActor = vtkActor()
            earthActor.SetMapper(earthMapper)
            earthActor.GetProperty().SetColor(colors.GetColor3d('Black'))
            earthActor.RotateX(-90.0)
            # earthActor.RotateY(45.0)
            earthActor.RotateZ(240.0)
            
            sphereMapper = vtkPolyDataMapper()
            sphereMapper.SetInputConnection(sphere.GetOutputPort())
            
            sphereActor = vtkActor()
            sphereActor.SetMapper(sphereMapper)
            sphereActor.GetProperty().SetColor(colors.GetColor3d('DeepSkyBlue'))
            
            # Add the actors to the scene
            ren.AddActor(earthActor)
            ren.AddActor(sphereActor)

        # This allows the interactor to initalize itself. It has to be
        # called before an event loop.
        iren.Initialize()

        # We'll zoom out a little by accessing the camera and invoking a "Zoom"
        # method on it.
        ren.ResetCamera()
        ren.GetActiveCamera().Zoom(0.9)
        renWin.Render()

        # Start the event loop.
        iren.Start()

        return 0
