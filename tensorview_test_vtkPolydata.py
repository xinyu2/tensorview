
import sys

sys.path.append('/users/chen_xy/software/paraview_build/lib/')

sys.path.append('/users/chen_xy/software/paraview_build/lib/site-packages')

from paraview.simple import *
import vtkPVVTKExtensionsCorePython

try: paraview.simple
except: from paraview.simple import *

from paraview import coprocessing
import vtkPVCatalystPython as vtkCoProcessorPython

import vtk
import numpy as np




def coProcess(grid, time, step):
    # initialize data description
    datadescription = vtkCoProcessorPython.vtkCPDataDescription()
    datadescription.SetTimeData(time, step)
    datadescription.AddInput("input")
    RequestDataDescription(datadescription)
    inputdescription = datadescription.GetInputDescriptionByName("input")
    if inputdescription.GetIfGridIsNecessary() == False:
        return
    if grid != None:
        # attach VTK data set to pipeline input
        inputdescription.SetGrid(grid)
        # execute catalyst processing
        DoCoProcessing(datadescription)
         

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      act1 = coprocessor.CreateProducer( datadescription, "input" )
      
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  freqs = {'input': []}
  coprocessor.SetUpdateFrequencies(freqs)
  return coprocessor


coprocessor = CreateCoProcessor()
 
 #--------------------------------------------------------------
 # Enable Live-Visualizaton with ParaView
coprocessor.EnableLiveVisualization(True)
 

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor
    if datadescription.GetForceOutput() == True:
        # We are just going to request all fields and meshes from the simulation
        # code/adaptor.
        for i in range(datadescription.GetNumberOfInputDescriptions()):
            datadescription.GetInputDescription(i).AllFieldsOn()
            datadescription.GetInputDescription(i).GenerateMeshOn()
        return

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)
# ------------------------ Processing method ------------------------
def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);
 
    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=False)
 
    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
     
print(">>> test vtkPolyData starts >>>")
ugrid = vtk.vtkPolyData()
tmstep=100
coProcess(ugrid,tmstep,tmstep)
print("=========== test ends =========")

 

