'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
###==========================================================
###==========================================================
### visualize the cnv-activations of CNN in training 
### based on the mnist cnn example above
### by using catalyst libraries to render images in paraview
### Author: Xinyu Chen
### input:  MNIST_data
### output: catalyst-pipeline
### parms:
###     interval: number of epochs to visualize
###     alpha:    similarity threshold for group coloring
###     totalepoch:  total training epochs
###==========================================================
###==========================================================

from __future__ import print_function
import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--interval', type=int,default=1)
parser.add_argument('--alpha', type=float,default=0.85)
parser.add_argument('--totalepoch', type=int,default=4)
parser.add_argument('--configure', type=str,default='')

args = parser.parse_args()
eps,alp,training_epochs,configure=args.interval,args.alpha,args.totalepoch,args.configure

from timeit import default_timer as timer
import resource,os,sys

from util import cfg
cf=cfg()
cf.parseConfig(configure)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(cf.datadir, one_hot=True)
# mnist = input_data.read_data_sets("../dnn/mnist/MNIST_data/", one_hot=True)
import buildSet as bs
import reNN as rn
import act2grdcp as ag

sys.path.append(cf.paraviewdir)
sys.path.append(cf.paraviewdir+'/python2.7/site-packages')

from paraview.simple import *
import vtkPVVTKExtensionsCorePython

try: paraview.simple
except: from paraview.simple import *

from paraview import coprocessing
import vtkPVCatalystPython as vtkCoProcessorPython
import vtk
#--------------------------------------------------------------
# catalyst functions
#--------------------------------------------------------------
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

#--------------------------------------------------------------
# Global variables that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
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
#--------------------------------------------------------------

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
ALPHA=0.99
# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
keep_list1=np.arange(32) #filters in conv1
keep_list2=np.arange(64) #filters in conv2
val=[mnist.validation.images[0:100,:],mnist.validation.labels[0:100,:]]

# Create model
def multilayer_perceptron(x, keep_prob, weights, biases):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # Hidden layer with RELU activation
    h_conv1 = tf.nn.relu(conv2d(x_image, weights['k1']) + biases['b1'])
    h_pool1 = max_pool_2x2(h_conv1)
    # Hidden layer with RELU activation
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['k2']) + biases['b2'])
    h_pool2 = max_pool_2x2(h_conv2)
    # Output layer with linear activation
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*cnv2size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['f1w']) + biases['f1b'])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, weights['f2w']) + biases['f2b']
    return y_conv,h_conv1,h_conv2,weights,biases

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def nmlAct(a):
  anm=(a-np.mean(a,axis=0))/np.std(a,axis=0)        #normalize activation
  lena=np.max(anm,axis=0)-np.min(anm,axis=0)
  anm/=lena
  anm-=np.min(anm,axis=0)
  return anm

def getCorr(anm):
  corr=np.corrcoef(anm.T)
  return corr

def get10bTime(tprev):
    tnew=timer()
    t10b=tnew-tprev
    return tnew,t10b

r0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
cost_hist,t10b_hist=[],[]
totalPoint=0
for epoch in range(training_epochs):
    t0=timer()
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/(batch_size*1))
    # Construct model
    if epoch==0:
        cnv1size,cnv2size=32,64
        weights, biases = rn.initNN(cnv1size,cnv2size)
    else:
        cnv1size,cnv2size=sb['b1'].shape[0],sb['b2'].shape[0]
        weights, biases = rn.restoreNN(sw,sb)
        del sw,sb
    pred,hc1,hc2,w,b = multilayer_perceptron(x, keep_prob, weights, biases)
    del weights,biases
    act1=np.zeros([28*28,cnv1size],dtype=np.float32)
    act2=np.zeros([14*14,cnv2size],dtype=np.float32)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Initializing the variables
    init = tf.global_variables_initializer()
    r1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print ("@test epoch(%d) mem:%dM" % (epoch,(r1-r0)/1024))
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        print("\ntraining... with size",cnv1size,cnv2size)
        print("===============================")
        # Training cycle
        # Loop over all batches
        tprev,t10=0,0
        for i in range(total_batch*eps):
            tprev=timer()
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c,a1,a2,wa,ba = sess.run([optimizer, cost, hc1,hc2,w,b], feed_dict={x: batch_x,
                   y: batch_y, keep_prob: 0.6})
            # Compute average loss
            avg_cost += c / (total_batch*eps)
            # sum conv1 activation
            act1+=np.sum(a1.reshape(a1.shape[0],-1,a1.shape[-1]),axis=0)
            act2+=np.sum(a2.reshape(a2.shape[0],-1,a2.shape[-1]),axis=0)
            if (i+0)%100==0 or i==(total_batch*eps)-1:
                tprev,t10b=get10bTime(tprev)
                anm=nmlAct(act1)
                cr=getCorr(anm)
                smc=bs.findSimilar(cr,ALPHA)
                while len(smc)==0 and ALPHA>0.9:
                    ALPHA-=0.01
                    smc=bs.findSimilar(cr,ALPHA)
                if len(smc)>0:
                    ms=bs.mergeSl(smc)
                else:
                    ms=[]
                tmstep=epoch*total_batch+i
                if tmstep>10:
                    act=ag.act1ScatterPipe(anm,ms,keep_list1,tmstep)
                    totalPoint=act.shape[0]
                    
                    cost_hist.append(c)
                    t10b_hist.append(t10b)
                    points = vtk.vtkPoints()
                    vertices = vtk.vtkCellArray()
                    Colors = vtk.vtkUnsignedCharArray()
                    Colors.SetNumberOfComponents(3)
                    Colors.SetName("Colors")
                    for i in range(totalPoint):
                        p=act[i,:].tolist()
                        id = points.InsertNextPoint(p)
                        points.SetPoint(id,p[0],p[1],p[2])
                        vertices.InsertNextCell(1)
                        vertices.InsertCellPoint(id)
                        Colors.InsertNextTuple3(128+p[2]*30,0,0)
                    ugrid = vtk.vtkPolyData()
             
                    # Set the points and vertices we created as the geometry and topology of the polydata
                    ugrid.SetPoints(points)
                    ugrid.SetVerts(vertices)
                    ugrid.GetPointData().SetScalars(Colors)        
                    coProcess(ugrid,tmstep,tmstep)
                    del anm,cr,smc,ms    
        sw,sb=wa,ba
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch), "cost=", \
                "{:.9f}".format(avg_cost))
        print("===============================")
        t1=timer()
        tm_trn=t1-t0
        sum_accu,num_batch=0.0,5
        for i in range(num_batch):
            batch = mnist.test.next_batch(2000)
            test_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y: batch[1], keep_prob: 1.0})
            sum_accu+=test_accuracy
        del batch
        r1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        avg_accu=sum_accu/float(num_batch)
        t2=timer()
        tm_tst=t2-t1
        print ("@test epoch(%d)mem usage:read file %dM accuracy(%g)" % (epoch,(r1-r0)/1024,avg_accu))
        # print("Test Accuracy %g"%avg_accu)
    anm=nmlAct(act1)
    c=getCorr(anm)
    smc=bs.findSimilar(c,ALPHA)
    while len(smc)==0 and ALPHA>=alp:
        ALPHA-=0.01
        smc=bs.findSimilar(c,ALPHA)
    if len(smc)>0:
        ms=bs.mergeSl(smc)
    ms=[]
    anm=nmlAct(act2)
    c=getCorr(anm)
    smc=bs.findSimilar(c,ALPHA)
    while len(smc)==0 and ALPHA>alp:
        ALPHA-=0.01
        smc=bs.findSimilar(c,ALPHA)
    if len(smc)>0:
        ms=bs.mergeSl(smc)
    del act1,act2,anm,c,smc,ms
    print("epo(%d) training:(%f)sec testing:(%f)sec"%(epoch,tm_trn,tm_tst))
    epoch+=eps
