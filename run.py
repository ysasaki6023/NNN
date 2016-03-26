import NNN
import numpy as np
import pandas as pd
import random

nnn = NNN

####################################################3
# Set variables
n_cells_input  = 28*28
n_cells_output = 10
n_hidden_layer_input  = 10
n_hidden_layer_middle = 00
n_hidden_layer_output = 10
n_hidden_layer_avg_number_of_connection = 10
n_cells_inout  = n_cells_input+n_cells_output
n_cells_hidden = n_hidden_layer_input+n_hidden_layer_middle+n_hidden_layer_output
n_cells_total  = n_cells_inout + n_cells_hidden
nnn.AllocNeurons(n_cells_input,n_cells_output,n_cells_hidden)

####################################################3
# Load data
#d = pd.read_pickle("input.pickle")
#dans = d.ix[:,1]
#dx   = d.ix[:,2:785]
#dy   = d.ix[:,786:]
d = pd.read_csv("output.csv")
dans = d.ix[:,1]
dx   = d.ix[:,2:2+n_cells_input]
dy   = d.ix[:,2+n_cells_input:2+n_cells_input+n_cells_output]


ndans = dans.as_matrix()
ndx   = dx.as_matrix()
ndy   = dy.as_matrix()

nnn.SetSamples(ndx,ndy)

####################################################3
# Arrange connections
## input,output -> hidden(input)
for i in range(n_cells_inout):
    for j in range(n_cells_inout,n_cells_inout+n_hidden_layer_input):
        nnn.SetConnection(i,j)
        #nnn.SetRecordTarget2D("W",i,j)
## hidden(output) -> input,output
for i in range(n_cells_inout):
    for j in range(n_cells_inout+n_hidden_layer_input,n_cells_inout+n_hidden_layer_input+n_hidden_layer_output):
        nnn.SetConnection(j,i)
        #nnn.SetRecordTarget2D("W",j,i)
## input -> output
for i in range(n_cells_input):
    for j in range(n_cells_input,n_cells_inout):
        nnn.SetConnection(i,j)
        nnn.SetConnection(j,i)

## hidden
for i in range(n_cells_inout, n_cells_inout+n_cells_hidden):
    for j in random.sample(range(n_cells_inout,n_cells_inout+n_cells_hidden), n_hidden_layer_avg_number_of_connection):
        if i==j: continue
        nnn.SetConnection(i,j)
        #nnn.SetRecordTarget2D("W",i,j)

####################################################3
# Setup Recordings
#nnn.SetRecordTarget2D("W",1,2)
#for i in range(n_cells_total):
"""
for i in range(0, n_cells_input):
    nnn.SetRecordTarget1D("X",i)
    nnn.SetRecordTarget1D("R",i)
"""
for i in range(n_cells_input, n_cells_inout):
    nnn.SetRecordTarget1D("X",i)
    nnn.SetRecordTarget1D("R",i)
for i in range(n_cells_inout, n_cells_total):
    nnn.SetRecordTarget1D("X",i)
    nnn.SetRecordTarget1D("R",i)

####################################################3
# Start learning
nSamples    = 5
startSample = 0
import sys
#nIterOverSamples = int(sys.argv[1])
#nIterPerOneTrial = int(sys.argv[2])
#Kupdate = float(sys.argv[3])
nIterOverSamples = 10
nIterPerOneTrial = 20
Kupdate = 0.1

nnn.OpenOutputfile("output/out_%d_%d_%.5f.dat"%(nIterOverSamples,nIterPerOneTrial,Kupdate),True)
nnn.Initialize()
nnn.SetXsigma(0.1)
nnn.SetXupdateLambda(1.0)

for Phase in range(20):
    #Kupdate *= 0.95
    #print Kupdate
    nnn.SetKupdate(Kupdate)
    for iIter in range(nIterOverSamples):
        for iSam in random.sample(range(startSample,startSample+nSamples),nSamples):
            print iIter,iSam
            nnn.LearnOnce(iSam,nIterPerOneTrial,"Learning%d"%Phase,True,True)
            #nnn.LearnOnce(iSam,100,"ImmedEvaluate1-%d"%Phase,True,False)
            #nnn.LearnOnce(iSam,100,"ImmedEvaluate2-%d"%Phase,True,False)
        nnn.ApplyDeltaW()

    nnn.SetKupdate(0)
    #nnn.SetXsigma(0.000001)
    for iSam in range(startSample,startSample+nSamples):
        print iSam
        nnn.LearnOnce(iSam,100,"Evaluate%d"%Phase,True,False)
        #nnn.LearnOnce(iSam,100,"Evaluate2-%d"%Phase,True,False)
