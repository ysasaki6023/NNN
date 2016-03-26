# -*- coding: UTF-8 -*-
import NNN
import numpy as np
import pandas as pd
import random

nnn = NNN

####################################################3
# Set variables
# n_cells_input : 入力セルの数。ここでは、28 x 28pixelの白黒画像を入力とするので、28*28を設定
n_cells_input  = 28*28
# n_cells_output: 出力セルの数。この例では、入力された手書き数字画像から数字を読み取り0~9に分類を行う。出力は、判別した数字がiならばi番目の出力セルが発火1し、その他のセルは-1となるよう学習する
n_cells_output = 10

# n_hidden_layer_input/middle/output: 中間層のセル数。中間層とは呼ぶものの、特に層状の構造にはなっておらず、相互結合を許した構造となっている。相互結合はランダムに設定され、n_hidden_layer_avg_number_of_connectionによって、１つのニューロンからつながる数を指定。n_hidden_layer_input/outputは、入力セル・出力セルにつながる中間層のセル数をそれぞれ指定
n_hidden_layer_input  = 10
n_hidden_layer_middle = 00
n_hidden_layer_output = 10
n_hidden_layer_avg_number_of_connection = 10
n_cells_inout  = n_cells_input+n_cells_output
n_cells_hidden = n_hidden_layer_input+n_hidden_layer_middle+n_hidden_layer_output
n_cells_total  = n_cells_inout + n_cells_hidden

nnn.AllocNeurons(n_cells_input,n_cells_output,n_cells_hidden)

####################################################3
# Arrange connections
# ニューロン間の接続を指定
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
# Load data
# トレーニングデータは、Kaggleの"digit recognizer"課題よりダウンロード。以下で、(1)画像データの2値化 (2) output layerに与える教示入力データを作成

Number_of_Training_Data = 100
data = pd.read_csv("train.csv",nrows=Number_of_Training_Data+1)
answer = data.ix[:,0]
image  = data.ix[:,1:]
def toBinary(x):
    if x>128: return +1
    else    : return -1
binaryImage = image.applymap(toBinary)
ddy = []
for i in range(len(answer)):
    tmp = [-1 for j in range(10)]
    tmp[answer[i]] = +1
    ddy.append(tmp)

ndx   = binaryImage.as_matrix()
ndy   = np.matrix(ddy)

# cxxのプログラムへデータを渡す。ndxは入力セルに設定され、ndyは出力セルに設定される。Number_of_Training_Dataの数のサンプルを一斉に渡しておき、cxxプログラム内に格納しておく
nnn.SetSamples(ndx,ndy)

####################################################3
# Recording setup
# 出力するデータの種別を指定。Xは発火判定をする前の、そのニューロンへ集まっている信号x結合強度の総和、Rは発火判定を実施した後の発火具合
#nnn.SetRecordTarget2D("W",1,2)
#for i in range(n_cells_total):
"""
for i in range(0, n_cells_input):
    nnn.SetRecordTarget1D("X",i)
    nnn.SetRecordTarget1D("R",i)
"""
"""
for i in range(n_cells_input, n_cells_inout):
    nnn.SetRecordTarget1D("X",i)
    nnn.SetRecordTarget1D("R",i)
"""
for i in range(n_cells_inout, n_cells_total):
    nnn.SetRecordTarget1D("X",i)
    nnn.SetRecordTarget1D("R",i)

####################################################3
# Start learning
nIterOverSamples = 5 # サンプルごとに何回学習を行うか？
nIterPerOneTrial = 20 # 入力してからネットワークの状態が安定するまでに何ステップ待つか？

nnn.OpenOutputfile("output.csv",True)
nnn.Initialize()
nnn.SetXsigma(0.1) # 中間層への入力をランダムにふらつかせるための係数。0でランダム要素なし
nnn.SetXupdateLambda(1.0) # 毎ステップごとに、Xの値を完全に更新してしまうか、それとも1ステップ前の数値を引きずるか？引きずる場合0で、引きずらない場合1
nnn.SetKupdate(0.1) # ニューロン間の結合強度の更新にかける係数。1で理論値通りの更新料。

for iIter in range(nIterOverSamples):
    print "Training:",iIter
    for iSam in random.sample(range(Number_of_Training_Data),Number_of_Training_Data):
        nnn.LearnOnce(iSam,nIterPerOneTrial,"Learning",True,True)
    nnn.ApplyDeltaW() # データセットを一巡したので、ニューロン間の結合変化量を用いて、現状の結合を更新

print "Estimating..."
for iSam in range(Number_of_Training_Data):
    nnn.LearnOnce(iSam,100,"Estimate",True,False)

 
####################################################3
# Check the result
import math

output = pd.read_csv("output.csv")
for i in range(0,10):
    output["OutputX(%d)"%i] = output["X(%d)"%(i+10+28*28)]
    output["OutputR(%d)"%i] = output["R(%d)"%(i+10+28*28)]
estimate = output[output["comment"]=="Estimate"] # Select the data from "Estimate" period
estimate = estimate[estimate["iStep"]==(nIterPerOneTrial-1)] # Take the output from the last iteration
#print estimate["OutputR(1)"]
tot = 0.
for i in range(Number_of_Training_Data):
    for j in range(10):
        num_estimated = int(estimate["OutputR(%d)"%j][estimate["sampleIndex"]==i])
        num_answer    = ndy[i,j]
        tot += pow(num_estimated - num_answer,2)
tot = math.sqrt(tot / 10 / Number_of_Training_Data)
print "E = Sqrt( sum ( ( Estimated value - Correct value ) ^ 2 ) / 10 / # of samples ) = %.3f"%tot

randtot = 0.
for i in range(Number_of_Training_Data):
    for j in range(10):
        rand_estimated = random.choice([-1,+1])
        rand_answer    = random.choice([-1,+1])
        randtot += pow(rand_estimated - rand_answer,2)
randtot = math.sqrt(randtot / 10 / Number_of_Training_Data)
print "For refference, even random answers can give E = %.3f"%randtot
