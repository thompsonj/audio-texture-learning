#!/usr/bin/env python
# encoding: utf-8
"""
texture-learning.py


Created by Jessica Thompson on 2012-03-03 for MUS 105 - Musical Systems
Updated April 21, 2012

sample usage:
    python texture_learning.py waterboiling_800.wav 60 3 .4 1 ifft2

Instructor: David Dunn
Dartmouth College
"""
import sys
import os
from bregman.suite import *
from scikits.audiolab import *
import numpy as np
import scipy as sp
import scipy.stats
from scipy.fftpack import ifft2
from scipy.spatial.distance import euclidean 
import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def texture_learning2(targetf,numsecs=60, blobmean=3, blobvar=.4, skip=1, vid='ifft2'):
#targetf = 'static_train_200.wav'
    [target,fs,enc] = wavread(targetf)
    length=target.shape[0]
    #target = target[:,0].transpose()
    target = target.transpose()
    # start with three seconds of repeated original sounds and two secs of silence
    reporig = int(round((2*fs)/length))
    orig = np.tile(target,(1,reporig))
    silence = np.zeros((1,2*fs))
    out = np.append(orig,silence)
    random.seed()
    p = Features.default_params()
    p['feature']='stft'
    p['nfft']=1024
    p['wfft']=512
    p['nhop']=256
    #p['magnitude']=False
    F = Features(target,p)
    # target info
    STFTmag = abs(F.STFT)
    STFTpha = np.angle(F.STFT)
    STFTmag_ravel = STFTmag.ravel()
    STFTpha_ravel = STFTpha.ravel()
    min_mag = min(STFTmag_ravel)
    max_mag = max(STFTmag_ravel)
    min_pha = min(STFTpha_ravel)
    max_pha = max(STFTpha_ravel)
    mean_mag = np.mean(STFTmag_ravel)
    mean_pha = np.mean(STFTpha_ravel)
    std_mag = np.std(STFTmag_ravel)
    std_pha = np.std(STFTpha_ravel)
    # evolving estimations
    MAG = np.zeros(STFTmag.shape)
    PHA = np.zeros(STFTpha.shape)
    #Y = np.zeros(STFTmag.shape)
    err = ((STFTmag - MAG)**2).mean()
    print "%05d err = %2.9f"%(0,err)
    numsucessfulMAGsteps = 0
    numsucessfulPHAsteps = 0
    # generate numsecs seconds of sound
    numsteps = int(round((numsecs*fs)/length))*skip
    noext = os.path.splitext(targetf)[0]
    dirname = noext + '_phase_'+str(blobmean)+'_'+str(int(blobvar*100))+'_'+str(skip)+'_'+vid
    os.mkdir(dirname)
    fig = plt.figure()
    imagesc(STFTmag,dbscale=True)
    for j in range(reporig):
        plt.savefig(dirname+'/'+'%05d'%j+'.png')
    fig = plt.figure()
    imagesc(MAG,dbscale=True)
    for k in range(reporig):
        plt.savefig(dirname+'/'+'%05d'%(j+k)+'.png')
    currout = j+k
    for i in range(numsteps):
        dev = blobmean*blobvar
        if blobmean == 1:
            strt=1
        else:
            strt = int(np.round(blobmean-dev))
        end = int(np.round(blobmean+dev))+1
        size = random.randrange(strt,end)
        if size ==1:
            halfsize =1
        else:
            halfsize = int(np.round(size/2))
        #rval = random.normalvariate(mean,std)
        # choose value to set segment of spectrogram to
        # sample from target sound randomly
       #spect_val = random.expovariate(1/(mean_mag))
        spect_val = random.sample(STFTmag.ravel(), 1)[0]*1.8
        #phase_val = random.expovariate(1/(mean_pha))
        phase_val = random.sample(STFTpha.ravel(), 1)[0]*1.8
        #a,b = random.randint(0+halfsize,int(STFTmag.shape[0]/8)), random.randint(0+halfsize,STFTmag.shape[1]-halfsize)
        a,b = random.randint(0+halfsize,STFTmag.shape[0]-halfsize), random.randint(0+halfsize,STFTmag.shape[1]-halfsize)
        #a,b = random.expovariate(1/(-1*int(round(STFTmag.shape[0]/16))))
        if size ==1:
            aa =a
            bb = b
        else:
            ast = a-halfsize
            aed =a+halfsize
            bst = b-halfsize
            bed =b+halfsize
            aa = np.ravel([[e]*size for e in range(ast,aed)])
            #print 'aa: ',aa
            bb = np.tile(range(bst,bed),(1,size)).ravel()
            #print 'bb:',bb
        thisstep_mag = MAG.copy()
        thisstep_pha = PHA.copy()
        #thisstep = Y.copy()
        #return aa,bb,rval,thisstep,Y
        thisstep_mag[aa,bb] = spect_val
        thisstep_pha[aa,bb] = phase_val
        thisstep = thisstep_mag*np.exp(1j*thisstep_pha)
        #plt.show()
        if i%skip ==0:
            out = np.append(out,F.inverse(thisstep,pvoc=False))
            fig = plt.figure()
            imagesc(thisstep_mag,dbscale=True)
            plt.savefig(dirname+'/stft_'+'%05d'%(currout+1)+'.png')
            # ffmpeg -sameq -i ../laughing_200_phase_2_40_1000_ifft2_learned.wav -i ifft2.mov ifft2_sound.mov 
            # ffmpeg -sameq -i ../static_train_500_phase_3_40_900_ifft2_learned.wav -i ifft2.mov ifft2_sound.mov 
            # ffmpeg -f image2 -r 2 -i ifft2_%05d.png -sameq ifft2.mov
            # ffmpeg -f image2 -r 5 -i ifft2_%05d.png -sameq ifft2.mov
            # ffmpeg -f image2 -vf crop=497:481:100:60 -r 4.8 -i ifft2_%05d.png -sameq ifft2_nosilence.mov THIS ONE GETS THE TIME ALIGNMENT CORRECT 4.8 fps instead of 5, and cropping
            inv = np.real(ifft2(thisstep))
            # normalize between 0 and 1
            if inv.min() < 0:
                inv = inv-inv.min()
            inv = np.divide(inv,abs(inv).max())
            if i > 0 and i < 3:
                inv[inv>.8] = 0
            if i >= 3 and i < 5:
                inv[inv>.7] = 0
            if i >= 6 and i < 15:
                inv[inv>.6] = 0
            if i >= 15:
                inv[inv>.5] = 0
            mn = abs((inv*-1).max())
            inv = inv -mn
            inv = np.divide(inv,abs(inv).max())
            #plt.hist(inv)
            #plt.imshow(np.real(inv),cmap=cm.Greys_r)
            #plt.savefig(dirname+'/hist_'+'%05d'%(currout)+'.png')
            imagesc(inv)
            plt.savefig(dirname+'/ifft2_'+'%05d'%(currout+1)+'.png')

            currout += 1
        # update phase and magnitude separately
        if euclidean(STFTmag[aa,bb],thisstep_mag[aa,bb]) < euclidean(STFTmag[aa,bb],MAG[aa,bb]):
            MAG[aa,bb] = spect_val
            numsucessfulMAGsteps += 1
            #print 'helpful magnitude'
        if euclidean(STFTpha[aa,bb],thisstep_pha[aa,bb]) < euclidean(STFTpha[aa,bb],PHA[aa,bb]):
            MAG[aa,bb] = spect_val
            numsucessfulPHAsteps += 1
            #print 'helpful phase'
        
        if not i%100:
            err = ((STFTmag - MAG)**2).mean()
            print "%05d err = %2.9f"%(i,err)
    
    fname = noext + '_phase_'+str(blobmean)+'_'+str(int(blobvar*100))+'_'+str(skip)+'_'+vid+'_learned.wav'
    wavwrite(out, fname,fs)
    print 'number of steps that reduced euclidean distance of magnitude: ',numsucessfulMAGsteps
    print 'number of steps that reduced euclidean distance of phase: ',numsucessfulPHAsteps

def chopwav(targetf):
    [target,fs,enc] = wavread(targetf)
    # make several versions from length 100ms to 1sec
    for i in [200]:
        numsamples = i*(fs/1000)
        tosave = target[0:numsamples-1]
        noext = os.path.splitext(targetf)[0]
        wavwrite(tosave, noext + '_'+str(i)+'.wav', fs)


def main():
    argv=sys.argv
    print argv[1]
    if len(sys.argv) == 7:
        texture_learning2(argv[1], int(argv[2]), int(argv[3]),float(argv[4]),int(argv[5]),argv[6])
    else:
        texture_learning2(argv[1])


if __name__ == '__main__':
    main()