#!/usr/bin/env python
import sys

from distutils.core import setup, Extension

import cv2 as cv
import numpy as np

import scipy.io as sio
import numpy.matlib as npm

import time
import json

# imports for calling C++ reg.growing code
import ctypes
import callRGR

# for parallelism
import multiprocessing

import pickle

import time
import copy

from skimage import measure
#from shapely.geometry import Polygon


#####

def _get_polygons_from_mask(mask):
    mask2 = mask.copy()
    #print(mask2)
    count = 0
    for idx,m in enumerate(mask2):
        if m == 1576 or m == 0:
            #count = count + 1
            mask2[idx] = 255
        else:
            mask2[idx] = 0
    #clsList = np.unique(mask2)
    #print(clsList)
    #print(count)
    mask2 = np.reshape(mask2,(512,512))    
    #print(mask.shape)
    #print(mask2.shape)
    #contours = measure.find_contours(mask2, 0.5, positive_orientation='low')
    #print(len(contours))
    #print(contours)
    segmentations = _get_polygons_from_mask2(mask)
    print(segmentations)

def _get_polygons_from_mask2(mask):
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    return segmentations


def sampleSeedToImg(s, height, width, itSet):
    #print(s[0])
    #print(s[1])
    #print(len(s[0]))
    #print(len(s[1]))
    
    init_img = np.zeros((height,width,3),dtype=int)
    img = np.uint8(init_img)
    
    #img = cv.circle(img, (100,100), radius=1, color=(255), thickness=-1)
    for idx, x in enumerate(s[0]):
        y = s[1][idx]
        #print(x,y)
        #cv.circle(img, (x,y), radius=1, color=(0,0,255), thickness=-1)
        img[y][x] = (0,255,0)
        
    cv.imwrite('static/seeds_'+str(itSet)+'.png', img)
    
    
def sampleSeedToImg2(S, itSet):
    pass
    #cv.imwrite('static/seeds_'+str(itSet)+'.png', S*250)
    

def regGrowing(area,numSamples,R_H,height,width,sz,preSeg,m,img_r,img_g,img_b,clsMap,numCls,return_dict,itSet):
    #print('regGrowing')
    #if seeds is None:
    #if True:
    # h is the index o pixels p_h within R_H. We randomly sample seeds
    # according to h~U(1,|R_H|)
    # round down + Uniform distribution
    h = np.floor(area * np.random.random((area,)))
    h = h.astype(np.int64)
 
    # s is the index of each seed for current region growing step
    # sequence
    idSeeds = np.arange(0,numSamples) # IDs of random seeds
    idSeeds = idSeeds.astype(np.int64)

    posSeeds = h[idSeeds] # get the position of these seeds within R_H

    # S is the corresponding set of all seeds, mapped into
    # corresponding img-size matrix
    s = R_H[posSeeds]    
    #print(s)
    S = np.zeros((height, width))
            
    S[np.unravel_index(s, S.shape, 'F')] = 1  
    #sampleSeedToImg(np.unravel_index(s, S.shape, 'C'), height, width, itSet)
    #sampleSeedToImg2(S, itSet)

    # for reporting
    

    # allocate memory for output returned by reg.growing C++ code
    RGRout = np.zeros((width*height), dtype=int)    
    
    S = S.flatten(order='F')

    
    #print('regGrowing callrgr')
    
    # call reg.growing code (adapted SNIC) in C++, using Cython (see callRGR.pyx and setup.py)
    # perform region growing. PsiMap is the output map of generated
    out_ = callRGR.callRGR(img_r, img_g, img_b, preSeg.astype(np.int32), S.astype(np.int32), width, height, numSamples, m,RGRout.astype(np.int32))
    PsiMap = np.asarray(out_)     


    #_get_polygons_from_mask(PsiMap)

    # number of generated clusters.  We subtract 2 to disconsider the pixels pre-classified as background (indexes -1 and 0)
    N = np.amax(PsiMap)-2  # jumlah cluster/centroid selain cluster piksel background/yang tidak terkategorikan

    clsScores = clsMap.flatten(order='F')
    clsScores = clsScores.astype(np.double)  # 512*512*jumlahkelas
    

    # majority voting per cluster
    for k in range(0, N):       
        p_j_ = np.nonzero(PsiMap == k)  # index2 dari array yang masuk ke cluster k
        p_j_ = np.asarray(p_j_)

        for itCls in range(0, numCls):
            #print(itCls)
            idxOffset = sz*itCls;
            p_j_cls = p_j_ + idxOffset;  # index2 dari array yang masuk ke cluster k di subset array (512*512) yang mewakili kelas itCls

            noPositives =  (np.count_nonzero(clsScores[p_j_cls] > 0));
            clsScores[p_j_cls] = float(noPositives)/p_j_.size  # set score untuk piksel2 di cluster k untuk kelas itCls
    
    clsScores = np.reshape(clsScores,(height,width,numCls),order='F')    


    return_dict[itSet] = clsScores

    
    #print('regGrowing end')


def regGrowing2(area,numSamples,R_H,height,width,sz,preSeg,m,img_r,img_g,img_b,clsMap,numCls,return_dict,itSet,return_dict2, seeds=None, record_num_seed=False):
    #print('regGrowing')
    if seeds is None:
    #if True:
        # h is the index o pixels p_h within R_H. We randomly sample seeds
        # according to h~U(1,|R_H|)
        # round down + Uniform distribution
        h = np.floor(area * np.random.random((area,)))
        h = h.astype(np.int64)
     
        # s is the index of each seed for current region growing step
        # sequence
        idSeeds = np.arange(0,numSamples) # IDs of random seeds
        idSeeds = idSeeds.astype(np.int64)

        posSeeds = h[idSeeds] # get the position of these seeds within R_H

        # S is the corresponding set of all seeds, mapped into
        # corresponding img-size matrix
        s = R_H[posSeeds]    
        #print(s)
        S = np.zeros((height, width))
                
        S[np.unravel_index(s, S.shape, 'F')] = 1  
        #sampleSeedToImg(np.unravel_index(s, S.shape, 'C'), height, width, itSet)
        sampleSeedToImg2(S, itSet)
    else:
        S = seeds
        #print('definite')
        sampleSeedToImg2(S, itSet)

    # for reporting
    

    # allocate memory for output returned by reg.growing C++ code
    RGRout = np.zeros((width*height), dtype=int)    
    
    S = S.flatten(order='F')

    debug = True    
    if debug:
        ts0 = time.time()
        print('a', time.time() - ts0)
    
    #print('regGrowing callrgr')
    
    # call reg.growing code (adapted SNIC) in C++, using Cython (see callRGR.pyx and setup.py)
    # perform region growing. PsiMap is the output map of generated
    out_ = callRGR.callRGR(img_r, img_g, img_b, preSeg.astype(np.int32), S.astype(np.int32), width, height, numSamples, m,RGRout.astype(np.int32))
    PsiMap = np.asarray(out_)     

    if debug:
        ts1 = time.time()
        print('b', ts1 - ts0)

    #_get_polygons_from_mask(PsiMap)

    # number of generated clusters.  We subtract 2 to disconsider the pixels pre-classified as background (indexes -1 and 0)
    N = np.amax(PsiMap)-2  # jumlah cluster/centroid selain cluster piksel background/yang tidak terkategorikan

    if debug:
        pass
        #print(N)
    clsScores = clsMap.flatten(order='F')
    clsScores = clsScores.astype(np.double)  # 512*512*jumlahkelas
    
    if debug:
        ts2 = time.time()
        print('c', ts2 - ts1)

    # majority voting per cluster
    for k in range(0, N):       
        p_j_ = np.nonzero(PsiMap == k)  # index2 dari array yang masuk ke cluster k
        p_j_ = np.asarray(p_j_)

        for itCls in range(0, numCls):
            #print(itCls)
            idxOffset = sz*itCls;
            p_j_cls = p_j_ + idxOffset;  # index2 dari array yang masuk ke cluster k di subset array (512*512) yang mewakili kelas itCls

            noPositives =  (np.count_nonzero(clsScores[p_j_cls] > 0));
            clsScores[p_j_cls] = float(noPositives)/p_j_.size  # set score untuk piksel2 di cluster k untuk kelas itCls
    
    if debug:
        ts3 = time.time()
        print('d', ts3 - ts2)

    clsScores = np.reshape(clsScores,(height,width,numCls),order='F')    

    if debug:
        ts4 = time.time()
        print('e', ts4 - ts3)

    if debug:
        ts5 = time.time()
        print('f', ts5 - ts4)

    return_dict[itSet] = clsScores
    
    if record_num_seed: 
        return_dict2[itSet] = np.count_nonzero(S)
    
    #print('regGrowing end')
    
    
########
def main(username,img,anns,weight_,m,num_sets=8,border='',arr_seeds=None,definite=False):
    try:
        #print(arr_seeds)
        #print("p8")
        record_num_seed = False
        #definite = False
        debug = False
        single_process = False
        #num_sets = 8
        cell_size = 1.333
        #cell_size = 4
        is_border = False
        #print('sets', num_sets)
        
        if definite and arr_seeds is not None:
            num_sets = len(arr_seeds)
        print('num_sets', num_sets)
        
        ts0 = time.time()
        if debug:
            print(time.time() - ts0)
        
        # get image size, basically height and width
        height, width, channels = img.shape
        heightAnns, widthAnns = anns.shape

        if(widthAnns != width):
            img = cv.resize(img, (widthAnns, heightAnns)) 

        height, width, channels = img.shape

        # flattening (i.e. vectorizing) matrices to pass it to C++ function (** OPENCV LOADS BGR RATHER THAN RGB!)
        img_b = img[:,:,0].flatten() # R channel
        img_g = img[:,:,1].flatten() # G channel
        img_r = img[:,:,2].flatten() # B channel
        #print(img_b.shape)

        img_b = img_b.astype(np.int32)
        img_g = img_g.astype(np.int32)
        img_r = img_r.astype(np.int32)

        # image size 
        sz = width*height

        # load PASCAL colormap in CV format
        lut = np.load('static/images/PASCALlutW.npy')
        #lutnow = np.load('static/images/PASCALlut.npy')
        #print(lutnow.shape)
        #print(lutnow)
        
        ts1 = time.time()
        if debug:    
            print(ts1 - ts0)

        ## RGR parameters
        # fixed parameters
        # m = .1  # theta_m: balance between
        numSets = num_sets    # number of seeds sets (samplings)
        # cellSize = 10-int(weight_)   # average spacing between samples
        #cellSize = 1.333   # average spacing between samples
        #cellSize = 2.666
        #cellSize = 4
        cellSize = cell_size

        # Rectangular Kernel - equal to strel in matlab
        SE = cv.getStructuringElement(cv.MORPH_RECT, (80, 80))  # used for identifying far background

        # RGR - refine each class
        # list of annotated classes
        clsList = np.unique(anns)
        clsList = np.delete(clsList,0) # remove class 0 
        numCls = clsList.size # number of classes
        
        ts2 = time.time()
        if debug:
            print(ts2 - ts1)

        # annotations masks per class
        clsMap = np.zeros((height,width,numCls))
        for itCls in range(0, numCls):
            np.putmask(clsMap[:,:,itCls],anns == clsList[itCls],1) 

        # mask of annotated pixels: 
        # in this case, only annotated traces are high-confidence (index 2),
        # all others are uncertain (index 0)
        preSeg = np.int32(np.zeros((height,width)))        
        
        np.putmask(preSeg,anns > 0,2)
        #for aa in preSeg:
            #for a in aa:
                #if a > 0:
                    #print(a)
        
        RoI = preSeg

        # identify all high confidence pixels composing the RoI
        area = np.count_nonzero(RoI)

        # R_H is the high confidence region, the union of R_nB and R_F
        R_H = np.nonzero(RoI.flatten('F') > 0)
        R_H = R_H[0]

        # number of seeds to be sampled is defined by the ratio between
        # |R_H| and desired spacing between seeds (cellSize)
        # round up
        numSamples = np.ceil(area / cellSize)

        if border != '':
            h, w = anns.shape[:2]
            #print(h)
            #print(w)
            mask = np.zeros((h+2, w+2), np.uint8)
            if is_border:
                cv.floodFill(preSeg, mask, (0,0), -1);
            
        preSeg = preSeg.flatten()
        #print(preSeg)
        #for p in preSeg:
            #if p > -1:
                #pass
                #print(p)

        # matrix that will contain the scoremaps for each iteration
        # ref_cls = np.zeros((height, width, numCls, numSets),dtype=float)    
        ref_cls = np.zeros((height*width*numCls, numSets),dtype=float)    
        
        num_cores = multiprocessing.cpu_count()

        return_dict2 = None
        if not(single_process):
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            if record_num_seed:
                return_dict2 = manager.dict()
        else:
            return_dict = dict()
            if record_num_seed:
                return_dict2 = dict()

        ts3 = time.time()
        if debug:
            print(ts3 - ts2)
        
        ###
        seeds = None
        if not(single_process):
            print('multiprocess')
            jobs = []
            for itSet in range(0, numSets):
                if definite and arr_seeds is not None:
                    seeds = arr_seeds[itSet]
                #p = multiprocessing.Process(target=regGrowing2, args=(area,numSamples,R_H,height,width,sz,preSeg,m,img_r,img_g,img_b,clsMap,numCls,return_dict,itSet,return_dict2,seeds,record_num_seed))
                p = multiprocessing.Process(target=regGrowing, args=(area,numSamples,R_H,height,width,sz,preSeg,m,img_r,img_g,img_b,clsMap,numCls,return_dict,itSet))
                jobs.append(p)
                p.start()
        else:
            print('singleprocess')
            for itSet in range(0, numSets):
                #print(itSet)
                if definite and arr_seeds is not None:
                    seeds = arr_seeds[itSet]            
                #regGrowing2(area,numSamples,R_H,height,width,sz,preSeg,m,img_r,img_g,img_b,clsMap,numCls,return_dict,itSet,return_dict2,seeds,record_num_seed)
                regGrowing(area,numSamples,R_H,height,width,sz,preSeg,m,img_r,img_g,img_b,clsMap,numCls,return_dict,itSet)

        ts4 = time.time()
        if True:
            print(42)
            print(ts4 - ts3)

        if not(single_process):
            for proc in jobs:
                proc.join()

        ts5 = time.time()
        if True:
            print(5)
            print(ts5 - ts4)

        #return_dict_copy = return_dict
        #return_dict_copy = return_dict.copy()
        #return_dict_copy = copy.deepcopy(return_dict)
        ts6a = time.time()
        if False:
            print(61)
            print(ts6a - ts5)

        if not(single_process):
            outputPar = return_dict.values()
            if record_num_seed:
                outputPar2 = return_dict2.values()
        else:
            outputPar = list(return_dict.values())
            if record_num_seed:
                outputPar2 = list(return_dict2.values())
        #print(outputPar)
        ts6 = time.time()
        if True:
            print(6)
            print(ts6 - ts6a)

        #outputPar = list(return_dict_copy.values())
        #print(outputPar2)
        ts6z = time.time()
        if False:
            print(699)
            print(ts6z - ts6)

        outputPar = np.asarray(outputPar)
        if record_num_seed:
            outputPar2 = np.asarray(outputPar2)
        #print(outputPar2)
        if record_num_seed:
            numSeed = np.average(outputPar2)
        else:
            numSeed = 0
        #print(numSeed)
        ts7 = time.time()
        if debug:
            print(7)
            print(ts7 - ts6)
        
        # swapping axes, because parallel returns (numSets,...)
        ref_cls = np.moveaxis(outputPar,0,3)

        ts8 = time.time()
        if debug:
            print(8)
            print(ts8 - ts7)

        # averaging scores obtained for each set of seeds
        ref_M = (np.sum(ref_cls,axis=3))/numSets        

        ts9 = time.time()
        if debug:
            print(9)
            print(ts9 - ts8)

        # maximum likelihood across refined classes scores ref_M
        maxScores = np.amax(ref_M,axis=2)
        maxClasses = np.argmax(ref_M,axis=2)

        detMask = np.uint8(maxClasses+1)

        finalMask = np.zeros((height,width),dtype=float);    
        for itCls in range(0, numCls):       
           np.putmask(finalMask,detMask == itCls+1,clsList[itCls]) 

        finalMask = np.uint8(finalMask-1)
        
        ts10 = time.time()
        if debug:
            print(10)
            print(ts10 - ts9)
        
        
        ###

        np.save('static/'+username+'/lastmask.npy', np.asarray(finalMask,dtype=float))
        # sio.savemat('intermediate.mat', mdict={'anns':anns,'ref_M': ref_M,'ref_cls':ref_cls,'finalMaskRGR':finalMask})  
        # apply colormap
        _,alpha = cv.threshold(finalMask,0,255,cv.THRESH_BINARY)

        finalMask = cv.cvtColor(np.uint8(finalMask), cv.COLOR_GRAY2RGB)    
        im_color = cv.LUT(finalMask, lut)    

        b, g, r = cv.split(im_color)
        rgba = [b,g,r, alpha]
        im_color = cv.merge(rgba,4) 
        #print('main')

        return im_color, numSeed
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
        height, width, channels = img.shape
        
        return np.zeros(height, width, channels+1)
    
    
def createSeedsFromPrevResult(im_color):
    # image as array black/white
    _, _, _, a = cv.split(im_color)
    cv.imwrite('static/debug/'+'dummy1'+'/orig'+''+'.png', a)
    
    # copy
    z = np.copy(a)
    
    
    #print(a)
    #print(a.shape)
  
    # cv dilation
    kernel = np.ones((4,4),np.uint8)
    dilation = cv.dilate(z,kernel,iterations = 1)
    cv.imwrite('static/debug/'+'dummy1'+'/dilation'+''+'.png', dilation)
    
    # edge
    #cv.imwrite('static/debug/'+'dummy1'+'/edgebg'+''+'.png', a)
    
    # cv erotion
    kernel = np.ones((3,3),np.uint8)
    erotion = cv.erode(z,kernel,iterations = 1)
    cv.imwrite('static/debug/'+'dummy1'+'/erotion'+''+'.png', erotion)
    
    # edge
    #cv.imwrite('static/debug/'+'dummy1'+'/edgefg'+''+'.png', a)
    
    # print image array black/white
    """
    try:
        cv.imwrite('static/debug/'+'dummy1'+'/orig'+''+'.png', a)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    """
    
    arr_seeds = []
    user_anns = []
    
    # user annotations
    user_anns = createUserAnnsFromPrevResult(im_color, dilation, erotion)
    cv.imwrite('static/debug/'+'dummy1'+'/useranns'+''+'.png', user_anns)
    
    seed = np.copy(user_anns)
    seed[seed > 0] = 1
    arr_seeds.append(seed)
     
    return arr_seeds, user_anns


def createUserAnnsFromPrevResult(im_color, dilated, eroded):
    pass
    
    bg_cat = 1
    fg_cat = 2
    bg_cat_debug = 128
    fg_cat_debug = 255
    
    img_size = im_color.shape
    height = int(img_size[0])
    width = int(img_size[1])

    # create array with users annotations (same dimensions as image
    bg_seeds_orig = np.zeros((height,width),dtype=int)
    # seeds background
    step_col = 5
    step_row = 5
    count = 0
    for h in range(0,height,step_row):
        for w in range(0,width,step_col):
            bg_seeds_orig[h][w] = bg_cat
            count = count + 1
    print('count seed', count)    
    # minus dilated
    bg_seeds_orig = bg_seeds_orig - dilated
    bg_seeds_orig[bg_seeds_orig < 0] = 0

    # create array with users annotations (same dimensions as image)
    fg_seeds_orig = np.zeros((height,width),dtype=int)
    # seeds foreground
    step_col = 3
    step_row = 3
    count = 0
    for h in range(0,height,step_row):
        for w in range(0,width,step_col):
            fg_seeds_orig[h][w] = 1
            count = count + 1
    print('count seed', count)
    fg_seeds_orig = fg_seeds_orig.astype(bool)
    # minus negative of eroded
    eroded = eroded.astype(bool)
    fg_seeds_orig = np.logical_and(fg_seeds_orig, eroded)
    fg_seeds_orig = fg_seeds_orig.astype(int)
    fg_seeds_orig[fg_seeds_orig > 0] = fg_cat
    
    # combine
    user_anns = fg_seeds_orig + bg_seeds_orig
    
    user_anns_debug = np.copy(user_anns)
    user_anns_debug[user_anns_debug == bg_cat] = bg_cat_debug
    user_anns_debug[user_anns_debug == fg_cat] = fg_cat_debug    
    cv.imwrite('static/debug/'+'dummy1'+'/userannsdebug'+''+'.png', user_anns_debug)
    
    return user_anns
    

def startRGR(username,imgnp,userAnns,cnt,weight_,m,num_sets=8,border='',arr_seeds=None):
    ts0 = time.time()
    #print(time.time() - ts0)

    img = cv.imdecode(imgnp, cv.IMREAD_COLOR)
    #print(type(img))
    #print(img.shape)
    #print(img)
    
    #print(time.time() - ts0)
    ts1 = time.time()
    
    im_color, numSeed = main(username,img,userAnns,weight_,m,num_sets,border,arr_seeds)
    #print(type(im_color))
    #print(im_color.shape)
    #print(im_color)
    #for x in im_color:
    #    for y in x:
    #        if y[3] != 0:
    #            print(y)

    #print(time.time() - ts1)
    ts2 = time.time()
    print('static/'+username+'/refined'+str(cnt)+'.png')
    cv.imwrite('static/'+username+'/refined'+str(cnt)+'.png', im_color)
    #print('static/'+username+'/refined'+str(cnt)+'.png')
    
    # call with seeds based on previous result
    arr_seeds_2, user_anns_2 = createSeedsFromPrevResult(im_color)
    num_sets_2 = 1
    im_color, numSeed = main(username,img,user_anns_2,weight_,m,num_sets_2,border,arr_seeds_2,definite=True)

    #print(time.time() - ts1)
    ts2 = time.time()
    print('static/'+username+'/r_enhanced'+str(cnt)+'.png')
    cv.imwrite('static/'+username+'/r_enhanced'+str(cnt)+'.png', im_color)
    #print('static/'+username+'/refined'+str(cnt)+'.png')
    
    #print(time.time() - ts2)
    return ts2-ts1, numSeed
    

def traceLine(img,r0,c0,r1,c1,catId,thick):
    cv.line(img,(c0,r0),(c1,r1),catId,thick)

    return img

def tracePolyline(img,pts,catId,thick):    
    pts = pts.reshape((-1,1,2))
    cv.polylines(img,np.int32([pts]),False,catId,thick)
   
    return img

def saveGTasImg(username,id_):
    # load PASCAL colormap in CV format
    lut = np.load('static/images/PASCALlutW.npy')

    GTfile = 'static/'+username+'/GT.mat';
    # load ground truth (GT)
    matvar = sio.loadmat(GTfile)
    gtim =  np.asarray(matvar['mtx'],dtype=float)

    # apply colormap
    _,alpha = cv.threshold(np.uint8(gtim),0,255,cv.THRESH_BINARY)

    gtim = cv.cvtColor(np.uint8(gtim), cv.COLOR_GRAY2RGB)    
    im_color = cv.LUT(gtim, lut)    

    b, g, r = cv.split(im_color)
    rgba = [b,g,r, alpha]
    im_color = cv.merge(rgba,4) 

    cv.imwrite('static/'+username+'/GTimage'+ str(id_) +'.png', im_color)


def cmpToGT(username):
    # load current mask generated by the user
    resim = np.load('static/'+username+'/lastmask.npy')

    GTfile = 'static/'+username+'/GT.mat';
    # load ground truth (GT)
    matvar = sio.loadmat(GTfile)
    gtim =  np.asarray(matvar['mtx'],dtype=float)

    # get image size, basically height and width
    height, width = gtim.shape
    heightAnns, widthAnns = resim.shape

    if(widthAnns != width):
        resim = cv.resize(resim, (width, height)) 

    # number of categories
    num = 21

    # pixel locations to include in computation
    locs = np.nonzero(gtim.flatten('F') < 255)
    locs = locs[0]

    # joint histogram
    sumim0 = 1+gtim+(resim*num)
    sumim = sumim0.flatten('F')
    sumim = sumim[locs]
    hs = np.histogram(sumim,range(1, num*num +2))

    count = len(locs)
    confcounts = np.reshape(hs[0],(num,num),'F')

    # confusion matrix - first index is true label, second is inferred label
    sumconf = np.reshape(np.sum(confcounts,1),(num,1))
    denom = npm.repmat(1E-20+sumconf, 1, num)

    conf = 100*confcounts/denom
    rawcounts = confcounts

    accuracies = np.zeros((num,1),dtype=float)

    gtj = np.zeros((num,1),dtype=float)
    resj = np.zeros((num,1),dtype=float)
    gtjresj = np.zeros((num,1),dtype=float)

    for j in range(0, num):
        gtj[j] = np.sum(confcounts[j,:]) + 1E-20
        resj[j] = np.sum(confcounts[:,j]) + 1E-20
        gtjresj[j] = np.sum(confcounts[j,j]) 

        accuracies[j] = 100*gtjresj[j]/float(gtj[j]+resj[j]-gtjresj[j])         

    meanacc = 100*np.sum(gtjresj[1:])/float(np.sum(gtj[1:])+np.sum(resj[1:])-np.sum(gtjresj[1:]))

    return np.append(accuracies,meanacc)

if __name__== "__main__":
    main()

