#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataFlow_01.py
** Data pre-processing pipeline

Includes the following steps
  1. Filling in NaN
  2. train-validation-test partition 
"""

import numpy as np
import pandas as pd

def fillNanWithinDay(df,dayCol,fillCol,spanFillNa=1):
    """fill NaN within date
    
    This function does (forward) fill without crossing dates, using EMA of a trailing
    window. Equal value in dayCol column indicates same date. "Date" here can
    be generalized to block with equal dayCol value
    Parameter:
      df: dataframe, original data
      dayCol: string, column name. Equal value indicates same date (block)
      fillCol: list of straings, names of columns to fill NaN
      spanFillNa: integer. Using a trailing ema of given span to fill NaN.
          spanFillNa=1 is equivalent to 'ffill' of df.fillna()
    return:
      list of pd.DataFrame of day, NaN replaced
    """
    dfList=[]
    dayList=df[dayCol].unique()
    for day in dayList:
        data_1=df.loc[df['date']==day]
        data_1_=data_1[fillCol].ewm(span=spanFillNa).mean()
        data_1_fill=data_1.copy()
        for cname in fillCol:
            toFill=data_1[cname].isna()
            data_1_fill.loc[toFill,cname]=data_1_.loc[toFill,cname]
        dfList.append(data_1_fill)
    
    return dfList
    
def splitTvBlock(dataBlock,trRatio=0.8,randSeed=None):
    """Train-validation split, by block
    
    Arguments:
        dataBlock: list of DataFrame, whcih 
        trRaio: ratio goes into training set
        randSeed: if None, no random shuffle;
    """
    nBlock=len(dataBlock)
    if randSeed is None:
        idxPerm=np.arange(nBlock)
    else:
        np.random.seed(randSeed)
        idxPerm=np.random.permutation(nBlock)
    nTrain=int(round(nBlock*trRatio))
    
    return (pd.concat([dataBlock[idx] for idx in idxPerm[:nTrain]]),
            pd.concat([dataBlock[idx] for idx in idxPerm[nTrain:]]) )



