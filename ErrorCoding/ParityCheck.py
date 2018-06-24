"""
奇偶校验，包含一维校验与二维校验，二维校验包含纠错算法。
但应明确，所有校验算法都不能100%保证检出所有错误或恢复所有错误。
"""
import numpy as np


def ParityCheck1D_Rec(dataframe,option):
    """
    0:odd check
    1:even check
    """
    data = bin(dataframe)[2:]
    count = 0
    for i in range(len(data)):
        if(data[i] == '1'):
            count += 1
    
    count = count % 2
    
    if(option == 0):
        if(count != 0):
            return True, (dataframe >> 2)
        else:
            return False, dataframe
    else:
        if(count != 0):
            return False, dataframe
        else:
            return True, (dataframe >> 2)


def ParityCheck1D_Send(dataframe,option):
    """
    0:odd check
    1:even check
    """
    data = bin(dataframe)[2:]
    count = 0
    for i in range(len(data)):
        if(data[i] == '1'):
            count += 1
    count = count % 2

    if(option == 0):
        if(count != 0):
            return dataframe << 1
        else:
            return (dataframe << 1) + 1

    else:
        if(count != 0):
            return (dataframe << 1) + 1
        else:
            return dataframe << 1       




def ParityCheck2D_Rec(datamatrix,option):
    """
    0:odd check
    1:even check
    """
    errorrow = []
    errorcol = []
    row = np.shape(datamatrix)[0]
    col = np.shape(datamartix)[1]
    ReturnData = []
    for i in range(len(row)):
        state = ParityCheck1D_Rec(datamatrix[i,:],option)
        if(state == False):
            errorrow.append(i)

    for i in range(len(row)):
        state = ParityCheck1D_Rec(datamatrix[:,i],option)
        if(state == False):
            errorcol.append(i)
    
    if(len(errorow) == 0 and len(errorcol) == 0):
        for i in range(len(row-1)):
            ReturnData.append(int(datamatrix >> 1))
        return True, ReturnData
 
    else:
        if(len(row) == 0 or len(col) == 0):
            return False, datamatrix
        if(len(row) % 2 == 0 and len(col) % 2 == 0):
            return False, datamatrix
        
        for i in range(len(row)):
            xi = i
            for j in range(len(col)):
                xj = j
                datamatrix[xi][xj] = (datamatrix[xi][xj] + 1) % 2

        for i in range(len(row-1)):
            ReturnData.append(int(datamatrix >> 1))
        return True, ReturnData
        
