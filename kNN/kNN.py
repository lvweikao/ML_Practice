from numpy import *
import operator
import os

# 输入：无，此函数为测试函数，可忽略
# 输出：返回矩阵格式的数据group，以及各组数据对应标签类别
# 功能：初始化测试数据
def createDataSet ():
    group = array([
                    [1.0, 1.1],
                    [1.0, 1.0],
                    [0, 0],
                    [0, 0.1]
                 ])
    # print(group)
    labels = ['A', 'A', 'B', 'B']
    # print(labels)
    return group, labels
# 简单获取数值
# group, labels = createDataSet()
# print(group,"\n\r",labels)

# k-近邻算法
# 输入：所需测试数据inX,原始数据集dataSet,原始数据集对应的类别信息labels，可供参考的数据个数（经过排序）
# 输出：类别信息
# 功能：同时输入测试集和待分类数据，判断新的数据应当属于的类别
def classify0(inX, dataSet, labels, k):
# 获取第一维的数据,即测试数据的组数，目的是28line需要行数构造一个相似的矩阵
    dataSetSize = dataSet.shape[0]
# 获取各个对应值的差
# tile的第一个参数是矩阵，第二个参数是分别在列和行的方向上面重复的次数，以此类推，从低维度到高维度方向
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # print (diffMat)
# 去掉负数，同时取平方
    sqDiffMat = diffMat**2
# axis为0的时候对应列值，为1的时候对应行值（依然按照方向定义）
# 此处为1，计算出了新点与各组数据之间的欧式距离
    sqDistances = sqDiffMat.sum(axis = 1)
# 求平方和
    distances = sqDistances**0.5
# 默认升序排序，返回的是index
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        # 获取从小到大的距离对应的分类标签
        # 欧式距离最近的前几个点，哪种分类最多
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(voteIlabel)
        # print(classCount)
        # 字典.get方法，参数：查找的键，和不存在时返回的默认值，返回指定值
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 字典的item方法能够将字典返回成一个可以迭代的序列list
    # 参数1 需要排序的对象，使用字典.items使得字典操作后返回一个元组列表
    # 参数2 规则，比如想只比较第2列的值
    # 参数3 True降，false升
    # 返回值
    # 此处，操作对象是第二列，即类别值，返回第1行第1列的值，即与所要求的数据最接近的类别（键）可能。
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1),
                              reverse = True)
    return sortedClassCount[0][0]
# # 测试KNN
# print(classify0((0.4,0.4), group, labels, 3))

# 处理约会数据
# 输入：文件路径
# 输出：1000行约会对象3特征矩阵，以及每个对象的打分
# 功能：数据预处理
def file2matrix(filename):
    fr = open(filename, encoding='utf-8')
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)

    returnMat = zeros((numberOfLines,3))
    classLabelVector = []

    # 赋值操作，相当于i，因为下面的for的参数是对应的行，所以要另外定义
    for index,line in enumerate(arrayOLines):
        # strip去掉开头或者末尾的字符，默认去除空白符
        line = line.strip('﻿')
        line = line.strip()
        # 切片，以缩进作为分隔
        listFromLine = line.split('\t')
        # 进行赋值操作,listFromLine中实际有4个参数，只需要3个参数
        returnMat[index,:] = listFromLine[0:3]
        # 类别信息放在classLabelVector中,-1指的是最后一项
        classLabelVector.append(int(listFromLine[-1]))
    return returnMat,classLabelVector
# datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat,datingLabels)

# 数据归一化处理，（值-最小值）/(最大值-最小值)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
# a,b,c = autoNorm(datingDataMat)
# print(c)

# 分类器验证程序，选择百分之10数据测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                        datingLabels[numTestVecs:m], 4)
        print("运行结果:%d   真实数据:%d" %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):    errorCount += 1.0
    print("error rate is: %d", (errorCount/float(numTestVecs)))

# 随机输入3个参数，判断属于哪种类别
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("requent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per week?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ",resultList[classifierResult - 1])

# 获取手写系统的文件，然后存进一个1*1024的向量里面
# 为了方便后面的对比吧
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    # 获取目录
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    # 从文件名解析分类数字
    # 构建矩阵，并且获取分类信息
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    # 准备获取测试集数据进行测试
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print("the classifier came back with: %d, the real answer is: %d"
        #       %(classifierResult, classNumStr))
        if(classifierResult != classNumStr): errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f%%" % (errorCount/float(mTest)*100.0))

# handwritingClassTest()
# test = handwritingClassTest()
# print(test)
