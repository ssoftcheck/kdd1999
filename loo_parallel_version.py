import pandas
import numpy
from numpy.random import uniform
from multiprocessing import Pool

class loo:
	def __init__(self,df,yvar,vars,wvar=None):
		self.lookup = {}
		self.vars = list(set(vars))
		self.yvar = yvar
		
		if wvar is None:
			df = df[list(set(self.vars + [yvar]))]
			for i in self.vars:
				self.lookup[i] = df[[yvar,i]].groupby(i).sum()
			self.popMean = df[yvar].mean()
			self.weightSum = df.shape[0]
		else:
			df = df[list(set(self.vars  + [wvar] + [yvar]))]
			for i in self.vars:
				self.lookup[i] = df[[yvar,wvar,i]].apply(lambda x: x[yvar]*x[wvar],axis=1).groupby(i).sum()
			self.popMean = df.apply(lambda x: x[yvar]*x[wvar],axis=1).mean()
			self.weightSum = df[wvar].sum()
			
	
	def calcTrain(self,z,i,w):
		if w == '':
			return z.apply(lambda x: uniform(1-x['jitter'],1+x['jitter']) * ((self.lookup[i].loc[x[i]] - x[self.yvar] + self.popMean) / (self.weightSum - 1 + x['meanWeight'])),axis=1)
		else:
			return z.apply(lambda x: uniform(1-x['jitter'],1+x['jitter']) * ((self.lookup[i].loc[x[i]] - x[self.yvar]*x[wvar] + self.popMean) / (self.weightSum - x[wvar] + x['meanWeight'])),axis=1)
		
	def calcTest(self,z,i,w):
		return z.apply(lambda x: self.lookup[i].loc[x[i]] / self.weightSum,axis=1)
	
	def parallelApply(self,data,varName,weightName,func,partitions,cores):
		split = numpy.array_split(data,partitions)
		pool = Pool(cores)
		if weightName is None:
			data = pandas.concat(pool.starmap(func,[(i,varName,'') for i in split]))
		else:
			data = pandas.concat(pool.starmap(func,[(i,varName,weightName) for i in split]))
		pool.close()
		pool.join()
		return data
		
	def applyLookup(self,df,trainVar,trainValue,wvar=None,jitter=0.2,meanWeight=1,keep=[]):
		train_set = set(trainValue)
		trainInd = [i in train_set for i in df[trainVar]]
		testInd = [not i for i in trainInd]
		
		if wvar is None:
			df = df.loc[:,self.vars + [self.yvar]]
		else:
			df = df.loc[:,self.vars + [self.yvar,wvar]]
			
		df['meanWeight'] = meanWeight
		df['jitter'] = jitter
		for i in self.vars:
			# training cases
			df.loc[trainInd,'loo_'+i] = self.parallelApply(df.loc[trainInd],i,wvar,self.calcTrain,8,8)
			# test cases
			df.loc[testInd,'loo_'+i] = self.parallelApply(df.loc[testInd],i,wvar,self.calcTest,8,8)
		
		return df[ list(filter(lambda x: x[:4]=='loo_' or x in keep,df.columns)) ]
