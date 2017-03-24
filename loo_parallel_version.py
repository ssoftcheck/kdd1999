import pandas
import numpy as np
from numpy.random import uniform
from multiprocessing import Pool

class loo:
	def __init__(self,df,yvar,vars,wvar=None):
		self.lookup = {}
		self.vars = list(set(vars))
		self.yvar = yvar
		self.wvar = wvar
		if wvar is None:
			df = df[list(set(self.vars + [yvar]))]
			for i in self.vars:
				self.lookup[i] = df[[yvar,i]].groupby(i).agg({yvar:{sum,len}})
				self.lookup[i].columns = self.lookup[i].columns.droplevel()
				self.lookup[i].rename(columns={'sum':yvar,'len':'weight'},inplace=True)
			self.popMean = df[yvar].mean()
		else:
			df = df[list(set(self.vars  + [wvar] + [yvar]))]
			for i in self.vars:
				self.lookup[i] = pandas.concat([df.apply(lambda x: x[yvar]*x[wvar],axis=1).to_frame().rename(columns={0:yvar}),df[[i,wvar]]],axis=1).groupby(i).sum()
				self.lookup[i].rename(columns={wvar:'weight'})
			self.popMean = df.apply(lambda x: x[yvar]*x[wvar],axis=1).sum() / df[wvar].sum()	
	
	def applyCalc(self,baseSum, yvar, jitter, weightSum, meanWeight):
		assert (len(baseSum) == len(yvar) and len(baseSum) == len(jitter) and len(baseSum) == len(weightSum))
		n = len(baseSum)
		result = np.empty(n)
		for i in range(n):
			result[i] = jitter[i] * ((baseSum[i] - yvar[i] + self.popMean) / (weightSum[i] - 1 + meanWeight))
		result[np.isnan(result)] = self.popMean
		return result
		
	def applyCalcWeight(self,baseSum, yvar, jitter, weights, weightSum, meanWeight):
		assert (len(baseSum) == len(yvar) and len(baseSum) == len(jitter) and len(baseSum) == len(weights) and len(baseSum) == len(weightSum))
		n = len(baseSum)
		result = np.empty(n)
		for i in range(n):
			result[i] = jitter[i] * ((baseSum[i] - yvar[i]*weights[i] + self.popMean) / (weightSum[i] - weights[i] + meanWeight))
		result[np.isnan(result)] = self.popMean
		return result
		
	def applyCalcTest(self,baseSum, weightSum):
		assert (len(baseSum) == len(weightSum))
		n = len(baseSum)
		result = np.empty(n)
		for i in range(n):
			result[i] = baseSum[i] / weightSum[i]
		result[np.isnan(result)] = self.popMean
		return result
	
	def parallelApply(self,data,varName,jitter,meanWeight,func,train,partitions,cores):
		split = np.array_split(data,partitions)
		pool = Pool(cores)
		if not train:
			data = np.concatenate(pool.starmap(func,[(self.lookup[varName].loc[i[varName],self.yvar].values,self.lookup[varName].loc[i[varName],'weight'].values) for i in split]))
		elif self.wvar is None:
			data = np.concatenate(pool.starmap(func,[(self.lookup[varName].loc[i[varName],self.yvar].values,i[self.yvar].values,uniform(1-jitter,1+jitter,len(i)),self.lookup[varName].loc[i[varName],'weight'].values,meanWeight) for i in split]))
		else:
			data = np.concatenate(pool.starmap(func,[(self.lookup[varName].loc[i[varName],self.yvar].values,i[self.yvar].values,uniform(1-jitter,1+jitter,len(i)),i[self.wvar].values,self.lookup[varName].loc[i[varName],'weight'].values,meanWeight) for i in split]))
		pool.close()
		pool.join()
		return data
		
	def applyLookup(self,df,trainVar,trainValue,jitter=0.2,meanWeight=1,keep=set()):
		train_set = set(trainValue)
		trainInd = [i in train_set for i in df[trainVar]]
		testInd = [not i for i in trainInd]
		
		if self.wvar is None:
			df = df.loc[:,self.vars + [self.yvar] + list(keep)]
			for i in self.vars:
				df['loo_'+i] = np.empty(len(df))
				# training cases
				df.loc[trainInd,'loo_'+i] = self.parallelApply(df.loc[trainInd],i,jitter,meanWeight,self.applyCalc,True,8,8)
				# test cases
				df.loc[testInd,'loo_'+i] = self.parallelApply(df.loc[testInd],i,jitter,meanWeight,self.applyCalcTest,False,8,8)\

		else:
			df = df.loc[:,self.vars + [self.yvar,self.wvar] + list(keep)]
			for i in self.vars:
				df['loo_'+i] = np.empty(len(df))
				# training cases
				df.loc[trainInd,'loo_'+i] = self.parallelApply(df.loc[trainInd],i,self.applyCalcWeight,True,8,8)
				# test cases
				df.loc[testInd,'loo_'+i] = self.parallelApply(df.loc[testInd],i,self.applyCalcTest,False,8,8)
		
		return df[ list(filter(lambda x: x[:4]=='loo_' or x in keep,df.columns)) ]
