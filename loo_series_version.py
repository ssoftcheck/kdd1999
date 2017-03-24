import pandas
from numpy.random import uniform
import numpy as np

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
			self.popMean = df.apply(lambda x: x[yvar]*x[wvar],axis=1).mean()

	
	def applyLookup(self,df,trainVar,trainValue,jitter=0.2,meanWeight=1,keep=set()):
		train_set = set(trainValue)
		trainInd = [i in train_set for i in df[trainVar]]
		testInd = [not i for i in trainInd]

		if self.wvar is None:
			df = df.loc[:,self.vars + [self.yvar] + list(keep)]
			for i in self.vars:
				df['loo_'+i] = np.empty(len(df))
				# training cases
				df.loc[trainInd,'loo_'+i] = (self.lookup[i].loc[df.loc[trainInd,i],self.yvar] - df.loc[trainInd,self.yvar] + self.popMean) / (self.lookup[i].loc[df.loc[trainInd,i],'weight'] - 1 + meanWeight)
				df.loc[trainInd and np.isnan(df[trainInd,'loo_'+i]),'loo_'+i] = self.popMean
				df.loc[trainInd,'loo_'+i] = df.loc[trainInd,'loo_'+i] * uniform(1-jitter,1+jitter,sum(trainInd))
				# test cases
				df.loc[testInd,'loo_'+i] = self.lookup[i].loc[df.loc[testInd,i],self.yvar] / self.lookup[i].loc[df.loc[testInd,i],'weight']
		else:
			df = df.loc[:,self.vars + [self.yvar,self.wvar] + list(keep)]
			for i in self.vars:
				df['loo_'+i] = np.empty(len(df))
				# training cases
				df.loc[trainInd,'loo_'+i] = (self.lookup[i].loc[df.loc[trainInd,i],self.yvar] - df.loc[trainInd,self.yvar]*df.loc[trainInd,self.wvar] + self.popMean) / (self.lookup[i].loc[df.loc[trainInd,i],'weight']- df.loc[trainInd,self.wvar] + meanWeight)
				df.loc[trainInd and np.isnan(df[trainInd,'loo_'+i]),'loo_'+i] = self.popMean
				df.loc[trainInd,'loo_'+i] = df.loc[trainInd,'loo_'+i] * uniform(1-jitter,1+jitter,sum(trainInd))
				# test cases)
				df.loc[testInd,'loo_'+i] = self.lookup[i].loc[df.loc[testInd,i],self.yvar] / self.lookup[i].loc[df.loc[testInd,i],'weight']
		
		return df[ list(filter(lambda x: x[:4]=='loo_' or x in keep,df.columns)) ]
