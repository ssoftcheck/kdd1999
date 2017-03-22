import pandas
from numpy.random import uniform

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

	
	def applyLookup(self,df,trainVar,trainValue,wvar=None,jitter=0.2,meanWeight=1,keep=None):
		train_set = set(trainValue)
		trainInd = [i in train_set for i in df[trainVar]]
		testInd = [not i for i in trainInd]

		if wvar is None:
			df = df.loc[:,self.vars + [self.yvar]]
			for i in self.vars:
				# training cases
				df.loc[trainInd,'loo_'+i] = (self.lookup[i].loc[df.loc[trainInd,i]] - df.loc[trainInd,self.yvar] + self.popMean) / (self.weightSum - 1 + self.meanWeight)
				df.loc[trainInd,'loo_'+i] = df.loc[trainInd,'loo_'+i] * uniform(1-jitter,1+jitter,sum(trainInd))
				# test cases
				df.loc[testInd,'loo_'+i] = self.lookup[i].loc[df.loc[testInd,i]] / self.weightSum
		else:
			df = df.loc[:,self.vars + [self.yvar,wvar]]
			for i in self.vars:
				# training cases
				df.loc[trainInd,'loo_'+i] = (self.lookup[i].loc[df.loc[trainInd,i]] - df.loc[trainInd,self.yvar]*df.loc[trainInd,wvar] + self.popMean) / (self.weightSum - df.loc[trainInd,wvar] + self.meanWeight)
				df.loc[trainInd,'loo_'+i] = df.loc[trainInd,'loo_'+i] * uniform(1-jitter,1+jitter,sum(trainInd))
				# test cases)
				df.loc[testInd,'loo_'+i] = self.lookup[i].loc[df.loc[testInd,i]] / self.weightSum
		
		return df[ list(filter(lambda x: x[:4]=='loo_' or x in keep,df.columns)) ]
