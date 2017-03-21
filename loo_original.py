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
			self.lookup['popMean'] = df[yvar].mean()
			self.weightSum = df.shape[0]
		else:
			df = df[list(set(self.vars  + [wvar] + [yvar]))]
			for i in self.vars:
				self.lookup[i] = df[[yvar,wvar,i]].apply(lambda x: x[yvar]*x[wvar],axis=1).groupby(i).sum()
			self.lookup['popMean'] = df.apply(lambda x: x[yvar]*x[wvar],axis=1).mean()
			self.weightSum = df[wvar].sum()
	
	def looVal(self,x,xv,yv,t,tv,j,mw,wv=1):
		if t in tv:
			result = (self.lookup[x].loc[xv,self.yvar] - yv*wv + self.lookup['popMean'])/(self.weightSum - wv + mw)
		else:
			result = (self.lookup[x].loc[xv,self.yvar]/self.weightSum)
		return result * uniform(low=1-j,high=1+j)
	
	def applyLookup(self,df,trainVar,trainValue,wvar=None,jitter=0.2,meanWeight=1,keep=None):
		if wvar is None:
			df = df[self.vars + [self.yvar] + [trainVar]]
			for i in self.vars:
				df['loo_'+i] = df.apply(lambda x: self.looVal(i,x[i],x[self.yvar],x[trainVar],[trainValue],jitter,meanWeight),axis=1)
		else:
			df = df[self.vars + [self.yvar] + [wvar] [trainVar]]
			for i in self.vars:
				df['loo_'+i] = df.apply(lambda x: self.looVal(i,x[i],x[self.yvar],x[trainVar],[trainValue],jitter,meanWeight,x[wvar]),axis=1)
		return df[ list(filter(lambda x: x[:4]=='loo_' or x in keep,df.columns)) ]
