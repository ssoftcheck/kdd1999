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
			self.popMean = df.apply(lambda x: x[yvar]*x[wvar],axis=1).sum() / df[wvar].sum()
	
	def looVal(self,x,xv,yv,t,tv,j,mw,wv=1):
		if t in tv:
			result = (self.lookup[x].loc[xv,self.yvar] - yv	*wv + self.popMean) / (self.lookup[x].loc[xv,'weight'] - wv + mw)
		else:
			result = self.lookup[x].loc[xv,self.yvar] / self.lookup[x].loc[xv,'weight']
		if np.isnan(result):
			result = self.popMean
		return result * uniform(low=1-j,high=1+j)
	
	def applyLookup(self,df,trainVar,trainValue,jitter=0.2,meanWeight=1,keep=set()):
		if self.wvar is None:
			df = df[self.vars + [self.yvar,trainVar] + list(keep)]
			for i in self.vars:
				df['loo_'+i] = df.apply(lambda x: self.looVal(i,x[i],x[self.yvar],x[trainVar],set(trainValue),jitter,meanWeight),axis=1)
		else:
			df = df[self.vars + [self.yvar,self.wvar,trainVar] + list(keep)]
			for i in self.vars:
				df['loo_'+i] = df.apply(lambda x: self.looVal(i,x[i],x[self.yvar],x[trainVar],set(trainValue),jitter,meanWeight,x[self.wvar]),axis=1)
		return df[ list(filter(lambda x: x[:4]=='loo_' or x in keep,df.columns)) ]
