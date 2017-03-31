library(data.table)

lookup = function(vars,xdata,yvar,wvar=NULL) {
	result = vector(mode='list',length = length(vars))
	names(result) = vars
  
	if(is.null(wvar)) {
		pb = progress_bar$new(format=' [:bar] :percent Time Left: :eta',total=length(vars),clear=FALSE)
		for(v in 1:length(vars)) {
			groupBy = vars[v]
			result[[v]] = xdata[!is.na(get(yvar)),.(value=mean(get(yvar)),weight=length(get(yvar))),by=groupBy]
			pb$tick()
		}
		result$popMean = mean(xdata[!is.na(get(yvar)),get(yvar)])
	}
	else {
		if(any(is.na(xdata[[wvar]]))) {
			message('Negative Weights')
			stop()
		}
		pb = progress_bar$new(format=' [:bar] :percent Time Left: :eta',total=length(vars),clear=FALSE)
		for(v in 1:length(vars)) {
			groupBy = vars[v]
			result[[v]] = xdata[!is.na(get(yvar)),.(value=sum(get(yvar) * get(wvar)),weight=sum(get(wvar))),by=groupBy]
			pb$tick()
		}
		result$popMean = weighted.mean(xdata[!is.na(get(yvar)),get(yvar)],xdata[!is.na(get(yvar)),get(wvar)])
	}
	return(result)
}

applyLookup = function(xdata,input,wvar=NULL,yvar,train,trainvalue,jitter=0.1,meanWeight=1) {
	vars = setdiff(names(input),'popMean')
	
	result = vector(mode='list',length = length(vars))
	names(result) = vars
	
	hash = new.env(hash=TRUE)
	for(tv in trainvalue) {
		e[[as.character(tv)]] = NULL
	}
	isTrain = sapply(xdata[[train]],function(x) exists(as.character(x),envir=hash))
	rm(list=ls(envir=hash,all=TRUE),envir=hash)
	rm(hash)
	
	pb = progress_bar$new(format=' [:bar] :percent Time Left: :eta',total=length(vars),clear=FALSE)
	
	if(is.null(wvar)) {
		for(v in vars) {
			temp = xdata[,c(v,yvar),with=FALSE]
			temp[,rownum:=1:nrow(temp)]
			temp[,trainInd:=isTrain]
		
			temp = merge(temp,input[[v]],by=v,all.x=TRUE,all.y=FALSE)
			temp = temp[order(rownum)]
			temp[,rownum := NULL]
			temp[,calc := (value - ifelse(trainInd,get(yvar),0) + ifelse(trainInd,meanWeight * input$popMean,0)) / ifelse(trainInd,(weight - 1 + meanWeight),weight)]
			temp[is.na(calc),calc := input$popMean]
			temp[(trainInd), calc := calc * runif(n = sum(trainInd),min = 1-jitter,max = 1+jitter)]
			result[[v]] = temp$calc
			rm(temp)
			pb$tick()
		}
	}
	else {
		if(any(is.na(xdata[[wvar]]))) {
			message('Negative Weights')
			stop()
		}
		
		for(v in vars) {
			temp = xdata[,c(v,yvar,wvar),with=FALSE]
			temp[,rownum:=1:nrow(temp)]
			temp[,trainInd:=isTrain]
		
			temp = merge(temp,input[[v]],by=v,all.x=TRUE,all.y=FALSE)
			temp = temp[order(rownum)]
			temp[,rownum := NULL]
			temp[,calc := (value - ifelse(trainInd,get(yvar)*get(wvar),0) + ifelse(trainInd,meanWeight * input$popMean,0)) / ifelse(trainInd,(weight - get(wvar) + meanWeight),weight)]
			temp[is.na(calc),calc := input$popMean]
			temp[(trainInd), calc := calc * runif(n = sum(trainInd),min = 1-jitter,max = 1+jitter)]
			result[[v]] = temp$calc
			rm(temp)
			pb$tick()
		}
	}
	
	result = as.data.table(result)
	setnames(result,names(result),paste('meancode',yvar,names(result),sep='.'))
	return(result)
}