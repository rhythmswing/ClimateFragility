data <- read.csv('mau_hist.csv')
data <- data[1:(nrow(data)-2),]
#coef.sigs.g = c('gdp','gpc','gdp_growth','gpc_growth')
#coef.sigs.g = c('gdp_growth','gpc_growth')
#coef.sigs.g = c('gpc_growth','gpc')
#coef.sigs.g = c('gdp','gpc')

coef.sigs.g=c('gdp','gdp_growth')
data_econ <- data[,coef.sigs.g]
#data_econ <- data[,'gpc']


# centralize
data_econ <- as.data.frame(sapply(data_econ, scale))

# sigmoid

for (c in colnames(data_econ)) {
  data_econ[,c] <- 1/(1+exp(-data_econ[,c]))
}

EPI <- data$EPI
EPI
EPI_P <- data$EPI_P
EPI_MA <- EPI_P
beta=0.9
beta2=0.9
data_orig <- data_econ


for (c in colnames(data_econ)) {
  cross <- EPI_P * data_econ[,c]
  print(cross)
  data_econ[,paste(c,'_cross',sep='')] <- cross
}


for (i in 2:length(EPI_P)) {
  EPI_MA[i] = beta*EPI_MA[i-1]+(1-beta)*EPI_P[i]
}


data_ma <- data_econ
for (i in 2:nrow(data_ma)){
  data_ma[i,] = beta2*data_ma[i-1,] + (1-beta2)*data_econ[i,]
}


data_econ <- data_ma

data_econ[,'EPI_MA'] = EPI_MA
#data_econ = as.data.frame(sapply(data_econ,scale))

data_econ <- data_econ[2:nrow(data_econ),]
EPI<-EPI[2:length(EPI)]
EPI_P<-EPI_P[2:length(EPI_P)]


lmfit <- lm(EPI~.,data=data_econ)
print(summary(lmfit))

stepr <- step(lmfit, k=log(nrow(data_econ)))
summary(stepr)

fin_coef_name <- names(coef(stepr))[2:length(coef(stepr))]
data_fin <- data_econ[,fin_coef_name]

data_fin[,'EPI_P'] = EPI_P
data_fin[,'EPI'] = EPI
write.csv(data_fin,'regmodel_MA.csv')
