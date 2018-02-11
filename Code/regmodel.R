data <- read.csv('epi_hist.csv')

coef.sigs.g = c('gdp','gpc','gdp_growth','gpc_growth')
#coef.sigs.g = c('gdp_growth','gpc_growth')
#coef.sigs.g = c('gpc_growth','gpc')
#coef.sigs.g = c('gdp','gpc')
data_econ <- data[,coef.sigs.g]

EPI <- data$EPI
EPI_P <- data$EPI_P
EPI_MA <- EPI_P
beta=0.5
for (i in 2:length(EPI_P)) {
  EPI_MA[i] = beta*EPI_MA[i-1]+(1-beta)*EPI_P[i]
}

for (c in colnames(data_econ)) {
  cross <- EPI_P * data_econ[,c]
  data_econ[,paste(c,'_cross',sep='')] <- cross
}
data_econ[,'EPI_MA'] = EPI_MA

lmfit <- lm(EPI~.,data=data_econ)
print(summary(lmfit))

stepr <- step(lmfit)
summary(stepr)

fin_coef_name <- names(coef(stepr))[2:length(coef(stepr))]
data_fin <- data_econ[,fin_coef_name]

data_fin[,'EPI'] = EPI
write.csv(data_fin,'regmodel_MA.csv')
