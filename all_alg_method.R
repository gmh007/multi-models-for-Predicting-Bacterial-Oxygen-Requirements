#清空变量
rm(list = ls())

#加载必要的程序包
library(e1071)

#读取矩阵
raw_data <- read.csv('data.txt',header = T,sep = '\t',row.names = 1,check.names = F)

#看一下有多少列
ncol(raw_data)
#行
nrow(raw_data)

#读取实验设计表
design <- read.csv('design.txt',header = T,sep = '\t',row.names = 1)
head(design)

#以group1为训练集数据，group2为测试集数据，来划分数据类型

#设置一个索引index检查样本是否存在缺失或者多余或者重复等其他不应出现的问题
design_group1 <- subset(design,Group %in% c('group1')) 
head(design_group1)

#构建训练集数据
index_group1 <- rownames(design_group1) %in% rownames(raw_data)
index_group1
table(index_group1)
#检查group1有多少个样本
nrow(design_group1)
train_data <- raw_data[rownames(design_group1),]
#看一下训练集的维度以及行名
dim(train_data)
class(train_data)
head(rownames(train_data))
#训练集数据格式的转换

# train_data[,1:10351] <- as.numeric(unlist(train_data[,1:10351]))
train_data[,1:10351] <- as.data.frame(lapply(train_data[,1:10351],as.numeric))
class(train_data$Fer4_7)
#需氧类型转换为因子factor
class(train_data$Type)
train_data$Type <- factor(train_data$Type)
class(train_data$Type)

#构建测试集数据
design_group2 <- subset(design,Group %in% c('group2')) 
test_data <- raw_data[rownames(design_group2),]
test_data[,1:10351] <- as.data.frame(lapply(test_data[,1:10351],as.numeric))
test_data$Type <- factor(test_data$Type)

##############################训练集数据和测试集数据全部整好了###################################

#svm
set.seed(1213)
svm_model <- svm(Type~.,train_data)
train_result <- predict(svm_model,train_data)
#看一下训练集预测的结果
table(true=train_data$Type,predict=train_result)
conf_matrix <- table(true=train_data$Type,predict=train_result)
svm_train_result <-sum(diag(conf_matrix))/sum(conf_matrix)
svm_train_result<-paste(round(svm_train_result*100,2),"%",sep = '')
write.table(conf_matrix,file="svm_train_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)
#看一下测试集的预测结果
test_result <- predict(svm_model,test_data)
table(true=test_data$Type,predict=test_result)
conf_matrix <- table(true=test_data$Type,predict=test_result)
svm_test_result <-sum(diag(conf_matrix))/sum(conf_matrix)
paste(round(svm_test_result*100,2),"%",sep = '')
write.table(conf_matrix,file="svm_test_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)

#条件推理树算法
library(party)
set.seed(1213)
ctree_model <- ctree(Type~.,train_data)
train_result <- predict(ctree_model,train_data)
table(true=train_data$Type,predict=train_result)
conf_matrix <- table(true=train_data$Type,predict=train_result)
ctree_train_result <-sum(diag(conf_matrix))/sum(conf_matrix)

paste(round(ctree_train_result*100,2),"%",sep = '')
write.table(conf_matrix,file="ctree_train_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)
test_result <- predict(ctree_model,test_data)
table(true=test_data$Type,predict=test_result)
conf_matrix <- table(true=test_data$Type,predict=test_result)
ctree_test_result <-sum(diag(conf_matrix))/sum(conf_matrix)
paste(round(ctree_test_result*100,2),"%",sep = '')
write.table(conf_matrix,file="ctree_test_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)
#决策树
library(rpart)
set.seed(1213)
dtree_model <- rpart(Type~.,train_data,method = 'class')
dtree_model$cptable
# rpart.plot::rpart.plot(dtree_model)
dtree_model <- prune(dtree_model,cp=0.01)
train_result <- predict(dtree_model,train_data,type='class')
table(true=train_data$Type,predict=train_result)
conf_matrix <- table(true=train_data$Type,predict=train_result)
dtree_train_result <-sum(diag(conf_matrix))/sum(conf_matrix)

paste(round(dtree_train_result*100,2),"%",sep = '')
write.table(conf_matrix,file="dtree_train_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)
test_result <- predict(dtree_model,test_data,type='class')
table(true=test_data$Type,predict=test_result)
conf_matrix <- table(true=test_data$Type,predict=test_result)
dtree_test_result <-sum(diag(conf_matrix))/sum(conf_matrix)
paste(round(dtree_test_result*100,2),"%",sep = '')
write.table(conf_matrix,file="dtree_test_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)

#Naive Bayes（朴素贝叶斯算法分类）
library(e1071)
set.seed(1213)
nb_model <- naiveBayes(Type~.,train_data)
train_result <- predict(nb_model,train_data,type='class')
table(true=train_data$Type,predict=train_result)
conf_matrix <- table(true=train_data$Type,predict=train_result)
nb_train_result <-sum(diag(conf_matrix))/sum(conf_matrix)

paste(round(nb_train_result*100,2),"%",sep = '')
write.table(conf_matrix,file="nb_train_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)
test_result <- predict(nb_model,test_data,type='class')
table(true=test_data$Type,predict=test_result)
conf_matrix <- table(true=test_data$Type,predict=test_result)
nb_test_result <-sum(diag(conf_matrix))/sum(conf_matrix)
paste(round(nb_test_result*100,2),"%",sep = '')
write.table(conf_matrix,file="nb_train_test.txt",quote = F,sep = '\t', row.names = T, col.names = T)


#gbm
library(caret)
library(gbm)
set.seed(1213)
gbm_model <- train(Type~.,train_data,method='gbm')
train_result <- predict(gbm_model,train_data)
table(true=train_data$Type,predict=train_result)
conf_matrix <- table(true=train_data$Type,predict=train_result)
gbm_train_result <-sum(diag(conf_matrix))/sum(conf_matrix)

paste(round(gbm_train_result*100,2),"%",sep = '')
write.table(conf_matrix,file="gbm_train_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)
test_result <- predict(gbm_model,test_data)
table(true=test_data$Type,predict=test_result)
conf_matrix <- table(true=test_data$Type,predict=test_result)
gbm_test_result <-sum(diag(conf_matrix))/sum(conf_matrix)
paste(round(gbm_test_result*100,2),"%",sep = '')
write.table(conf_matrix,file="gbm_test_result.txt",quote = F,sep = '\t', row.names = T, col.names = T)





#########一些相关的分析
#ROC曲线(好像是二分类问题)来比较结果
#pca,pcoa,ndms降维
#热图展示差异
#domain的交叉验证--贡献度，交叉验证曲线

#随机取样










