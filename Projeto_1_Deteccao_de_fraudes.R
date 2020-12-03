##################################################################
#### Projeto 1 - Detecção de fraudes - TalkingData AdTracking ####
##################################################################

setwd("C:/DataScience/FCD/BigDataAnalytics-R-Azure/Projeto-1/")
getwd()

############################################################
#### Carregamento de dados e bibliotecas ###################
############################################################

library(dplyr)
library(ggplot2)
library(randomForest)
library(caret)
library(data.table)
library(e1071)

# Utilizando para testes e desenvolvimento na máquina local o arquivo train_sample.csv
# por motivos de menor utilização de memória da máquina. Na versão para produção, será utilizado
# o arquivo completo train.csv

dados<- read.csv("train_sample.csv")

head(dados)


##############################################
#### Análise Exploratória dos Dados ##########
##############################################

# Limpeza e manipulação

# Verificando a existência de NAs no arquivo
variaveis <- names(dados)


for(v in variaveis){
  print('-------------')
  print(paste("Número de NAs na variável", v))
  print(sum(is.na(dados[v])))
}

str(dados)

dados_filt <- dados %>% filter(attributed_time != '' & is_attributed != 1)
head(dados_filt)

dados %>%
  group_by(is_attributed) %>%
  summarise(Vazio = sum(attributed_time == ''),
            Alguma_coisa = sum(attributed_time != ''))


dados$attributed_time <- NULL
rm(dados_filt)

# Tratando a variável click_time: pode-se avaliar o tempo tanto como uma variável contínua,
# no sentido de se analisar uma tendência de aumento do número de fraudes ao longo do tempo
# como também de forma categorizada (por mês, dia do mês, dia da semana, hora do dia, etc.)
# para analisar possíveis sazonalidades no número de fraudes ao longo do ano.

# Para isto será criada uma variável click_time_posixct, que é a conversão da variável
# click_time para o formato POSIXct. Poderão ser criadas variáveis como rel_click_time,
# que será o tempo relativo entre o menor tempo do dataset e cada tempo registrado,
# e também variáveis categóricas como year, month, week, weekday, hour, minute, second. 
# A variável original click_time será eliminada do dataset.

dados$click_time_posixct <- as.POSIXct(dados$click_time)
dados$click_time <- NULL

t_1st <- min(dados$click_time_posixct)
t_last <- max(dados$click_time_posixct)

t_last - t_1st

# Como o tempo decorrido entre o primeiro e o último clique é de apenas 3 dias, não parece
# fazer sentido um aumento do número de fraudes significativo ao longo desse curto intervalo de
# tempo. De modo que as variáveis  year, month, week, weekday não serão criadas para poupar
# esforço computacional. Serão criadas, então, apenas as variáveis hour, minute, second.
# Também para reduzir esforço computacional, será eliminada do dataset de treino a variável
# click_time_posixct.

dados$hour <- hour(dados$click_time_posixct)
dados$minute <- minute(dados$click_time_posixct)
dados$second <- second(dados$click_time_posixct)
dados$click_time_posixct <- NULL
str(dados)
summary(dados)


# Inspeção gráfica de relações entre os atributos e o label.

ggplot(dados, aes(is_attributed)) + geom_bar(fill = "blue", alpha = 0.5) +
  ggtitle("Totais de is_attributed por categoria")

table(dados$is_attributed)

variaveis <- names(dados)
variaveis <- variaveis[variaveis != "is_attributed"]


lapply(variaveis, function(x){
    ggplot(dados, aes_string(x)) +
      geom_histogram(fill = "blue", alpha = 0.5) +
      ggtitle(paste("Histograma de",x)) +
      facet_grid(dados$is_attributed ~ .)})


# Balanceando o dataset
# Criando um arquivo CSV para carregar no Azure ML
write.csv(dados, "dados_para_balancear.csv", row.names = FALSE)
# Lendo um arquivo CSV retornado pelo Azure ML
dados <- read.csv("dados_balanceados.csv")
str(dados)
# Retornando o label para o tipo fator
dados$is_attributed <- factor(dados$is_attributed)

################################################################
######### Alternativa ao Azure ML ##############################
##### Implementar na versão 2.0: utilizar o SMOTE no pacote DMwR
################################################################

# Inspeção gráfica após o balanceamento

ggplot(dados, aes(is_attributed)) + geom_bar(fill = "blue", alpha = 0.5) +
  ggtitle("Totais de is_attributed por categoria após balanceamento")

table(dados$is_attributed)

variaveis <- names(dados)
variaveis <- variaveis[variaveis != "is_attributed"]


lapply(variaveis, function(x){
  ggplot(dados, aes_string(x)) +
    geom_histogram(fill = "blue", alpha = 0.5) +
    ggtitle(paste("Histograma de",x,"após balanceamento")) +
    facet_grid(dados$is_attributed ~ .)})

## Criando um dataset categorizado para posterior teste dos modelos
variaveis <- names(dados)
dados_cat <- dados

for(v in variaveis){
  dados_cat[[v]] <- factor(dados_cat[[v]])
}
str(dados_cat)

dados_cat <- dados
variaveis <- names(dados)
variaveis <- variaveis[!(variaveis %in% c("is_attributed", "hour"))]

for(v in variaveis){
  dados_cat[[v]] <- cut(dados[[v]], 53)
}

# Categorizando também o atributo hour, que tem menos que 53 níveis
dados_cat$hour <- factor(dados_cat$hour)

str(dados_cat)
str(dados)



# Fazendo o split data entre dados de treino e dados de validação, uma vez que os dados
# de teste não tem label disponível.

split <- createDataPartition(y = dados$is_attributed, p = 0.7, list = FALSE)
dados_treino <- dados[split, ]
dados_valid <- dados[-split, ]

dados_cat_treino <- dados_cat[split, ]
dados_cat_valid <- dados_cat[-split, ]


## Feature selection

# Função para seleção de variáveis
#run.feature.selection <- function(num.iters=20, feature.vars, class.var){
#  set.seed(10)
#  variable.sizes <- 1:10
#  control <- rfeControl(functions = rfFuncs, method = "cv", 
#                        verbose = FALSE, returnResamp = "all", 
#                        number = num.iters)
#  results.rfe <- rfe(x = feature.vars, y = class.var, 
#                     sizes = variable.sizes, 
#                     rfeControl = control)
#  return(results.rfe)
#}
#head(dados)
# Executando a função para o dataset não-categorizado
#rfe.results <- run.feature.selection(feature.vars = dados[,-6], 
#                                     class.var = dados[,6])


# Visualizando os resultados
#rfe.results
#varImp((rfe.results))

## Feature selection
#formula <- "is_attributed ~ ."
#formula <- as.formula(formula)
#control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
#model <- train(formula, data = data_train, method = "glm", trControl = control)
#importance <- varImp(model, scale = FALSE)
#plot(importance)



###################################################
#### Primeiro modelo de Machine Learning ##########
###################################################
# Algoritmo: randomForest
# Atributos: não-categorizados

modelo_rf1 <- randomForest(is_attributed ~ .,
                           data = dados_treino,
                           ntree = 100, nodesize = 10, importance = T)

previsao_rf1 <- predict(modelo_rf1, dados_valid, type = 'class')


# Matriz de confusão
confusionMatrix(previsao_rf1, dados_valid$is_attributed)



###################################################
#### Segundo modelo de Machine Learning ###########
###################################################
# Algoritmo: randomForest
# Atributos: categorizados

modelo_rf2 <- randomForest(is_attributed ~ .,
                           data = dados_cat_treino,
                           ntree = 100, nodesize = 10, importance = T)

previsao_rf2 <- predict(modelo_rf2, dados_cat_valid, type = 'class')



# Matriz de confusão
confusionMatrix(previsao_rf2, dados_cat_valid$is_attributed)



###################################################
#### Terceiro modelo de Machine Learning ##########
###################################################
# Algoritmo: SVM
# Atributos: não-categorizados

modelo_svm1 <- svm(is_attributed ~ .,
                           data = dados_treino,
                            type = 'C-classification',
                            kernel = 'radial')

previsao_svm1 <- predict(modelo_svm1, dados_valid, type = 'class')


# Matriz de confusão
confusionMatrix(previsao_svm1, dados_valid$is_attributed)


###################################################
#### Quarto modelo de Machine Learning ############
###################################################
# Algoritmo: SVM
# Atributos: categorizados

modelo_svm2 <- svm(is_attributed ~ .,
                   data = dados_cat_treino,
                   type = 'C-classification',
                   kernel = 'radial')

previsao_svm2 <- predict(modelo_svm2, dados_cat_valid, type = 'class')


# Matriz de confusão
confusionMatrix(previsao_svm2, dados_cat_valid$is_attributed)


###################################################
#### Quinto modelo de Machine Learning ############
###################################################
# Algoritmo: Naive-Bayes
# Atributos: não-categorizados

modelo_nb1 <- naiveBayes(dados[, -6], dados[, 6])


previsao_nb1 <- predict(modelo_nb1, dados_valid, type = 'class')


# Matriz de confusão
confusionMatrix(previsao_svm1, dados_valid$is_attributed)


###################################################
#### Sexto modelo de Machine Learning #############
###################################################
# Algoritmo: Naive-Bayes
# Atributos: categorizados

modelo_nb2 <- naiveBayes(dados_cat[, -6], dados[, 6])


previsao_nb2 <- predict(modelo_nb2, dados_cat_valid, type = 'class')


# Matriz de confusão
confusionMatrix(previsao_svm1, dados_valid$is_attributed)


###################################################
#### Predição do arquivo test.csv #################
###################################################

# Melhor modelo: modelo_rf1

test <- as.data.frame(fread("test.csv"))
submission <- as.data.frame(fread("sample_submission.csv"))
head(test)
head(submission)
str(test)
str(submission)

# Para adequar o dataset de teste ao dataset de treinamento do modelo 1
# é necessário excluir a variável click_id e formatar a variável click_time
test$click_id <- NULL

test$click_time_posixct <- as.POSIXct(test$click_time)
test$click_time <- NULL


test$hour <- hour(test$click_time_posixct)
test$minute <- minute(test$click_time_posixct)
test$second <- second(test$click_time_posixct)
test$click_time_posixct <- NULL
str(test)

previsao_final <- predict(modelo_rf1, test)
table(previsao_final)

# Gravando o arquivo submission para envio para o Kaggle
submission$is_attributed <- previsao_final
fwrite(submission, "submission.csv")
