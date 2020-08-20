################### Packages required ############

require(caret)

install.packages("tm")
require(tm)#### text mining package
install.packages("wordcloud")
require(wordcloud)
require(e1071)
install.packages("MLmetrics")
require(MLmetrics)


########### Reading and understanding the data ##########

rawData <-  read.csv(file.choose())
View(rawData)

str(rawData)

colnames(rawData)

#Converting the text to utf-8 format
rawData$text <- iconv(rawData$text, to = "utf-8")
View(rawData$text)


#Type as factor
rawData$type <- factor(rawData$type)


summary(rawData)

table(rawData$type)

prop.table(table(rawData$type)) * 100##### % of ham and spam






########## Splitting of data into test and train ##########

set.seed(1234)

trainIndex <- createDataPartition(rawData$type, p = .75, 
                                  list = FALSE, 
                                  times = 1)
trainData <- rawData[trainIndex,]
nrow(trainData)
testData <- rawData[-trainIndex,]
nrow(testData)


prop.table(table(trainData$type)) * 100### % of ham and spam in train dataset


prop.table(table(testData$type)) * 100## % of ham and spam in test dataset



######## Exploratory Data analysis ###########

trainData_ham <- trainData[trainData$type == "ham",]
head(trainData_ham$text)


tail(trainData_ham$text)


trainData_spam <- trainData[trainData$type == "spam",]
head(trainData_spam$text)


tail(trainData_spam$text)


trainData_spam <- NULL
trainData_ham <- NULL


############### Data Cleaning ##############
#steps and approach
#reduce all messages to lowe case
#remove numbers
#remove stopwords
#remove punctuations
#normalize whitespeces

#create the corpus
corpus <- Corpus(VectorSource(trainData$text))

#basic info about the corpus
print(corpus)


#Inspect 4 documents
corpus[[1]]$content

corpus[[2]]$content

corpus[[50]]$content

corpus[[100]]$content


#normalize to lowercase (not a standard tm transformation)
corpus <- tm_map(corpus, content_transformer(tolower))

#remove numbers
corpus <- tm_map(corpus, removeNumbers)

#remove stopwords e.g. to, and, but, or (using predefined set of word in tm package)
corpus <- tm_map(corpus, removeWords, stopwords())

#remove punctuation
corpus <- tm_map(corpus, removePunctuation)

# normalize whitespaces
corpus <- tm_map(corpus, stripWhitespace)

#Inspect the same 4 documents to visualize how the documents have been
#transformed
corpus[[1]]$content

corpus[[2]]$content

corpus[[50]]$content

corpus[[100]]$content



############ Visual analysis of (Ham Vs spam) ####################



pal1 <- brewer.pal(9,"YlGn")

pal1 <- pal1[-(1:4)]

pal2 <- brewer.pal(9,"Reds")
pal2 <- pal2[-(1:4)]

#min.freq initial settings -> around 10% of the number of docs in the corpus (40 times)
par(mfrow = c(1,2))
wordcloud(corpus[trainData$type == "ham"], min.freq = 40, random.order = FALSE, colors = pal1)
wordcloud(corpus[trainData$type == "spam"], min.freq = 40, random.order = FALSE, colors = pal2)


######### Transforming the data ###########

sms_dtm <- DocumentTermMatrix(corpus, control = list(global = c(2, Inf)))
print(sms_dtm)
inspect(sms_dtm[1:10, 5:13])

sms_features <- findFreqTerms(sms_dtm, 5) #find words that appears at least 5 times
summary(sms_features)

head(sms_features)


sms_dtm_train <- DocumentTermMatrix(corpus, list(global = c(2, Inf), dictionary = sms_features))
print(sms_dtm_train)


convert_counts <- function(x){
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0,1), labels = c("No", "Yes"))
  return (x)
}
sms_dtm_train <- apply(sms_dtm_train, MARGIN = 2, convert_counts)

head(sms_dtm_train[,1:5])


############## training the model ################


sms_classifier <- naiveBayes(sms_dtm_train, trainData$type)

sms_classifier[[2]][1:5]



## Evalute the model

corpus <- Corpus(VectorSource(testData$text))# create


# normalize to lowercase 
corpus <- tm_map(corpus, content_transformer(tolower))

# remove numbers
corpus <- tm_map(corpus, removeNumbers)

# remove stopwords e.g. to, and, but, or (using predefined set of word in tm package)
corpus <- tm_map(corpus, removeWords, stopwords())

# remove punctuation
corpus <- tm_map(corpus, removePunctuation)

# normalize whitespaces
corpus <- tm_map(corpus, stripWhitespace)


sms_dtm_test <- DocumentTermMatrix(corpus, list(global = c(2, Inf), dictionary = sms_features))
print(sms_dtm_test)


sms_dtm_test <- apply(sms_dtm_test, MARGIN = 2, convert_counts)
sms_dtm_test[1:10, 5:12]




##### Evalute the model #######

sms_test_pred <- predict(sms_classifier, sms_dtm_test)

#table actual (row) vs. predicted (col): confusion matrix
table(testData$type, sms_test_pred)


ConfusionMatrix(sms_test_pred, testData$type)


Accuracy(sms_test_pred, testData$type)
#[1] 0.9784017

F1_Score(sms_test_pred, testData$type)
#[1] 0.987634
