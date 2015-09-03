# Определение тональности обзоров фильмов
Морозов Глеб  
30 августа 2015 г.  



В данной работе рассматриваются различные возможности применения инструментов предоставляемых языком R для определения тональности обзоров фильмов.

### Данные

Данные для работы предоставлены в рамках соревнования [Bag of Words](https://www.kaggle.com/c/word2vec-nlp-tutorial) проходящего на сайте [Kaggle](https://www.kaggle.com) и представляют собой обучающую выборку из 25000 обзоров с сайта IMBD каждый из которых отнесё с одному из классов: негативный/позитивный. Задача предсказать к какому из классов будет относится каждый обзор из тестовой выборки.


```r
library(magrittr)
library(tm)
require(plyr)
require(dplyr)
library(ggplot2)
library(randomForest)
```


Загрузим данные в оперативную память.


```r
data_train <- read.delim("labeledTrainData.tsv",header = TRUE, sep = "\t",
                               quote = "", stringsAsFactors = F)
```

Полученная таблица состоит из трёх столбцов: `id`, `sentiment` и `review`. Именно последний столбец и является объектом нашей работы. Посмотрим, что же из себя представляет сам обзор. (т.к. обзор достаточно длинный я приведу только первые 700 знаков)


```r
paste(substr(data_train[1,3],1,700),"...")
```

```
## [1] "\"With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely lik ..."
```

Видно, что в тексте присутствует мусор в виде HTML тегов. 

### Bag of Words

Bag of Words или мешок слов - это модель часто используемая при обработке текстов, представляющая собой неупорядоченный набор слов, входящих в обрабатываемый текст. Часто модель представляют в виде матрицы, в которой строки соответствуют отдельному тексту, а столбцы - входящие в него слова. Ячейки на пересечении являются числом вхождения данного слова в соответствующий документ. Данная модель удобна тем, что переводит человеческий язык слов в понятный для компьтера язык цифр. 

### Обработка данных.

Для обработки данных я буду использовать возможности пакета `tm`. В следующем блоке кода производятся следующие действия:

- создаётся вектор из текстов
- создеётся корпус - коллекция текстов
- все буквы приводятся к строчным
- удаляются знаки пунктуации
- удаляются так называемые "стоп-слова", т.к. часто встречающиеся слова в языке, не несущие сами по себе информации (в английском языке, например `and`) Кроме этого я решил сразу убрать слово, которое наверняка будет часто встречаться в обзорах, но интереса для модели не представляет - `movie`.
- производится стеммирование, т.е. слова преобразуются в свою основную форму


```r
train_corpus <- data_train$review %>% VectorSource(.)%>%
        Corpus(.) %>% tm_map(., tolower) %>% tm_map(., PlainTextDocument) %>%
        tm_map(., removePunctuation) %>% 
        tm_map(., removeWords, c("movie", stopwords("english"))) %>%
        tm_map(., stemDocument)
```

Теперь создадим частотную матрицу.


```r
frequencies <- DocumentTermMatrix(train_corpus)
frequencies
```

```
## <<DocumentTermMatrix (documents: 25000, terms: 92244)>>
## Non-/sparse entries: 2387851/2303712149
## Sparsity           : 100%
## Maximal term length: 64
## Weighting          : term frequency (tf)
```

Наша матрица содержит более 90000 терминов, т.е. модель на её основе будет содержать 90000 признаков! Её необходимо уменьшать и для этого используем тот факт, что в ней очень много редко встречающихся в обзорах слов, т.е. она разряжена (термин `sparse`). Я решил её сократить очень сильно (для того, чтобы модель уместилась в оперативной памяти с учётом 25000 объектов в обучающей выборке) и оставить только те слова, что встречаются минимум в 5% обзоров.


```r
sparse <- removeSparseTerms(frequencies, 0.95)
sparse
```

```
## <<DocumentTermMatrix (documents: 25000, terms: 373)>>
## Non-/sparse entries: 1046871/8278129
## Sparsity           : 89%
## Maximal term length: 10
## Weighting          : term frequency (tf)
```

В итоге в матрице осталось 373 термина. Преобразуем матрицу в `data frame` и добавим столбец с целевым признаком.


```r
reviewSparse = as.data.frame(as.matrix(sparse))
vocab <- names(reviewSparse)
reviewSparse$sentiment <- data_train$sentiment %>% as.factor(.) %>% 
        revalue(., c("0"="neg", "1" = "pos"))
row.names(reviewSparse) <- NULL
```

Теперь обучим Random Forest модель на полученных данных. Я использую 100 деревьев в связи с ограничением оперативной памяти.


```r
model_rf <- randomForest(sentiment ~ ., data = reviewSparse, ntree = 100)
```

Используя обученную модель создадим прогноз для тестовых данных.


```r
data_test <- read.delim("testData.tsv", header = TRUE, sep = "\t",
                               quote = "", stringsAsFactors = F)
test_corpus <- data_test$review %>% VectorSource(.)%>%
        Corpus(.) %>% tm_map(., tolower) %>% tm_map(., PlainTextDocument) %>%
        tm_map(., removePunctuation) %>% 
        tm_map(., removeWords, c("movie", stopwords("english"))) %>%
        tm_map(., stemDocument)
test_frequencies <-  DocumentTermMatrix(test_corpus,control=list(dictionary = vocab))
reviewSparse_test <-  as.data.frame(as.matrix(test_frequencies))
row.names(reviewSparse_test) <- NULL
sentiment_test <- predict(model_rf, newdata = reviewSparse_test)
pred_test <- as.data.frame(cbind(data_test$id, sentiment_test))
colnames(pred_test) <- c("id", "sentiment")
pred_test$sentiment %<>% revalue(., c("1"="0", "2" = "1"))
write.csv(pred_test, file="Submission.csv", quote=FALSE, row.names=FALSE)
```

После загрузки и оценки на сайте Kaggle модель получила оценку по статистике AUC - 0.73184. 

Попробуем подойти к проблеме с другой стороны. При составлении частотной матрицы и обрезании её мы оставляем наиболее часто встречающиеся слова, но, скорее всего, много слов, которые часто встречаются в обзорах фильмов, но не отражают настроение обзора. Например такие слова как `movie`, `film` и т.д. Но, т.к. у нас есть обучающая выборка с отмеченным настроением обзоров, можно выделить слова, частоты которых существенно различаются у негативных и положительных обзоров. 

Для начала, создадим частотную матрицу для негативных обзоров.


```r
freq_neg <- data_train %>% filter(sentiment == 0) %>% select(review) %>% VectorSource(.)%>%
        Corpus(.) %>% tm_map(., tolower) %>% tm_map(., PlainTextDocument) %>%
        tm_map(., removePunctuation) %>%
        tm_map(., removeNumbers) %>%
        tm_map(., removeWords, c(stopwords("english"))) %>%
        tm_map(., stemDocument) %>% DocumentTermMatrix(.) %>% 
        removeSparseTerms(., 0.999) %>% as.matrix(.)
freq_df_neg <- colSums(freq_neg)
freq_df_neg <- data.frame(word = names(freq_df_neg), freq = freq_df_neg)
rownames(freq_df_neg) <- NULL
head(arrange(freq_df_neg, desc(freq)))
```

```
##   word  freq
## 1 movi 27800
## 2 film 21900
## 3  one 12959
## 4 like 12001
## 5 just 10539
## 6 make  7846
```

И для положительных обзоров.


```r
freq_pos <- data_train %>% filter(sentiment == 1) %>% select(review) %>% VectorSource(.)%>%
        Corpus(.) %>% tm_map(., tolower) %>% tm_map(., PlainTextDocument) %>%
        tm_map(., removePunctuation) %>%
        tm_map(., removeNumbers) %>%
        tm_map(., removeWords, c(stopwords("english"))) %>%
        tm_map(., stemDocument) %>% DocumentTermMatrix(.) %>% 
        removeSparseTerms(., 0.999) %>% as.matrix(.)
freq_df_pos <- colSums(freq_pos)
freq_df_pos <- data.frame(word = names(freq_df_pos), freq = freq_df_pos)
rownames(freq_df_pos) <- NULL 
head(arrange(freq_df_pos, desc(freq)))
```

```
##   word  freq
## 1 film 24398
## 2 movi 21796
## 3  one 13706
## 4 like 10138
## 5 time  7889
## 6 good  7508
```

Объединим полученные таблицы и посчитаем разницу между частотами.


```r
freq_all <- merge(freq_df_neg, freq_df_pos, by = "word", all = T)
freq_all$freq.x[is.na(freq_all$freq.x)] <- 0
freq_all$freq.y[is.na(freq_all$freq.y)] <- 0
freq_all$diff <- abs(freq_all$freq.x - freq_all$freq.y)
head(arrange(freq_all, desc(diff)))
```

```
##    word freq.x freq.y diff
## 1  movi  27800  21796 6004
## 2   bad   7660   1931 5729
## 3 great   2692   6459 3767
## 4  just  10539   7109 3430
## 5  love   2767   5988 3221
## 6  even   7707   5056 2651
```

Отлично! Мы видим, как и ожидалось, среди слов с наибольшей разницей такие термины как `bad`, `great` и `love`. Но также здесь и просто часто встречающиеся слова, как `movie`. Это произошло, что у частых слов даже небольшая процентная разница выдаёт высокую абсолютную разницу. Для того, чтобы устранить это упущение, нормализуем разницу, разделив её на сумму частот. Получившаяся метрика будет лежать в интервале между 0 и 1, и чем выше её значение - тем важнее данное значение в определении разницы между положительными и отрицательными отзывами. Но что же делать со словами, которые встречаются только у одного класса отзывов и при этом их частота мала? Для уменьшения их важности добавим к знаменателю коэффициент.


```r
freq_all$diff_norm <- abs(freq_all$freq.x - freq_all$freq.y)/
        (freq_all$freq.x +freq_all$freq.y + 300)
head(arrange(freq_all, desc(diff_norm)))
```

```
##      word freq.x freq.y diff diff_norm
## 1   worst   2436    246 2190 0.7344064
## 2    wast   1996    192 1804 0.7250804
## 3 horribl   1189    194  995 0.5912062
## 4  stupid   1525    293 1232 0.5816808
## 5     bad   7660   1931 5729 0.5792134
## 6    wors   1183    207  976 0.5775148
```

Отберём 500 слов с наивысшим показателем коэффициента разницы.


```r
freq_word <- arrange(freq_all, desc(diff_norm)) %>% select(word) %>% slice(1:500)
```

Используем полученный словарь для создания частотной матрицы, на которой обучим Random Forest модель.


```r
vocab <- as.character(freq_word$word)
frequencies = DocumentTermMatrix(train_corpus,control=list(dictionary = vocab))
reviewSparse_train <-  as.data.frame(as.matrix(frequencies))
row.names(reviewSparse_train) <- NULL
reviewSparse_train$sentiment <- data_train$sentiment %>% as.factor(.) %>%
revalue(., c("0"="neg", "1" = "pos"))

model_rf <- randomForest(sentiment ~ ., data = reviewSparse_train, ntree = 100)
```

После загрузки и оценки на сайте Kaggle модель получила оценку по статистике AUC - 0.83120, т.е. поработав с признаками мы получили улучшение статистики на 10%!

### TF-IDF

При создании матрицы документ-термин в качестве метрики важности слова мы использовали просто частоту появления слова в обзоре. В пакете `tm` есть возможность использовать другую меру, называемую tf-idf. TF-IDF (от англ. TF — term frequency, IDF — inverse document frequency) — статистическая метрика, используемая для оценки важности слова в контексте документа, являющегося частью коллекции документов или корпуса. Вес некоторого слова пропорционален количеству употребления этого слова в документе, и обратно пропорционален частоте употребления слова в других документах коллекции. 

Используя tf-idf, создадим словарь из 500 терминов с наиболее высоким показателем данной метрики. Для того, чтобы этот словарь наиболее релевантно отражал важность слов, будем использовать дополнительную обучающую выборку, в которой не размечено настроение обзоров. На базе полученного словаря создадим матрицу документ-термин и обучим модель.


```r
data_train_un <- read.delim("unlabeledTrainData.tsv",header = TRUE, sep = "\t",
                            quote = "", stringsAsFactors = F)
train_review <- c(data_train$review, data_train_un$review)
train_corpus <- train_review %>% VectorSource(.)%>%
        Corpus(.) %>% tm_map(., tolower) %>% tm_map(., PlainTextDocument) %>%
        tm_map(., removePunctuation) %>% tm_map(., removeNumbers) %>%
        tm_map(., removeWords, c(stopwords("english"))) %>%
        tm_map(., stemDocument)
tdm <- TermDocumentMatrix(train_corpus,
                          control = list(weighting = function(x) weightTfIdf(x, normalize = F)))
library(slam)
freq <- rollup(tdm, 2,FUN = sum)
freq <- as.matrix(freq)
freq_df <- data.frame(word = row.names(freq), tfidf = freq)
names(freq_df) <- c("word", "tf_idf")
row.names(freq_df) <- NULL
freq_df %<>% arrange(desc(tf_idf))
vocab <- as.character(freq_df$word)[1:500]
train_corpus <- data_train$review %>% VectorSource(.)%>%
        Corpus(.) %>% tm_map(., tolower) %>% tm_map(., PlainTextDocument) %>%
        tm_map(., removePunctuation) %>% tm_map(., removeNumbers) %>%
        tm_map(., removeWords, c(stopwords("english"))) %>%
        tm_map(., stemDocument)
frequencies = DocumentTermMatrix(train_corpus,control=list(dictionary = vocab,
                                                           weighting = function(x) weightTfIdf(x, normalize = F) ))
reviewSparse_train <-  as.data.frame(as.matrix(frequencies))
rm(data_train_un, tdm, dtm, train_review)
reviewSparse_train <-  as.data.frame(as.matrix(frequencies))
row.names(reviewSparse_train) <- NULL
colnames(reviewSparse_train) = make.names(colnames(reviewSparse_train))
reviewSparse_train$sentiment <- data_train$sentiment %>% as.factor(.) %>%
revalue(., c("0"="neg", "1" = "pos"))
rm(data_train, train_corpus, freq, freq_df)
model_rf <- randomForest(sentiment ~ ., data = reviewSparse_train, ntree = 100)
```

Используем данную модель на тестовой выборке и получим значение AUC - 0.81584.

### Заключение.

Данная работа представляет собой один из возможных вариантов создания предсказательной модели на основе текстовых данных. Одним из вариантов улучшить качество модели может быть увеличение количества используемых терминов из матрицы документ-термин, но этот путь требует существенного увеличения используемых машинных ресурсов. Также может привести к гораздо лучшим результатам обратиться не к частотам слов, а к их значениям и связям между ними. Для этого надо обратиться к модели `word2vec`. Кроме этого, большое поле для исследования представляет собой рассмотрение терминов в контексте документа.



