#ST3189 Machine Learning
#Prepared by: CHEN, PIN-SYUE
#Student ID: 200618629

#------------------------------------------------------------------------------#
#------------- Unsupervised Learning (World Happiness Report) -----------------#
#------------------------------------------------------------------------------#

#Data Link: https://www.kaggle.com/datasets/unsdsn/world-happiness

#Import Package
library(maps)
library(dplyr)
library(ggplot2)
library(mapdata)
library(ggcorrplot)

#Import Modelling Package
library(cluster)
library(FactoMineR)

#Load Data
setwd("/Users/dennis/Desktop/ML Coursework/World-Happiness")
happiness_rawdata <- read.csv("2015.csv", stringsAsFactors = F)

#Explore Data
summary(happiness_rawdata)

#----------------------------- DATA PREPARATION -------------------------------#

#Rename Columns into same format
happiness_rawdata <- happiness_rawdata %>% 
  rename("coun" = "Country",
         "regi" = "Region",
         "hapr" = "Happiness.Rank",
         "haps" = "Happiness.Score",
         "ster" = "Standard.Error",
         "GDP" = "Economy..GDP.per.Capita.",
         "fami" = "Family",
         "liex" = "Health..Life.Expectancy.",
         "free" = "Freedom",
         "gove" = "Trust..Government.Corruption.",
         "gene"  = "Generosity",
         "dyre" = "Dystopia.Residual")

#------------------------------- DATA PLOTTING --------------------------------#

#Correlation plot
pairs(~ ., panel = panel.smooth, data = subset(happiness_rawdata, select = -c(coun,regi) ), main = "scatterplot matrix")

ggcorrplot(cor(subset(happiness_rawdata, select = -c(coun,regi,hapr,ster,dyre) )), hc.order = TRUE, 
           title = "Correlation Matrix between variables",
           type = "lower", lab = TRUE, 
           lab_size = 3,
           colors = c("#6D9EC1", "white", "#E46726"))

#Happiness score by region
#Bar Chart
Bar_hapsregi <- happiness_rawdata %>% group_by(regi) %>% summarize(avg_haps = mean(haps))
Bar_hapsregi$regi <- factor(Bar_hapsregi$regi, levels = Bar_hapsregi$regi[order(Bar_hapsregi$avg_haps, decreasing = TRUE)])
ggplot(Bar_hapsregi, aes(x = regi, y = avg_haps, fill = regi)) + 
  geom_bar(stat = "identity") +
  xlab("Region") +
  ylab("Average Happiness Score") +
  ggtitle("Happiness Score by Region") +
  theme(axis.text.x = element_blank())

#World Map
map_data <- map_data("world")
setdiff(unique(happiness_rawdata$coun), unique(map_data("world")$region))
happiness_rawdata <- happiness_rawdata %>%
  mutate(coun = case_when(
    coun == "United States" ~ "USA",
    coun == "United Kingdom" ~ "UK",
    coun == "Trinidad and Tobago" ~ "Trinidad",
    coun == "North Cyprus" ~ "Cyprus",
    coun == "Somaliland region" ~ "Somalia",
    coun == "Macedonia" ~ "North Macedonia",
    coun == "Palestinian Territories" ~ "Palestine",
    coun == "Congo (Kinshasa)" ~ "Democratic Republic of the Congo",
    coun == "Congo (Brazzaville)" ~ "Republic of Congo",
    TRUE ~ coun))
ggplot(happiness_rawdata, aes(map_id = coun)) +
  geom_map(aes(fill = haps), map = map_data("world")) +
  expand_limits(x = map_data("world")$long, y = map_data("world")$lat) +
  scale_fill_gradient(low = "red", high = "green", name = "Happiness Level") +
  theme_void()

#Box Plot
Box_hapsregi <- happiness_rawdata %>% group_by(regi) %>% summarize(box_avg_haps = mean(haps))
happiness_rawdata <- merge(happiness_rawdata, Box_hapsregi, by = "regi")
happiness_rawdata$regi <- reorder(happiness_rawdata$regi, desc(happiness_rawdata$box_avg_haps))
ggplot(happiness_rawdata, aes(x = regi, y = haps, fill = regi)) +
  geom_boxplot() +
  xlab("Region") +
  ylab("Happiness Score") +
  ggtitle("Happiness Score by Region") +
  theme(axis.text.x = element_blank())

# Relationship between Life Expectancy and Happiness Score by Region
filtered_data <- happiness_rawdata %>% 
  filter(regi != "Australia and New Zealand", regi != "North America")
ggplot(filtered_data, aes(x = GDP, y = haps, color = regi)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ regi, scales = "free") +
  xlab("GDP") +
  ylab("Happiness Score") +
  ggtitle("Happiness vs. GDP by Regions")

#---------------------------------- Modeling ----------------------------------#

#PCA
happiness_data <- happiness_rawdata[, c("GDP", "fami", "liex", "free", "gove", "gene")]
pc <- prcomp(happiness_data,scale.=T)
screeplot(pc, type = "line", main = "Scree Plot")
summary(pc)
pc$rotation

loadings <- pc$rotation[, 1:2]
scores <- as.data.frame(pc$x[, 1:2])
biplot(pc, cex = 0.8)

#K-means Clustering
kmeans_model <- kmeans(scale(happiness_data), centers = 6, nstart = 25)
kmeans_model$centers

set.seed(123)
kmeans_result <- kmeans(pc$x[, 1:2], centers = 6)
kmeans_data <- data.frame(PC1 = pc$x[, 1], PC2 = pc$x[, 2], Cluster = factor(kmeans_result$cluster))
ggplot(kmeans_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  scale_color_discrete(name = "Cluster")

clustered_data <- data.frame(Country = happiness_rawdata$coun, Cluster = factor(kmeans_result$cluster))
category_list <- aggregate(Country ~ Cluster, clustered_data, function(x) paste(x, collapse = ", "))
for (i in 1:nrow(category_list)) {cat(paste0("Cluster ", i, ": ", category_list$Country[i], "\n\n"))}

#Hierarchical Clustering
hc <- hclust(dist(scale(happiness_data)), method = "ward.D2")
plot(hc, labels = happiness_rawdata$coun, hang = -1, cex = 0.65)



#------------------------------------------------------------------------------#
#---------------------- Regression (Life Expectancy) --------------------------#
#------------------------------------------------------------------------------#

#Data Link: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

#Import Package
library(car)
library(dplyr)
library(ggpubr)
library(ggplot2)
library(corrplot)
library(QuantPsyc)
library(ggcorrplot)
library(countrycode)

#Import Modelling Package
library(rpart)
library(glmnet)
library(randomForest)

#Load Data
setwd("/Users/dennis/Desktop/ML Coursework")
life_rawdata <- read.csv("Life Expectancy Data.csv", stringsAsFactors = F)

#Explore Data
summary(life_rawdata)

#------------------------------- DATA CLEANING --------------------------------#

#Rename Columns into same format
life_rawdata <- life_rawdata %>% 
  rename("coun" = "Country",
         "year" = "Year",
         "stat" = "Status",
         "LIEX" = "Life.expectancy",
         "admo" = "Adult.Mortality",
         "inde" = "infant.deaths",
         "alco" = "Alcohol",
         "peex" = "percentage.expenditure",
         "heab" = "Hepatitis.B",
         "meas" = "Measles",
         "BMI"  = "BMI",
         "u5de" = "under.five.deaths",
         "poli" = "Polio",
         "toex" = "Total.expenditure", 
         "diph" = "Diphtheria",
         "HIAI" = "HIV.AIDS",
         "GDP"  = "GDP",
         "popu" = "Population",
         "thi1" = "thinness..1.19.years",
         "thi5" = "thinness.5.9.years",
         "icor" = "Income.composition.of.resources",
         "scho" = "Schooling")

#Check Duplicate data
duplicate_data <- life_rawdata[duplicated(life_rawdata),]

#Modify Wrong Status Data
life_rawdata$stat[life_rawdata$coun == "Canada"] <- "Developed"
life_rawdata$stat[life_rawdata$coun == "Greece"] <- "Developed"
life_rawdata$stat[life_rawdata$coun == "France"] <- "Developed"
life_rawdata$stat[life_rawdata$coun == "Finland"] <-"Developed"

#Fill NA and Missing Value with mean based on Country Mean
#Formulate Function
Na_coun <- function(data, term) {
  country_avg <- data %>%
    group_by(coun) %>%
    summarize(avg = mean(!!sym(term), na.rm = TRUE))
  
  data <- left_join(data, country_avg, by = "coun")
  data[[term]][is.na(data[[term]])] <- data$avg[is.na(data[[term]])]
  data$avg <- NULL
  return(data)}

#Apply Function To The Columns With NA or Missing Value
colums <- c("LIEX","admo","alco","heab","BMI","poli","toex","diph","GDP","popu","thi1","thi5","icor","scho")
for (term in colums) {life_rawdata <- Na_coun(life_rawdata, term)}

#Using "library(countrycode)" To Classify The Region By Country
life_rawdata$region <- countrycode(sourcevar = life_rawdata$coun, origin = "country.name", destination = "region")

#Fill NA and Missing Value with mean based on Region Mean
#Formulate Function
Na_region <- function(data, term) {
  region_avg <- data %>%
    group_by(region) %>%
    summarize(avg = mean(!!sym(term), na.rm = TRUE))
  
  data <- left_join(data, region_avg, by = "region")
  data[[term]][is.na(data[[term]])] <- data$avg[is.na(data[[term]])]
  data$avg <- NULL
  return(data)}

#Apply Function To The Remaining Columns With NA or Missing Value
colum <- c("LIEX","admo","alco","heab","BMI","poli","toex","GDP","popu","thi1","thi5","icor","scho")
for (term in colum) {life_rawdata <- Na_region(life_rawdata, term)}

#Check The NA With All Columns
summary(life_rawdata)

#------------------------------- DATA PLOTTING --------------------------------#

#Correlation Between Columns
ggcorrplot(cor(subset(life_rawdata, select = -c(coun,stat,region) )), hc.order = TRUE, 
           title = "Correlation Matrix between variables",
           type = "lower", lab = TRUE, 
           lab_size = 3,
           colors = c("#6D9EC1", "white", "#E46726"))

#Developed VS Developing (Life.expectancy, infant.deaths, Adult.Mortality)
life_rawdata %>% 
  group_by(stat) %>% 
  summarize(count = n(),
            avg_Life.expectancy = mean(LIEX),
            avg_infant.deaths = mean(inde),
            avg_Adult.Mortality = mean(admo))

#Developed VS Developing (Life.expectancy, infant.deaths, Adult.Mortality) With Year
avgLIEX <- life_rawdata %>%
  group_by(year, stat) %>%
  summarize(avg_LIEX = mean(LIEX),
            avg_inde = mean(inde),
            avg_admo = mean(admo))

ggplot(avgLIEX, aes(x = year, y = avg_LIEX, group = stat, color = stat)) +
  geom_line(size = 2) +
  labs(title = "Life Expectancy Over Time",
       x = "Years",
       y = "Average Life Expectancy") +
  theme(legend.position = "top")

ggplot(avgLIEX, aes(x = year, y = avg_inde, group = stat, color = stat)) +
  geom_line(size = 2) +
  labs(title = "Infant.deaths Over Time",
       x = "Years",
       y = "Infant.deaths") +
  theme(legend.position = "top")

ggplot(avgLIEX, aes(x = year, y = avg_admo, group = stat, color = stat)) +
  geom_line(size = 2) +
  labs(title = "Adult.Mortality Over Time",
       x = "Years",
       y = "Adult.Mortality") +
  theme(legend.position = "top")

#The Relationship Between Life Expectancy And (years of Schooling, GDP, Adult Mortality, Income.composition.of.resources, Death by HIV.AIDS)
data_visualise <- subset(life_rawdata, icor > 0)
plot_cols <- c("scho", "GDP", "admo","icor","HIAI")
plot_titles <- c("Years of Schooling", "GDP", "Adult Mortality", "Income.composition.of.resources", "Death by HIV.AIDS")

for (i in seq_along(plot_cols)) {
  plot <- ggplot(data_visualise, aes_string(x = plot_cols[i], y = "LIEX", color = "stat")) +
    geom_point(size = 1.5) +
    theme(legend.position = "top") +
    labs(x = plot_titles[i], y = "Life Expectancy", title = paste("Life Expectancy vs.", plot_titles[i]))
  print(plot)}

#Schooling Impact Life Expectancy
sch_daata <- subset(life_rawdata, icor > 0)
corr <- cor(life_rawdata$LIEX, life_rawdata$scho)
ggscatter(sch_daata, x = "scho", y = "LIEX", color = "stat",
          add = "reg.line", 
          cor.coef = TRUE, cor.method = "pearson", cor.coef.size = 5,) +
  geom_smooth(method = "lm", se = FALSE, color = "brown") +
  ggtitle("Life Expectancy VS Years of Schooling") +
  labs(x = "Years of Schooling", y = "Life expectancy", 
       caption = paste("Correlation coefficient = ", round(corr, 2)))+
  theme(legend.position = "top")

# Relationship Between under.five.death And Several Health Condition
pairs(u5de ~ heab+meas+poli+diph+thi5, panel = panel.smooth, data = life_rawdata, main = "Age under-five death")

#---------------------------------- Modeling ----------------------------------#

#Create Dummy Variables (Country and Status)
country_dummies <- model.matrix(~coun, life_rawdata)[,-1]
status_dummies <- model.matrix(~stat, life_rawdata)[,-1]
data_final <- cbind(life_rawdata[,c("LIEX","year","admo","inde","alco","peex","heab","meas","BMI","u5de","poli","toex","diph","HIAI","GDP","popu","thi1","thi5","icor","scho")], country_dummies, status_dummies)

#Set Up Training set and Test set
set.seed(520)
trainIndex <- sample(1:nrow(data_final), 0.7*nrow(data_final))
trainData <- data_final[trainIndex,]
testData <- data_final[-trainIndex,]


#Linear regression model
linear_model <- lm(LIEX ~ ., data = trainData)
predictions <- predict(linear_model, newdata = testData)

#RMSE
lin_rmse <- sqrt(mean((testData$LIEX - predictions)^2))

#R-Squared
lin_rsq <- summary(linear_model)$r.squared

summary(linear_model)


#Ridge Regression
ridge_model <- glmnet(X <- model.matrix(LIEX~., data = trainData), 
                      y = trainData$LIEX, alpha = 0, lambda = 0.1)

ridge_pred <- predict(ridge_model, newx = model.matrix(LIEX ~ ., data = testData))

#RMSE
ridge_rmse <- sqrt(mean((ridge_pred - testData$LIEX)^2))
#R-Squared
ridge_rsq <- 1 - sum((testData$LIEX - ridge_pred)^2) / sum((testData$LIEX - mean(testData$LIEX))^2)


#Lasso regression
independent <- model.matrix(LIEX~year+admo+inde+alco+peex+heab+meas+BMI+u5de+poli+toex+diph+HIAI+GDP+popu+thi1+thi5+icor+scho+status_dummies + country_dummies, data = life_rawdata)

dependent <- life_rawdata$LIEX

lasso_mod <- glmnet(independent, dependent, alpha = 1)
lasso_pred <- predict(lasso_mod, independent)

#R-squared
lasso_rsq <- lasso_mod$dev.ratio
lasso_rsq <- mean(lasso_rsq)
#RMSE
lasso_rmse <- sqrt(mean((dependent - lasso_pred)^2))


#Random Forest
rf_model <- randomForest(LIEX~year+admo+inde+alco+peex+heab+meas+BMI+u5de+poli+toex+diph+HIAI+GDP+popu+thi1+thi5+icor+scho, data = trainData)
rf_pred <- predict(rf_model, testData)

#RMSE
rf_rmse <- sqrt(mean((testData$LIEX - rf_pred)^2))
#R-squared
rf_rsq <- cor(testData$LIEX, rf_pred)^2


#CART
fit <- rpart(LIEX ~ ., data = trainData)
predictions <- predict(fit, newdata = testData)

#RMSE
cart_rmse <- sqrt(mean((testData$LIEX - predictions)^2))
#R-squared
SSE <- sum((testData$LIEX - predictions)^2)
SST <- sum((testData$LIEX - mean(trainData$LIEX))^2)
cart_rsq <- 1 - SSE/SST

#----------------------------- Modele Comparasion -----------------------------#

#Create Comparasion Matrix
model_comparison <- data.frame(
  model = c("Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest", "CART"),
  rmse = c(lin_rmse, ridge_rmse, lasso_rmse, rf_rmse, cart_rmse),
  r2 = c(lin_rsq, ridge_rsq, lasso_rsq, rf_rsq, cart_rsq))
model_comparison <- model_comparison[order(model_comparison$rmse),]
print(model_comparison)



#------------------------------------------------------------------------------#
#--------------------- Classification (Mental Health) -------------------------#
#------------------------------------------------------------------------------#

#Data Link: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

#Import Package
library(pROC)
library(dplyr)
library(ggplot2)

#Import Modelling Package
library(nnet)
library(MASS)
library(rpart)
library(caret)
library(randomForest)

#Load Data
setwd("/Users/dennis/Desktop/ML Coursework")
mental_rawdata <- read.csv("Mental Health.csv", stringsAsFactors = F)

#Explore Data
summary(mental_rawdata)

#------------------------------- DATA CLEANING --------------------------------#

#Rename Columns into same format
mental_rawdata <- mental_rawdata %>% 
  rename("time" = "Timestamp",
         "age"  = "Age",
         "gend" = "Gender",
         "coun" = "Country",
         "stat" = "state",
         "seem" = "self_employed",
         "fahi" = "family_history",
         "trea" = "treatment",
         "woin" = "work_interfere",
         "noem" = "no_employees",
         "rewo" = "remote_work",
         "teco" = "tech_company",
         "bene" = "benefits",
         "caop" = "care_options", 
         "wepr" = "wellness_program",
         "sehe" = "seek_help",
         "anon" = "anonymity",
         "leav" = "leave",
         "mehc" = "mental_health_consequence",
         "phhc" = "phys_health_consequence",
         "cowo" = "coworkers",
         "supe" = "supervisor",
         "mehi" = "mental_health_interview",
         "phhi" = "phys_health_interview",
         "mvsp" = "mental_vs_physical",
         "obco" = "obs_consequence",
         "comm" = "comments")

#Remove The Columns We Don't Need
mental_rawdata <- mental_rawdata[,!names(mental_rawdata) %in% c("time","coun", "stat", "comm")]

#Clean Gender
male <- c("m","maile","male (cis)", "make", "male ", "man", "malr", "cis male","Male (CIS)","Man","Cis Man","M","Malr","Mail","mail","Male ","cis man","Make","Mal","Male-ish","mal","Male","Guy (-ish) ^_^","msle","Cis Male","male-ish", "guy (-ish) ^_^")
female <- c("cis female", "female", "woman", "femake", "cis-female/femme", "female (cis)","Female","Woman","Cis Female","Femake","Female ","Female (cis)","F","f","female ", "femail")
mental_rawdata <- mental_rawdata %>% mutate(gend = if_else(gend %in% male, "male", gend))
mental_rawdata <- mental_rawdata %>% mutate(gend = if_else(gend %in% female, "female", gend))
mental_rawdata <- subset(mental_rawdata, gend %in% c("male", "female"))

#Fill the NA
mental_rawdata$seem <- ifelse(is.na(mental_rawdata$seem), "No", mental_rawdata$seem)
mental_rawdata$woin <- ifelse(is.na(mental_rawdata$woin), "Not sure", mental_rawdata$woin)
table(mental_rawdata$leav)

#------------------------------- DATA Plotting --------------------------------#

#Gender VS Treatment-Seeking
gendtrea <- mental_rawdata %>%
  group_by(gend) %>%
  summarise(gend_yes = mean(trea == "Yes") * 100)

ggplot(gendtrea, aes(x = gend, y = gend_yes, fill = gend)) + 
  geom_bar(stat = "identity") +
  xlab("Gender") +
  ylab("Treatment-Seeking %") +
  ggtitle("Mental Health Treatment Seeking Behavior by Gender.") +
  scale_fill_manual(values = c("#DC3912","#3366CC")) +
  labs(fill = "Gender")+
  geom_text(aes(label = paste0(round(gend_yes, 1), "%")), 
            position = position_stack(vjust = 0.5)) +
  theme(legend.position = "top")

#Family History VS Treatment-Seeking
fahitrea <- mental_rawdata %>%
  group_by(fahi) %>%
  summarise(fahi_yes = mean(trea == "Yes") * 100)

ggplot(fahitrea, aes(x = fahi, y = fahi_yes, fill = fahi)) + 
  geom_bar(stat = "identity") +
  xlab("Family History") +
  ylab("Treatment-Seeking %") +
  ggtitle("Mental Health Treatment Seeking Behavior by Family History.") +
  labs(fill = "family history of mental illness") +
  geom_text(aes(label = paste0(round(fahi_yes, 1), "%")), 
            position = position_stack(vjust = 0.5)) +
  theme(legend.position = "top")

#Tech Industry Employment VS Treatment-Seeking
tecotrea <- mental_rawdata %>%
  group_by(teco) %>%
  summarise(teco_yes = mean(trea == "Yes") * 100)

ggplot(tecotrea, aes(x = teco, y = teco_yes, fill = teco)) + 
  geom_bar(stat = "identity") +
  xlab("Tech Industry Employment") +
  ylab("Treatment-Seeking %") +
  ggtitle("Mental Health Treatment Seeking by Tech Industry Employment.") +
  labs(fill = "Employer a tech company") +
  geom_text(aes(label = paste0(round(teco_yes, 1), "%")), 
            position = position_stack(vjust = 0.5)) +
  theme(legend.position = "top")

#Age VS Treatment-Seeking
mental_rawdata$age_group <- cut(mental_rawdata$age, breaks = seq(10, 70, 10), right = FALSE)

agetrea <- aggregate(trea ~ age_group, data = mental_rawdata, FUN = function(x) {
  age_yes <- sum(x == "Yes")
  total <- length(x)
  age_yes / total})

ggplot(agetrea, aes(x = age_group, y = trea)) +
  geom_bar(stat = "identity", fill = "darkgoldenrod2") +
  scale_y_continuous(labels = scales::percent_format()) +
  xlab("Age group") +
  ylab("Treatment-Seeking %") +
  ggtitle("Mental Health Treatment Seeking Behavior by Age Group.") +
  labs(fill = "Treatment-Seeking")

#Work Interferes VS Treatment-Seeking
wointrea <- data.frame(table(mental_rawdata$woin))
wointrea$Freq <- wointrea$Freq / sum(wointrea$Freq) * 100

wointrea$order <- factor(wointrea$Var1, levels = c("Never", "Rarely","Not sure", "Sometimes", "Often"))

ggplot(wointrea, aes(x = order, y = Freq, fill = Var1)) + 
  geom_bar(stat = "identity") +
  xlab("Frequency") +
  ylab("Population %") +
  ggtitle("The frequency that mental health interferes with your work") +
  guides(fill = FALSE) +
  geom_text(aes(label = paste0(round(Freq, 1), "%")), 
            position = position_stack(vjust = 0.5))

#------------------------------ DATA PREPARATION -------------------------------#

#Change to Dummy Variables
dummy_data <- mental_rawdata[,-24]
binary_cols <- c("fahi", "trea", "rewo", "teco", "obco","seem")
for (col in binary_cols) {
  dummy_data[[col]] <- ifelse(dummy_data[[col]] == "Yes", 1, 0)}
dummy_data$gend <- ifelse(dummy_data$gend == "male", 1, 0)
cols <- c("bene", "caop", "wepr", "sehe", "anon", "mehc", "phhc", "supe", "cowo", "mehi", "phhi", "mvsp")
for (col in cols) {
  dummy_data[[col]] <- ifelse(dummy_data[[col]] == "Yes", 2, ifelse(dummy_data[[col]] == "No", 0, 1))}
leav_values <- c("Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult")
dummy_data$leav <- match(dummy_data$leav, leav_values)
noem_values <- c("1-5","6-25","26-100","100-500","500-1000","More than 1000")
dummy_data$noem <- match(dummy_data$noem, noem_values)
woin_values <- c("Never","Rarely","Not sure","Sometimes","Often")
dummy_data$woin <- match(dummy_data$woin, woin_values)

#Scale Age
dummy_data <- subset(dummy_data, age > 0 & age < 100)
scaled_age <- scale(dummy_data$age, center = min(dummy_data$age), scale = max(dummy_data$age) - min(dummy_data$age))
dummy_data$age <- round(scaled_age, digits = 3)

#---------------------------------- Modeling ----------------------------------#

#Set Up Training set and Test set
set.seed(520)
training_indices <- createDataPartition(as.factor(dummy_data$trea), p = 0.7, list = FALSE)
training_data <- dummy_data[training_indices, ]
testing_data <- dummy_data[-training_indices, ]
Act_class <- testing_data$trea

#Logit Model
logit_model <- glm(trea ~ ., data = training_data, family = "binomial")
logit_pre_value <- predict(logit_model, newdata = testing_data, type = "response")
logit_pre_class <- ifelse(logit_pre_value > 0.5, 1, 0)
logit_accuracy <- mean(logit_pre_class == Act_class)

logit_confusion_matrix <- table(Predicted = logit_pre_class, Actual = Act_class)
logit_confusion_matrix

logit_roc <- roc(Act_class, logit_pre_value)
plot(logit_roc, print.thres = c(0.5), grid=c(0.1, 0.2),
     print.auc = TRUE, auc.polygon = TRUE, max.auc.polygon = TRUE, 
     grid.col = "lightgray", lwd = 2, legacy.axes = TRUE, main = "Logit Regression ROC Curve",
     xlab = "1 - Specificity", ylab = "Sensitivity")
logit_roc_value <- round(auc(logit_roc), 2)

#CART Model
cart_model <- rpart(trea ~ ., data = training_data, method = "class")
cart_pre_prob <- predict(cart_model, newdata = testing_data, type = "prob")[,2]
cart_pre_class <- predict(cart_model, newdata = testing_data, type = "class")
cart_accuracy <- mean(cart_pre_class == testing_data$trea)
cart_confusion_matrix <- table(Predicted = cart_pre_class, Actual = Act_class)
cart_confusion_matrix

cart_roc <- roc(Act_class, cart_pre_prob)
plot(cart_roc, print.thres = c(0.5), grid=c(0.1, 0.2),
     print.auc = TRUE, auc.polygon = TRUE, max.auc.polygon = TRUE, 
     grid.col = "lightgray", lwd = 2, legacy.axes = TRUE, main = "CART ROC Curve",
     xlab = "1 - Specificity", ylab = "Sensitivity")
cart_roc_value <- round(auc(cart_roc), 2)

#LDA Model
lda_model <- lda(trea ~ ., data = training_data)
lda_pre_class <- predict(lda_model, newdata = testing_data)$class
lda_predictions <- predict(lda_model, newdata = testing_data)
lda_probabilities <- as.numeric(lda_predictions$posterior[,2])

lda_accuracy <- mean(lda_pre_class == Act_class)
lda_confusion_matrix <- table(lda_pre_class, Act_class)
lda_confusion_matrix

lda_roc <- roc(Act_class, lda_probabilities)
plot(lda_roc, print.thres = c(0.5), grid=c(0.1, 0.2),
     print.auc = TRUE, auc.polygon = TRUE, max.auc.polygon = TRUE, 
     grid.col = "lightgray", lwd = 2, legacy.axes = TRUE, main = "LDA ROC Curve",
     xlab = "1 - Specificity", ylab = "Sensitivity")
lda_roc_value <- round(auc(lda_roc), 2)

#Random Forest Model
rf_model <- randomForest(trea ~ ., data = training_data)
rf_pre_value <- predict(rf_model, newdata = testing_data)
rf_pre_class <- ifelse(rf_pre_value > 0.5, 1, 0)
rf_accuracy <- mean(rf_pre_class == Act_class)

rf_roc <- roc(Act_class, rf_pre_value)
plot(rf_roc, print.thres = c(0.5), grid=c(0.1, 0.2),
     print.auc = TRUE, auc.polygon = TRUE, max.auc.polygon = TRUE, 
     grid.col = "lightgray", lwd = 2, legacy.axes = TRUE, main = "Random Forest ROC Curve",
     xlab = "1 - Specificity", ylab = "Sensitivity")
rf_roc_value <- round(auc(rf_roc), 2)

#Neural Network Model
training_data$trea <- as.factor(training_data$trea)
testing_data$trea <- as.factor(testing_data$trea)
Act_class<-testing_data$trea

nn_model <- nnet(trea ~ ., data = training_data, size = 5)
nn_pre_class <- predict(nn_model, newdata = testing_data, type = "class")
nn_pre_prob <- predict(nn_model, newdata = testing_data, type = "raw")[,1]
nn_accuracy <- mean(nn_pre_class == testing_data$trea)
nn_confusion_matrix <- table(Predicted = nn_pre_class, Actual = Act_class)
nn_confusion_matrix

nn_roc <- roc(Act_class, nn_pre_prob)
plot(nn_roc, print.thres = c(0.5), grid=c(0.1, 0.2),
     print.auc = TRUE, auc.polygon = TRUE, max.auc.polygon = TRUE, 
     grid.col = "lightgray", lwd = 2, legacy.axes = TRUE, main = "Neural Network ROC Curve",
     xlab = "1 - Specificity", ylab = "Sensitivity")
nn_roc_value <- round(auc(nn_roc), 2)

#KNN Model
knn_model <- train(trea ~ ., data = training_data, method = "knn", trControl = trainControl(method = "cv", number = 10))
knn_pre_class <- predict(knn_model, newdata = testing_data)
knn_pre_prob <- predict(knn_model, newdata = testing_data, type = "prob")
knn_confusion_matrix <- confusionMatrix(knn_pre_class, Act_class)
knn_accuracy <- knn_confusion_matrix$overall["Accuracy"]

knn_roc <- roc(testing_data$trea, knn_pre_prob[, 2])
plot(knn_roc, print.thres = c(0.5), grid=c(0.1, 0.2),
     print.auc = TRUE, auc.polygon = TRUE, max.auc.polygon = TRUE, 
     grid.col = "lightgray", lwd = 2, legacy.axes = TRUE, main = "KNN ROC Curve",
     xlab = "1 - Specificity", ylab = "Sensitivity")

knn_roc_value <- round(auc(knn_roc), 2)
knn_confusion_matrix <- table(Predicted = knn_pre_class, Actual = Act_class)
knn_confusion_matrix
knn_roc_value <- round(auc(knn_roc), 2)

#----------------------------- Modele Comparasion -----------------------------#

#Create Comparasion Matrix
model_comparison <- data.frame(
  model = c("Logit Regression", "Random Forest", "CART", "Neural Network", "KNN", "LDA"),
  Accuracy = c(logit_accuracy, rf_accuracy, cart_accuracy, nn_accuracy, knn_accuracy, lda_accuracy),
  AUC = c(logit_roc_value, rf_roc_value, cart_roc_value, nn_roc_value, knn_roc_value, lda_roc_value))
model_comparison <- model_comparison[order(desc(model_comparison$Accuracy)),]
print(model_comparison)

