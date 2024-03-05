
# Task One: Load and explore the data

## Load data and install packages

```{r}
## Import required packages

install.packages(c('timetk','tidymodels','tidyverse','lubridate','car','modeltime','glmnet'))
library(timetk)
library(tidymodels)
library(tidyverse)
library(lubridate)
library(car)
library(modeltime)
```


## Describe and explore the data

```{r}
#import data
df<-bike_sharing_daily
View(df)

#Analysing the difference between temp and atemp :
plot(df$temp, df$atemp, xlab = "Temp", ylab = "Feeling Temp")
cor(df$temp, df$atemp)
#from the plot + corr between the temp and atemp it is clear that they are nearly equal. We will keep the atemp.


#Analysing the data and seeing the important features that may influence the bike sharing
cor(df$atemp, df$cnt)
#we have 0.63 corr between atemp and cnt. There is strong positive correlation between them

cor(df$windspeed, df$cnt)
#we have -0.23 corr between windspeed and cnt.There is weak negative correlation between them so windspeed has no real impact on the cnt variable.

cor(df$hum, df$cnt)
#we have -0.1 corr between hum and cnt.There is weak negative correlation between them so hum has no real impact on the cnt variable.




bike_sharing<-bike_sharing_daily %>% select(dteday,cnt)
View(bike_sharing)
```



# Task Two: Create interactive time series plots

```{r}

bike_sharing %>% plot_time_series(dteday,cnt,.interactive = TRUE, .plotly_slider = TRUE)


```




# Task Three: Smooth time series data

```{r}
#Weekly smoothing : 
data_ts <- zoo(bike_sharing$cnt, order.by = bike_sharing$dteday,frequency=24)
bike_sharing.xts.weekly=apply.weekly(data_ts,FUN="mean")
plot(bike_sharing.xts.weekly)
bike_sharing.xts_weekly <- ts(bike_sharing.xts.weekly, frequency = 7)

```



# Task Four: Decompose and assess the stationarity of time series data

```{r}
df$dteday <- as.POSIXct(df$dteday, format = "%Y-%m-%d",by="day")
data_ts <- zoo(df$cnt, order.by = df$dteday,frequency=7)
frequency(data_ts)
multiplicative.decomposition<-decompose(data_ts)
#There is no seasonality on our dataset
acf(bike_sharing.xts_weekly,lag=21,main="ACF of weekly pollution",lwd=2)
pacf(bike_sharing.xts_weekly,lag=21,main="ACF of weekly pollution",lwd=2)
# from the acf and pacf we can see that the bike shared is essentially a random amount which is uncorrelated with that of previous days.
```



# Task Five: Fit and forecast time series data using ARIMA models

```{r}
#TRIN / TEST splits
splits <- time_series_split(
  bike_sharing,
  assess = "3 months",
  cumulative = TRUE
)
splits %>% tk_time_series_cv_plan() %>% plot_time_series_cv_plan(dteday,cnt)


#Forecasting with autoarima
model_arima<-arima_reg() %>% 
             set_engine("auto_arima") %>%
            fit(cnt~dteday,training(splits))

model_arima

#Forecasting with Prophet
model_prophet <- prophet_reg(
  seasonality_yearly = TRUE
) %>% set_engine('prophet') %>%
  fit(cnt ~ dteday,training(splits))
model_prophet

#Forecasting with Machine Learning GLM
model_glmnet <- linear_reg(penalty = 0.01) %>%
  set_engine('glmnet') %>%
  fit(
    cnt~ wday(dteday , label = TRUE)
    +month(dteday,label = TRUE)
    + as.numeric(dteday),
    training(splits)
  )
model_glmnet

#Comparing the 3 forcasting results using modeltime table
model_tbl<- modeltime_table (model_arima,
            model_prophet,
            model_glmnet)
#Calibrate (calculate the predictoins and residuals for the testset)
calib_tbl<-model_tbl %>% modeltime_calibrate(testing(splits))

#Accuracy
calib_tbl %>% modeltime_accuracy()
#We can see that the PROPHET and GLMNET score the best mae scores 


#Test set visualisation
calib_tbl %>% 
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = df
  ) %>%
  plot_modeltime_forecast()
#Graphicaly it is clear that the arima is performing the worst

#Forecast future (Here we will train the model on the full dataset and forcast the next 3 months)
future_forecast_tbl <- calib_tbl %>%
  modeltime_refit(df) %>%
  modeltime_forecast(
    h = "3 months",
    actual_data = df
  )

future_forecast_tbl %>%
  plot_modeltime_forecast()

```



# Task Six: Findings and Conclusions
#Arima seems to give worst results from PROPHET and GLMNET who give simliar results


