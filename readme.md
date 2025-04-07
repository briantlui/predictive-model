# Hotel Demand and Cancellation Forecast
======================================================================
# Table of Contents
1. [Project Overview](#project-overview)
2. [Project Flow](#project-flow)
3. [Data Loading and Exploration](#1-data-loading-and-exploration)
4. [Preprocessing](#2-preprocessing)
5. [Modeling](#3-modeling)
6. [Model Evaluation](#4-model-evaluation)
7. [Side Quests](#5-side-quests)
8. [Next Adventures](#next-adventures)
9. [Data Dictionary](#data-dictionary)
10. [Repository](#repository)
11. [Dataset](#dataset)



## Project Overview:

**The Problem** <br>
In the hotel industry, a shared challenge all hotels face is selling their available rooms to maximize revenue without overselling their hotel.  Why is this a problem? A hotel has limited perishable inventory, unlike other industries (ie: retail), inventory cannot be carried over to the next day. Therefore, any rooms left unsold is lost revenue.  But you might think, why would a hotel every be in an oversold situation? Why would they sell more rooms than they actually have? Consider this, how often have you or a family member booked a hotel room but at some point, your plans changed?  Hotels need to account for that, so they sometimes need to sell more rooms than they actually have left.  If they are just reactive to a cancellation then they risk having unsold rooms.  But there is a tradeoff if they end the day oversold, depending on the hotel’s policy, they would have to refund the guest, pay for a ‘comparable hotel room’, transportation costs, and they risk damaging their reputation.

**Objective**<br>
My goal is to utilize this dataset to develop a machine learning model to help predict if a reservation will cancel based on patterns from historical data. Another strong consideration I will be evaluating the model on is reliability and generalization. Because not all hotels are the same, it’s important that the model can perform reliably with different hotel datasets in the future.

**Solution**<br>
If a revenue management system is able to accurately predict the room cancellations for a given day, it will help a hotel have a more optimal room strategy and help operations teams properly staff their hotels on a given day or week.

**My Motiviation**<br>
I've worked in the hotel revenue management discipline for over 14 years. In my career, I've progressed from being an user of a revenue management (RM) system, to becoming a RM system consultant, to a RM system product owner. I continually want to learn how and why a product works. What better way continue feeding my curiosities by going under the hood and building my own machine learning model that can predict cancellations (which plays a large part of a RM system)?

**The Data**<br>
I have identified two datasets that share the same source. The Science Direct dataset was shared in 2019 and the kaggle dataset took this data and cleaned it for “#TidyTuesday” in 2020. The data contains booking information for 2 hotels, a city hotel and a resort hotel. It provides reservation information such as: booking dates, arrival dates, number of adults/children, market segment, and if the reservation was cancelled or changed.

There data consisted of 119,390 reservations


# Project Flow: 

My Project is organized into 4 mains sections and 1 "side quest" (Spoilers included!)
## 1. Data Loading and Exploration
- In this section, I load the data, clean, and begin my initial exploration to see if I can discover any insights before modeling. This was a fun challenge, taking my domain experience into EDA to see if my past experiences as a revenue management product owner would help me uncover any interesting insights.  My biggest problem here was analysis paralysis. I wanted to find more and learn more about the data before modeling!
- However when conducting my initial exploration, I first focused on what I believed was most important, defining demand and evaluating how it compares overtime with cancellations. I quickly realized that there was not one simple solution, but there were two different paths to venture down in order to solve my goal, predicting Demand or predicting Cancellations. Demand would require time series forecasting while Cancellations would be a binary classification problem. As fun as time series forecasting would have been (the shiny object that hotels love talking about), I decided to focus on Cancellations instead.  If I could gain a deeper understanding of how cancellations could work, I could take my predictive model and add it onto a demand forecast in the future and provide a complete revenue management system to help hotels predict occupancy.  I also was excited to practice the binary classification ML workflow and evaluate the multiple models that were available to me.
- After this decision, I explored more deeply into the total number of cancellations in this data set and found that there was a slight class imbalance. When looking at cancellations by customer type, having the domain knowledge of working with Group and Contract forecasts, they are very different and difficult to predict compared to transient. Also considering there is a very small sample of these customer types in our dataset, it was safe to drop them from our data.
## 2. Preprocessing
- In this section, I took the discoveries from my initial exploration of the data and make the necessary feature engineering with the goal of getting the data to a model-ready state. I addressed the customer_type feature by combining the two transient customer types and removing both Group and Contract customer types. I feature engineered a variable for length of stay which I thought would be interesting to explore if longer or shorter stay patterns influence cancellations or not. 
- After my initial iteration of baseline modeling, I returned to preprocessing to correct leakage that caused overfitting due to the reservation_status and reservation_status_date columns. Essentially, reservation_status contains data on if the reservation would ultimately be cancelled, no showed, or checked out.  Another discovery I identified was that when the arrival_date and reservation_status_date were both included, overfitting/leakage would occur because any reservations that cancelled would have a reservation status day BEFORE the arrival_date.  Whereas any reservation that checked out, would have the opposite, the reservation_status_date would be AFTER the arrival_date.  Both of these were addressed before proceeding to the next modeling phase.
## 3. Modeling
- In the modeling section, I addressed the class imbalance by applying SMOTE to the training data. Side note(And prior to handling this class imbalance, my baseline models were severely overfitted and have train accuracies and precision scores of nearly 100%.  My models were too perfect...which led me to handing class imbalance.)

![Percentage of reservations by Hotel](/references/Percent%20of%20Cancelled%20Pie.png)

- I first ran the baseline models for Logistic Regression, Random Forest, and XGBoost models. Then added hyperparameters via manual hyperparameter optimization and grid search in hopes of finding the best performing model.
- For logistic regression, optimizing the C value did not produce significant results. When applying principal component analysis to reduce dimensionality, I noticed that the accuracy and f1-scores also did not improve.
- For random forest, by manually optimizing hyperparameters, I was able to significantly decrease the overfitting. The train vs. test score accuracy difference decreased from 14% to less than 1%.  However, there was a 10% drop in the f1-score (with a 13% decrease in Recall). But when combining my manual optimization with grid search, I achieved a 3rd random forest model that maintained minimal overfitting and a 2% increase in f-1 score compared to my previous model run.
- For the XGBoost model, the results of adding hyperparameters were opposite of the Random Forest models. F1-score and accuracy displayed only a 1% increase, however it also increased overfitting. This pattern of increased overfitting continued when using gridsearch to further optimize the Hyperparameters.
## 4. Model Evaluation
- In this section, I evaluated all the machine learning models that have been run and began narrowing them down to select just 1 model. 
- Remembering that my primary goal is to not only have high accuracy, but the model needs to be reliable and able to perform well with different hotel datasets.  Therefore, I prioritized the lowest overfitting possible. 
- The metrics that I evaluated were using classification reports to evaluate accuracy, precision, recall, and f-1 scores. I also reviewed ROC curves to calculate the Area under the Curve (AUC) scores for each model to evaluate the model overall. 
- After selecting the “best” model, I continued model evaluation with a confusion matrix to identify True Positives, False Positives, True Negatives, and False Negatives. This helped identify where the model performs well and incorrectly predicts cancellations. 
- Lastly I evaluated which features are most impactful to the model and what the model believes to be the driving factors to predicting if a reservation will cancel or not.
- I ultimately chose the Random Forest model because it is only marginally lower in accuracy by 4%, AUC by ~3%, and F1-score by 4%.  The largest difference is that the random forest model is 4% lower in recall compared to the XGBoost model. However, it has less overfitting and suggests that it will likely perform more consistently on new data. A model that has higher overfitting is tailoring itself too closely to the training data and therefore may not perform as well on unseen data. This could lead to unreliable predictions.
- When evaluating the confusion matrix, I found that out of the 8500 total reservations that cancelled, the model predicted 1500 reservations to cancel, but they actually did not cancel. And the model predicted 2800 reservations would not cancel, but they actually did cancel.
- Recall: Of all the cancelled reservations (8581), the model correctly predicted 5764 reservations correctly
- Precision: Out of the 7316 reservations the model said were going to cancel, 5764 reservations actually cancelled?

- I further explored these errors in the model using box plot distributions and the most insightful feature I discovered that the model struggles with is lead time. The model’s average of true positives (accurately predicting a cancellation) were above 100 days, whereas the model’s average false positives (predicting a cancellation, but the reservation did not cancel) was below 100 days. This suggests that the model may struggle with reservations that have shorter lead times. *(See distribution plot below)*

![Lead Time Distribution by Confusion Matrix Label](/references/Error%20Analysis%20-%20Lead%20Time.png)


- I attempted to run Shapley values on the random forest model as well, however due to computational limitations with the model type and depth of trees, I substituted with the XGBoost model. This will provide similar results to my random forest model as the top 4 most important features were the same, just in a different order.

- When diving deeper into dependency plots of the SHAP values, I noticed the lead time’s pattern the most, where the shorter the lead time, the less likely the reservations are going to cancel. One interesting discovery I found over, is that between the 2 arrival features (Arrival Date Week Number and Arrival Month), they displayed inverse SHAP values. I would expect both of them to share a similar trend.  To investigate this further, I would evaluate additional possible overfitting or investigate my model complexity as the model may be focusing too much on individual weeks vs. the broader monthly patterns and trends. *(See SHAP dependency plots below)*

![SHAP Arrival Week Number](/references/SHAP_arrival_date_week_number.png)

![SHAP Arrival Month](/references/SHAP_arrival_month.png)


## 5. Side Quests
- The purpose of this last section was to keep the side quests and dead-ends that I ran into during my project. The biggest side quests in this notebook is running baseline models on the overfitted and umbalanced data. I progressed through the each model attempting multiple iterations in attemps to fix the overfitting and "make the models less perfect".  Ultimately, when I discovered the leakage that `reservation_status` and `reservation_status_date` columns brought to my data, I discontinued this work.  However, I wanted to keep this as an appendix if you want to see the side quests and problem solving I attempted in my earlier iteration.


### Next Adventures:
Next Steps:
- I chose my first workflow to take my domain knowledge and experience working with hotel cancellation forecasts out of the equation. I chose to do this so that I could have an unbaised evaluation of a dataset first before apply my domain knowledge into the model.  Perhaps my past experience was a bias, therefore I wanted to have a true end to end experience working with a dataset to see what I could uncover.  
    - With that in mind, my next iteration would include feature engineering and removal of some features that I truly did not believe would have any impact on predicting cancellations (such as `children`, `babies`, `meal`). 
    - I would also use my domain knowledge and bring back features such as `reserved_room_type` and `assigned_room_type` and apply one hot encoding to those features.  My 'unbiased' approach felt that this would have created too many features, however, I believe the type of room reserved strongly plays a role in demand and may also play a role in predicting cancellations.
- Return to feature engineering and remove stays in weekday nights and stay in weekend nights due to redundancy now that I have length of stay (los).
- Binning length of stay (`los`) and `lead_time` into smaller groupings to reduce complexity and help the model learn better.
- Explore Demand
    - I feel like this is only half of the journey. I ultimately want to continue uncovering what's under the hood of a revenue management system to better understand the revenue management products I've been helping build these last 5 years as a product owner.
    - I want to explore predicting demand using statistical time series models such as ARMA and SARIMA, however these would only focus on the target variable (demand) with no features. So I would also explore other Machine Learnings models such as Linear Regression and XGBoost as well.
 (Preview of my next adventures!)
![Weekly Demand Over Time](/references/Weekly%20Demand%20Over%20Time.png)

### Data Dictionary


| Column                        | Description                                                                                                                                                                                                                       |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| hotel                         | H1 = Resort Hotel or H2 = City Hotel                                                                                                                                                                                              |
| is_canceled                   | Value indicating if the booking was canceled (1) or not (0)                                                                                                                                                                       |
| lead_time                     | Number of days that elapsed between the entering date of the booking into the PMS and the arrival date                                                                                                                            |
| arrival_date_year             | Year of arrival date                                                                                                                                                                                                              |
| arrival_date_month            | Month of arrival date                                                                                                                                                                                                             |
| arrival_date_week_number      | Week number of year for arrival date                                                                                                                                                                                              |
| arrival_date_day_of_month     | Day of arrival date                                                                                                                                                                                                               |
| stays_in_weekend_nights       | Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel                                                                                                                                     |
| stays_in_week_nights          | Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel                                                                                                                                           |
| adults                        | Number of adults                                                                                                                                                                                                                   |
| children                      | Number of children                                                                                                                                                                                                                |
| babies                        | Number of babies                                                                                                                                                                                                                   |
| meal                          | Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch, and dinner) |
| country                       | Country of origin. Categories are represented in the ISO 3155–3:2013 format                                                                                                                                                        |
| market_segment                | Market segment designation. In categories, the term "TA" means "Travel Agents" and "TO" means "Tour Operators"                                                                                                                     |
| distribution_channel          | Booking distribution channel. The term "TA" means "Travel Agents" and "TO" means "Tour Operators"                                                                                                                                  |
| is_repeated_guest             | Value indicating if the booking name was from a repeated guest (1) or not (0)                                                                                                                                                      |
| previous_cancellations        | Number of previous bookings that were cancelled by the customer prior to the current booking                                                                                                                                       |
| previous_bookings_not_canceled| Number of previous bookings not cancelled by the customer prior to the current booking                                                                                                                                             |
| reserved_room_type            | Code of room type reserved. Code is presented instead of designation for anonymity reasons.                                                                                                                                        |
| assigned_room_type            | Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons. |
| booking_changes               | Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation                                                                                    |
| deposit_type                  | Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay. |
| agent                         | ID of the travel agency that made the booking                                                                                                                                                                                      |
| company                       | ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons                                                                                 |
| days_in_waiting_list          | Number of days the booking was in the waiting list before it was confirmed to the customer                                                                                                                                         |
| customer_type                 | Type of booking, assuming one of four categories: Contract – when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking |
| adr                           | Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights                                                                                                                |
| required_car_parking_spaces   | Number of car parking spaces required by the customer                                                                                                                                                                              |
| total_of_special_requests     | Number of special requests made by the customer (e.g. twin bed or high floor)                                                                                                                                                      |
| reservation_status            | Reservation status, assuming one of three categories: Canceled – booking was canceled by the customer; Check-Out – customer has checked in but already departed; No-Show – customer did not check-in and did not inform the hotel
| reservation_status_date       |  Date at which the last status was set. This variable can be   used in conjunction with the ReservationStatus to understand when was the   booking canceled or when did the customer checked-out of the hotel                      |

#### Repository 

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible cloud storage)
    - saved copy of aggregated / processed data as long as those are not too large 

* `model`
    - `joblib` dump of final model(s)
    - `joblib` dump of grid searches
    - note: random_forest_sm model was too large and cannot be uploaded to github. However it was the baseline model and was overfitted and not used as a top performing model

* `notebooks`
    - contains all final notebooks involved in the project

* `docs`
    - contains final report, presentations which summarize the project

* `references`
    - images used throughout the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `conda.yml`
    - Conda environment specification

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

#### Dataset

[Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data)

