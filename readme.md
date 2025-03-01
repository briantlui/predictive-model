# Hotel Demand and Cancellation Forecast
=========================

### Executive Summary

**Problem Area**<br>
My area of interest falls in the hotel industry.  My project could help hotels maximize profit by accurately predicting daily hotel demand and cancellations. By optimizing and accurately predicting the net rooms for a given day, a hotel more optimally maximizes revenue for projected sellout days as well as efficiently staff their hotels to improve daily operations such as housekeeping.


**The User**<br>
If a revenue management system is able to accurately predict the room demand and room cancellations for a given day, it will help a hotel have a more optimal room strategy and help operations teams properly staff their hotels on a given day or week.


**The Big Idea**<br>
Machine Learning can utilize predictive analytics to create demand forecasts based on patterns from historical data, seasonality, and external factors like events. Hotels can then optimize pricing strategies and also better staff their operations teams based on the predictions. In my specific dataset, I want to be able to predict demand and cancellations for a given day.


**The Impact**<br>
 Revenue management has become a growing need in the hotel industry. Gone are the days of simple spreadsheets to track what the “rate of the day” is. Hotel owners and general managers have progressively found the ROI to be high when investing in proper revenue management strategies deployed at their hotels. The goal of revenue management is “selling the right product to the right customer at the right time for the right price”. A revenue managers metrics of success include the following (depending on the hotel’s strategy):
- Maximizing the Revenue Per Available Room (RevPAR). 
- Reaching a “perfect sell”. This means not finishing the day with any rooms unsold. A hotel has limited perishable inventory. Any rooms that are left unsold receive 0 revenue. Unlike in other industries such as retail, inventory cannot be carried over the next day to be sold. The two biggest contributors to reaching a perfect sale are room demand and room cancellations.


**The Data**<br>
I have identified two datasets that share the same source. The Science Direct dataset was shared in 2019 and the kaggle dataset took this data and cleaned it for “#TidyTuesday” in 2020. The data contains booking information for 2 hotels, a city hotel and a resort hotel. It provides reservation information such as: booking dates, arrival dates, number of adults/children, market segment, and if the reservation was cancelled or changed.


### Methodology<br>
I would first start with linear regression models for any variables I’ve identified to have a linear relationship to either cancellation or demand. 
If the linear regression models prove inconclusive or unsuccessful, then I would move onto testing more advanced models like decision trees or random forests

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


### Organization

#### Repository 

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible cloud storage)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - `joblib` dump of final model(s)

* `notebooks`
    - contains all final notebooks involved in the project

* `docs`
    - contains final report, presentations which summarize the project

* `references`
    - contains papers / tutorials used in the project

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

