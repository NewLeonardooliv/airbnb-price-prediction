# Airbnb Price Prediction Using InsideAirbnb Dataset

## Overview
Airbnb claims to be a part of the "sharing economy" and aims to revolutionize the hospitality industry. The InsideAirbnb project (http://insideairbnb.com) has been collecting data on properties across various regions of the world to assess the impact of these rentals on urban areas.

In this project, you will use the dataset published by InsideAirbnb to create a Neural Network (NN) capable of predicting the rental price of a property in New York City through linear regression based on specific characteristics.

## Dataset
The dataset comprises 40,000 Airbnb listings in New York City from February 2023 to February 2024. It is available in two formats:
- Summary: [Listings Summary CSV](http://data.insideairbnb.com/united-states/ny/new-york-city/2024-02-06/visualisations/listings.csv)
- Complete: [Listings Complete CSV (Compressed)](http://data.insideairbnb.com/united-states/ny/new-york-city/2024-02-06/data/listings.csv.gz)

## Objective
Your model should predict the rental price (`price`) based on the following inputs:
- `neighbourhood_cleansed` (neighborhood)
- `property_type` (property type)
- `room_type` (room type)
- `accommodates` (number of accommodations)
- `bathrooms` (number of bathrooms)
- `bedrooms` (number of bedrooms)
- `beds` (number of beds)

## Requirements
- **Model Format:** Save your model in H5 format for deployment.
- **Documentation:** Submit a PDF with the notebook showing dataset loading, architecture, compilation parameters and training, training log, and the result of the evaluation.

## Notes
- Not all fields in the dataset are required to build a linear regressor. Remove unnecessary columns from a Pandas dataframe with `df.drop(['field1','field2'], axis=1)`.
- Pay attention to the data types. Normalize the data if necessary.
