# Project Charter Draft

## Business background

* The primary clients are consumer lending companies, banks developing new consumer products.
* Loan acceptance increase, profit increase

## Scope
* we are building AI-powered scoring solution provider
* delivery mode: web-based application (probably Microsfot Azure Web App Service #?)

## Metrics

## Plan
* Phases (milestones), timeline, short description of what we'll do in each phase.

## Architecture
* Data
  * What data do we expect? Raw historical data in in xls/csv format from the customer data sources (e.g. on-prem files, SQL, on-prem Hadoop etc.)
  * Sampled data enough for modeling 

* What tools and data storage/analytics resources will be used in the solution e.g.,
  * Azure Data Storage
  * Python/TensorFlow for feature construction, aggregation and sampling
  * AzureML for modeling and web service operationalization
* How will the score or operationalized web service(s) (RRS and/or BES) be consumed in the business workflow of the customer?
  * Azure Web App API
  * How will the customer use the model results to make decisions: Power BI connected to Azure Stream Analytics
  * Data movement pipeline in production
