How to run the trading system
1.	Make sure all the files (.py and csv) are in same folder.
2.	benchmark.csv and benchmark_test.csv are manually created.
3.	modify the path to Data folder in util.py
4.	Run the indicators.py to create indicators graphs. Four graphs will be created.
5.	Run the rule_based.py to create order file (orders_RuleBased.csv) and graph (Rule_Based_Chart.png). 
6.	Run the ML_based.py to create order file (ordersMLBased.csv) and graph (ML_Based_Both.png). 
7.	Make sure all the mentioned csv files are existed before you run this code. Run the Test_OutOfSample.py to create order file (orders_MLBased_TestingData.csv and orders_RuleBased_TestingData.csv) and graph (ML_RuleBased_Testing_Data.png)
