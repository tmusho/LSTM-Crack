# LSTM CRACK RESEARCH README

## ORDER OF OPERATIONS

1) Convert Binary data to Training/Testing/Predictions .csv files
	- Use Prepare_Bin_data.ipynb
	
2) Train LSTM using Generated training .csv Files
	- For Annealed sample use LSTM_Train.ipynb
	- For Heat Treated Sample use LSTM_Train_HeatTreat.ipynb
	
3) Create data for non-experimental cases
	- Use CreateTempData.ipynb
	
4) Convert non-experimental case data to needed prediction .csv format
	- Use Create_Predict_Data.ipynb
	
5) Validate data and test on non-experimental cases
	- For Annealed Sample use LSTM_Predict
	- For Heat Treated sample use LSTM_Predict_HeatTreat