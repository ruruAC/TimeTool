# TimeTool
This is a tool for time series analysis that enables preprocessing, interpolation, prediction, and more for input time series data. 
Quict Start:
''python TimeTool/TimeToolv1011_Chinese.py''

Here are some functions within this tool:

1. Click on "Time Interval (min)" to generate a time series data's time interval frequency and a sample distribution histogram. Select the time interval with the highest frequency as the basis for data segmentation. Use this generated time interval to perform interpolation for filling missing values. Different data files may result in different time intervals, so you need to click "Time Interval (min)" again when selecting a new file to get the corresponding time interval values.

2. Click on "Select Interpolation Algorithm" to choose from the following interpolation methods:
   - Actual: Use actual values.
   - Forward Fill: Use forward filling.
   - Backward Fill: Use backward filling.
   - Linear Interpolation: Apply linear interpolation.
   - Cubic Interpolation: Use cubic spline interpolation.
   - knn_mean: Implement k-nearest neighbor mean interpolation.
   - test-50: Use 5-fold cross-validation to calculate the Mean Squared Error (MSE) for different interpolation methods.
   - All: Try all available interpolation methods.

3. Click "Start" to initiate the interpolation process. Once completed, a corresponding .csv file will be generated in the current directory.'

4. Visualize the accuracy of each algorithm to choose reliable algorithms and their corresponding results.

This tool utilizes Streamlit to provide visual result presentation, enhancing data interaction and information retrieval. It also builds a remote API for data stream processing through FastAPI.

You can find a detailed demo [here](https://github.com/ruruAC/TimeTool/master/Demo.pdf).
