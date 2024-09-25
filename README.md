# BlackScholes-Insight Option Pricing Model 📊

This project implements a web application using Python and Streamlit to calculate and visualize option prices based on the Black-Scholes model. The app allows users to input parameters such as spot price, strike price, time to maturity, volatility, and interest rate to compute Call and Put prices. Additionally, it offers interactive heatmaps for sensitivity analysis based on varying spot prices and volatility.

## Features
- **Call and Put Option Pricing**: Calculate option prices using the Black-Scholes model.
- **Customizable Inputs**: Input parameters such as spot price, strike price, time to maturity, volatility, and interest rate.
- **Interactive Heatmaps**: Visualize how option prices change with spot price and volatility.
- **Responsive UI**: Clean and user-friendly interface with dark theme customization.

## Tech Stack
- **Python**
  - `Streamlit`: For creating an interactive web app.
  - `SciPy`: For statistical functions and calculations.
  - `NumPy`: For numerical operations.
  - `Plotly` & `Matplotlib`: For visualizing data.
  - `Seaborn`: For heatmaps and plotting.
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/black-scholes-option-pricing.git
   cd black-scholes-option-pricing
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Input the desired parameters such as **Spot Price**, **Strike Price**, **Volatility**, etc. on the sidebar.
2. The app will calculate and display Call and Put option prices.
3. Customize the heatmap settings for sensitivity analysis of option prices based on varying spot prices and volatility.
4. The heatmaps will update interactively, offering a visual representation of how price changes affect options.

## Screenshots
- *Add relevant screenshots of your app's interface here.*
