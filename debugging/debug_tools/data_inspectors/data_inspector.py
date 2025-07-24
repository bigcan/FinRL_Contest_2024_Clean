"""
Data Inspector - Utility for validating and analyzing dataset integrity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

class DataInspector:
    """Comprehensive data validation and analysis tool."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize the DataInspector."""
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
    def validate_btc_data(self, csv_path: str, npy_path: Optional[str] = None) -> Dict:
        """Validate Bitcoin LOB data for Task 1."""
        results = {"status": "success", "issues": []}
        
        try:
            # Load CSV data
            self.logger.info(f"Loading Bitcoin data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Basic validation
            results["csv_shape"] = df.shape
            results["csv_columns"] = list(df.columns)
            results["csv_dtypes"] = df.dtypes.to_dict()
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                results["issues"].append(f"Missing values found: {missing_values.to_dict()}")
            
            # Check data types
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) != len(df.columns):
                results["issues"].append("Non-numeric columns detected")
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                results["issues"].append(f"Duplicate rows found: {duplicates}")
            
            # Statistical summary
            results["statistical_summary"] = df.describe().to_dict()
            
            # Validate NPY data if provided
            if npy_path:
                self.logger.info(f"Loading prediction array from {npy_path}")
                pred_array = np.load(npy_path)
                results["npy_shape"] = pred_array.shape
                results["npy_dtype"] = str(pred_array.dtype)
                
                # Check alignment
                if pred_array.shape[0] != df.shape[0]:
                    results["issues"].append("CSV and NPY row counts don't match")
            
            self.logger.info("Bitcoin data validation completed")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            self.logger.error(f"Error validating Bitcoin data: {e}")
        
        return results
    
    def validate_news_data(self, news_path: str, stocks_path: str) -> Dict:
        """Validate news and stock data for Task 2."""
        results = {"status": "success", "issues": []}
        
        try:
            # Load news data
            self.logger.info(f"Loading news data from {news_path}")
            news_df = pd.read_csv(news_path)
            
            # Load stock data
            self.logger.info(f"Loading stock data from {stocks_path}")
            stocks_df = pd.read_csv(stocks_path)
            
            # Validate news data
            results["news_shape"] = news_df.shape
            results["news_columns"] = list(news_df.columns)
            
            # Check required columns
            required_news_cols = ["Date", "Ticker", "Headline"]
            missing_news_cols = [col for col in required_news_cols if col not in news_df.columns]
            if missing_news_cols:
                results["issues"].append(f"Missing news columns: {missing_news_cols}")
            
            # Validate stock data
            results["stocks_shape"] = stocks_df.shape
            results["stocks_columns"] = list(stocks_df.columns)
            
            # Check required columns
            required_stock_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
            missing_stock_cols = [col for col in required_stock_cols if col not in stocks_df.columns]
            if missing_stock_cols:
                results["issues"].append(f"Missing stock columns: {missing_stock_cols}")
            
            # Check date alignment
            if "Date" in news_df.columns and "Date" in stocks_df.columns:
                news_dates = set(pd.to_datetime(news_df["Date"]).dt.date)
                stock_dates = set(pd.to_datetime(stocks_df["Date"]).dt.date)
                
                date_overlap = len(news_dates.intersection(stock_dates))
                results["date_overlap"] = date_overlap
                
                if date_overlap == 0:
                    results["issues"].append("No overlapping dates between news and stock data")
            
            # Check ticker alignment
            if "Ticker" in news_df.columns and "Ticker" in stocks_df.columns:
                news_tickers = set(news_df["Ticker"])
                stock_tickers = set(stocks_df["Ticker"])
                
                ticker_overlap = len(news_tickers.intersection(stock_tickers))
                results["ticker_overlap"] = ticker_overlap
                
                if ticker_overlap == 0:
                    results["issues"].append("No overlapping tickers between news and stock data")
            
            # Check for missing values
            news_missing = news_df.isnull().sum()
            stocks_missing = stocks_df.isnull().sum()
            
            if news_missing.sum() > 0:
                results["issues"].append(f"Missing values in news data: {news_missing.to_dict()}")
            
            if stocks_missing.sum() > 0:
                results["issues"].append(f"Missing values in stock data: {stocks_missing.to_dict()}")
            
            self.logger.info("News and stock data validation completed")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            self.logger.error(f"Error validating news/stock data: {e}")
        
        return results
    
    def analyze_data_distribution(self, data_path: str, output_dir: str = "debugging/intermediate_outputs/visualizations/") -> None:
        """Generate data distribution analysis plots."""
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                self.logger.warning("Unsupported file format for distribution analysis")
                return
            
            # Generate distribution plots
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f"Data Distribution Analysis: {Path(data_path).name}")
                
                # Histogram
                df[numeric_columns].hist(bins=50, ax=axes[0, 0])
                axes[0, 0].set_title("Histograms")
                
                # Box plots
                df[numeric_columns].boxplot(ax=axes[0, 1])
                axes[0, 1].set_title("Box Plots")
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Correlation heatmap
                if len(numeric_columns) > 1:
                    sns.heatmap(df[numeric_columns].corr(), ax=axes[1, 0], annot=True, cmap='coolwarm')
                    axes[1, 0].set_title("Correlation Matrix")
                
                # Time series (if Date column exists)
                if "Date" in df.columns:
                    df_temp = df.copy()
                    df_temp["Date"] = pd.to_datetime(df_temp["Date"])
                    df_temp.set_index("Date")[numeric_columns[0]].plot(ax=axes[1, 1])
                    axes[1, 1].set_title(f"Time Series: {numeric_columns[0]}")
                
                plt.tight_layout()
                output_path = Path(output_dir) / f"distribution_analysis_{Path(data_path).stem}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Distribution analysis saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error in distribution analysis: {e}")
    
    def generate_data_report(self, data_paths: Dict[str, str], output_path: str = "debugging/logs/data_validation_report.html") -> None:
        """Generate comprehensive data validation report."""
        try:
            html_content = ["<html><head><title>Data Validation Report</title></head><body>"]
            html_content.append("<h1>FinRL Contest 2024 - Data Validation Report</h1>")
            html_content.append(f"<p>Generated on: {pd.Timestamp.now()}</p>")
            
            for data_name, data_path in data_paths.items():
                html_content.append(f"<h2>{data_name}</h2>")
                
                if "btc" in data_name.lower():
                    results = self.validate_btc_data(data_path)
                elif "news" in data_name.lower():
                    # Assume corresponding stocks file exists
                    stocks_path = data_path.replace("news", "stocks")
                    results = self.validate_news_data(data_path, stocks_path)
                else:
                    continue
                
                html_content.append(f"<h3>Status: {results['status'].upper()}</h3>")
                
                if results.get("issues"):
                    html_content.append("<h4>Issues Found:</h4><ul>")
                    for issue in results["issues"]:
                        html_content.append(f"<li>{issue}</li>")
                    html_content.append("</ul>")
                else:
                    html_content.append("<p>✅ No issues found</p>")
                
                # Add detailed results
                html_content.append("<h4>Details:</h4>")
                html_content.append("<pre>")
                for key, value in results.items():
                    if key not in ["status", "issues", "error"]:
                        html_content.append(f"{key}: {value}\n")
                html_content.append("</pre>")
            
            html_content.append("</body></html>")
            
            # Write report
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("\n".join(html_content))
            
            self.logger.info(f"Data validation report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating data report: {e}")

if __name__ == "__main__":
    # Example usage
    inspector = DataInspector()
    
    # Example data paths (adjust as needed)
    data_paths = {
        "Bitcoin LOB Data": "data/raw/task1/BTC_1sec.csv",
        "News Data": "data/raw/task2/task2_news_train.csv"
    }
    
    inspector.generate_data_report(data_paths)