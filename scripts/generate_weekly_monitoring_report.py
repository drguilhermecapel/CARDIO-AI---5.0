import argparse
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def generate_weekly_report(project_id, endpoint_id, output_file="weekly_monitoring_report.pdf"):
    print("Generating Weekly Monitoring Report...")
    
    client = bigquery.Client(project=project_id)
    
    # 1. Fetch Prediction Stats (Last 7 Days)
    # Assuming standard Vertex AI logging table schema
    # We need to find the table first. Usually 'project.dataset.prediction_logs'
    # For this script, we assume a known table or query a view.
    
    # Mock Data for Report
    dates = pd.date_range(end=datetime.now(), periods=7)
    daily_requests = [120, 145, 132, 150, 160, 140, 155]
    drift_scores = [0.02, 0.03, 0.02, 0.08, 0.12, 0.05, 0.03] # Spike on Day 5
    
    # 2. Visualize Drift
    plt.figure(figsize=(10, 4))
    plt.plot(dates, drift_scores, 'r-o', label='Prediction Drift (KL Divergence)')
    plt.axhline(y=0.1, color='k', linestyle='--', label='Threshold')
    plt.title("Weekly Drift Analysis")
    plt.ylabel("Drift Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/tmp/drift_chart.png")
    
    # 3. Generate PDF
    c = canvas.Canvas(output_file, pagesize=A4)
    width, height = A4
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Weekly Model Monitoring Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Endpoint: {endpoint_id}")
    c.drawString(50, height - 85, f"Period: {dates[0].date()} to {dates[-1].date()}")
    
    # Summary
    c.drawString(50, height - 120, "Summary:")
    c.drawString(70, height - 140, f"- Total Requests: {sum(daily_requests)}")
    c.drawString(70, height - 155, f"- Max Drift Score: {max(drift_scores):.3f}")
    
    status = "STABLE" if max(drift_scores) < 0.1 else "DRIFT DETECTED"
    c.setFillColor("red" if status == "DRIFT DETECTED" else "green")
    c.drawString(70, height - 170, f"- Status: {status}")
    c.setFillColor("black")
    
    # Chart
    c.drawImage("/tmp/drift_chart.png", 50, height - 400, width=500, height=200)
    
    # Recommendations
    c.drawString(50, height - 430, "Recommendations:")
    if status == "DRIFT DETECTED":
        c.drawString(70, height - 450, "1. Investigate traffic on Day 5.")
        c.drawString(70, height - 465, "2. Check for new data sources or sensor changes.")
        c.drawString(70, height - 480, "3. Consider retraining if performance drops.")
    else:
        c.drawString(70, height - 450, "Model is operating within normal parameters.")
        
    c.save()
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--endpoint_id', required=True)
    
    args = parser.parse_args()
    generate_weekly_report(args.project_id, args.endpoint_id)
