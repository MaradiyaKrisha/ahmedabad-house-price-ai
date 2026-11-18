Ahmedabad 3D Real Estate AI Valuator

Interactive 3D property configurator, XGBoost valuation engine, downloadable PDF valuation certificate, and built-in lead capture + Admin CRM built with Streamlit and Three.js.
Quick Setup

1. Install dependencies
pip install -r requirements.txt
2. Train the model (generates house_price_pipeline.pkl from housing_data.csv)
python train_model.py
3. Set Admin Password (required to unlock the Admin CRM tab)
export ADMIN_PASSWORD="your-strong-password"
(On Windows PowerShell: $env:ADMIN_PASSWORD="your-strong-password")
4. Run the app
streamlit run app.py
Production Notes

Database: By default, leads are saved to local SQLite (leads.db). For cloud deployments where local storage resets on restart, mount a persistent volume or switch DB_PATH/storage to a managed database (e.g., PostgreSQL or Supabase).

3D Renderer: Uses Three.js r128 via CDN, so an active internet connection is required for the 3D preview.