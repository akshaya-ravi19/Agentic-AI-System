@echo off
setlocal
rem Use GCP_PROJECT if PROJECT_ID not set
if "%PROJECT_ID%"=="" (
  if "%GCP_PROJECT%"=="" (
    echo Please set either PROJECT_ID or GCP_PROJECT environment variable before running this script.
    exit /b 1
  ) else (
    set "PROJECT_ID=%GCP_PROJECT%"
  )
)

rem Build container image using Cloud Build
gcloud builds submit --tag gcr.io/%PROJECT_ID%/food-safety-portal

rem Deploy to Cloud Run
gcloud run deploy food-safety-portal ^
    --image gcr.io/%PROJECT_ID%/food-safety-portal ^
    --platform managed ^
    --region us-central1 ^
    --allow-unauthenticated ^
    --port 8080
