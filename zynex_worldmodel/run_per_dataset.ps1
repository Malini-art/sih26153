param(
    [string]$DatasetFolder = "data\cicids2018-csv\datasets",
    [int]$Epochs = 15,
    [int]$K = 5
)

if (-not (Test-Path $DatasetFolder)) {
    Write-Host "Dataset folder not found: $DatasetFolder" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path "data", "models", "outputs" | Out-Null

$csvFiles = Get-ChildItem -Path $DatasetFolder -Filter "*.csv"

if ($csvFiles.Count -eq 0) {
    Write-Host "No CSV files found in $DatasetFolder" -ForegroundColor Red
    exit 1
}

Write-Host "`nFound $($csvFiles.Count) dataset file(s):" -ForegroundColor Cyan
$csvFiles | ForEach-Object { Write-Host "  - $($_.Name)" }

$summary = @()

foreach ($file in $csvFiles) {
    $day = ($file.BaseName -split "_TrafficForML")[0]
    $day = $day -replace "[^a-zA-Z0-9\-]", ""

    Write-Host "`n=================================================================" -ForegroundColor Yellow
    Write-Host " Processing: $($file.Name)  ->  tag: $day" -ForegroundColor Yellow
    Write-Host "=================================================================" -ForegroundColor Yellow

    $features = "data\features_$day.npz"
    $model    = "models\world_model_$day.pt"
    $baseline = "models\baseline_$day.pkl"
    $bench    = "outputs\benchmark_$day.json"
    $predict  = "outputs\predict_$day.txt"

    Write-Host "`n[1/5] Feature extraction..." -ForegroundColor Green
    python src\feature_extraction.py --input "$($file.FullName)" --dataset cic-ids-2018 --out $features
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED - skipping $day" -ForegroundColor Red; continue }

    Write-Host "`n[2/5] Training world model ($Epochs epochs)..." -ForegroundColor Green
    python src\train.py --data $features --epochs $Epochs --out $model
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED - skipping $day" -ForegroundColor Red; continue }

    Write-Host "`n[3/5] Training baseline..." -ForegroundColor Green
    python src\baseline_model.py --data $features --out $baseline
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED - skipping $day" -ForegroundColor Red; continue }

    Write-Host "`n[4/5] Evaluating (world model vs baseline)..." -ForegroundColor Green
    python src\evaluate.py --data $features --model $model --baseline $baseline --out $bench
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED - skipping $day" -ForegroundColor Red; continue }

    Write-Host "`n[5/5] K-step prediction..." -ForegroundColor Green
    python src\predict_engine.py --data $features --model $model --k $K | Tee-Object -FilePath $predict
    if ($LASTEXITCODE -ne 0) { Write-Host "  FAILED - skipping $day" -ForegroundColor Red; continue }

    if (Test-Path $bench) {
        $summary += [PSCustomObject]@{ Day = $day; BenchmarkFile = $bench }
    }

    Write-Host "`n>>> Done with $day <<<" -ForegroundColor Cyan
}

Write-Host "`n`n=================================================================" -ForegroundColor Magenta
Write-Host " ALL DATASETS PROCESSED" -ForegroundColor Magenta
Write-Host "=================================================================" -ForegroundColor Magenta
$summary | Format-Table -AutoSize

Write-Host "`nPer-day results are in outputs\benchmark_<day>.json and outputs\predict_<day>.txt"
Write-Host "Run 'python src\summarize_results.py' to combine them into one comparison table.`n"