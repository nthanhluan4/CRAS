# PowerShell launcher for CRAS on Windows
param (
    [string]$Dataset = "mvtec", # "mvtec", "itdd", "visa", "mpdd"
    [string]$DataPath = "",     # Path to dataset root directory
    [string]$Mode = "ckpt",     # "ckpt" (train + eval) or "test" (eval only)
    [string]$Setting = "multi", # "multi" or "single"
    [int]$BatchSize = 16,
    [int]$NumWorkers = 0,       # 0 is recommended on Windows to avoid multiprocessing issues
    [int]$MetaEpochs = 100,
    [string]$Backbone = "wideresnet50",
    [string[]]$Classes = @()
)

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "Please create it using: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Default classes for datasets if not specified
if ($Classes.Length -eq 0) {
    switch ($Dataset.ToLower()) {
        "itdd" {
            $Classes = @("cotton_fabric", "dyed_fabric", "hemp_fabric", "plaid_fabric")
        }
        "mvtec" {
            $Classes = @("carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper")
        }
        "visa" {
            $Classes = @("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
        }
        "mpdd" {
            $Classes = @("bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes")
        }
        default {
            Write-Host "[ERROR] Unknown dataset '$Dataset'. Please pass -Classes explicitly." -ForegroundColor Red
            exit 1
        }
    }
}

if ([string]::IsNullOrWhiteSpace($DataPath)) {
    Write-Host "[WARNING] No -DataPath provided. Please provide the dataset directory path, e.g.:" -ForegroundColor Yellow
    Write-Host ".\run_cras.ps1 -Dataset itdd -DataPath D:\Datasets\ITDD" -ForegroundColor Cyan
    exit 1
}

if (-not (Test-Path $DataPath)) {
    Write-Host "[ERROR] DataPath '$DataPath' does not exist." -ForegroundColor Red
    exit 1
}

# Build dataset flags (-d class1 -d class2 ...)
$FlagArgs = @()
foreach ($cls in $Classes) {
    $FlagArgs += "-d"
    $FlagArgs += $cls
}

Write-Host "=== Starting CRAS ===" -ForegroundColor Green
Write-Host "Dataset: $Dataset"
Write-Host "DataPath: $DataPath"
Write-Host "Mode: $Mode"
Write-Host "Setting: $Setting"
Write-Host "Classes: $($Classes -join ', ')"
Write-Host "Python: $VenvPython"
Write-Host "====================="

$ArgsList = @(
    "main.py",
    "--gpu", "0",
    "--seed", "0",
    "--test", $Mode,
    "net",
    "-b", $Backbone,
    "-le", "layer2",
    "-le", "layer3",
    "--pretrain_embed_dimension", "1536",
    "--target_embed_dimension", "1536",
    "--patchsize", "3",
    "--meta_epochs", "$MetaEpochs",
    "--eval_epochs", "1",
    "--dsc_layers", "3",
    "--pre_proj", "1",
    "--noise", "0.015",
    "--k", "0.3",
    "--limit", "-1",
    "dataset",
    "--setting", $Setting,
    "--batch_size", "$BatchSize",
    "--resize", "329",
    "--imagesize", "288",
    "--num_workers", "$NumWorkers"
) + $FlagArgs + @($Dataset, $DataPath)

& $VenvPython $ArgsList
