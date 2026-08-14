## run-deep-dive.ps1
## Processes each paper through Claude to extract deep technical details.
## Usage:
##   .\run-deep-dive.ps1                    # process all papers
##   .\run-deep-dive.ps1 -Filter "wang*"    # process matching papers only
##   .\run-deep-dive.ps1 -DryRun            # list files without processing

param(
    [string]$Filter = "*.md",
    [switch]$DryRun,
    [string]$Model = "sonnet",
    [string]$PaperDir = "D:\adags\research-wiki\papers",
    [string]$PromptFile = "D:\adags\research-wiki\deep-dive-prompt.txt"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PromptFile)) {
    Write-Error "Prompt file not found: $PromptFile"
    exit 1
}

$basePrompt = Get-Content $PromptFile -Raw -Encoding utf8
$files = Get-ChildItem -Path $PaperDir -Filter $Filter | Where-Object { $_.Name -ne "README.md" }
$total = $files.Count
$logDir = "D:\adags\research-wiki\deep-dive-logs"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Host "Found $total paper files to process." -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`nDry run - files that would be processed:" -ForegroundColor Yellow
    foreach ($f in $files) {
        Write-Host "  $($f.FullName)"
    }
    exit 0
}

$succeeded = 0
$failed = 0
$skipped = 0
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$summaryLog = Join-Path $logDir "run-$timestamp.log"
$tempPrompt = Join-Path $env:TEMP "deep-dive-prompt-current.txt"

foreach ($f in $files) {
    $i = $succeeded + $failed + $skipped + 1
    $name = $f.Name
    Write-Host "`n[$i/$total] Processing: $name" -ForegroundColor Cyan

    $filled = $basePrompt -replace '\{\{TARGET_FILE\}\}', $f.FullName
    $filled | Out-File -FilePath $tempPrompt -Encoding utf8 -NoNewline

    $logFile = Join-Path $logDir "$($f.BaseName).log"

    try {
        $output = Get-Content $tempPrompt -Raw -Encoding utf8 | claude -p - --model $Model --allowedTools "Edit,Read,WebFetch,WebSearch,Bash,Grep,Glob,Write" 2>&1
        $output | Out-File -FilePath $logFile -Encoding utf8

        if ($output -match "SKIPPED:|already contains") {
            Write-Host "  SKIPPED" -ForegroundColor Yellow
            "$name`tSKIPPED" | Add-Content -Path $summaryLog -Encoding utf8
            $skipped++
        } else {
            Write-Host "  Done" -ForegroundColor Green
            "$name`tSUCCESS" | Add-Content -Path $summaryLog -Encoding utf8
            $succeeded++
        }
    } catch {
        Write-Host "  FAILED: $_" -ForegroundColor Red
        "$name`tFAILED`t$_" | Add-Content -Path $summaryLog -Encoding utf8
        $failed++
    }
}

if (Test-Path $tempPrompt) { Remove-Item $tempPrompt -Force }

Write-Host "`n--- Summary ---" -ForegroundColor Cyan
Write-Host "Succeeded: $succeeded" -ForegroundColor Green
Write-Host "Skipped:   $skipped" -ForegroundColor Yellow
Write-Host "Failed:    $failed" -ForegroundColor Red
Write-Host "Log:       $summaryLog"
