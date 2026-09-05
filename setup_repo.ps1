# setup_repo.ps1 - FacilityPath Git repository bootstrap (ASCII only)
# Run once in C:\Work\FacilityPath after installing Git and (optionally) GitHub CLI.
$ErrorActionPreference = "Stop"
if (-not (Test-Path "facilitypath.py")) { Write-Host "[X] Run in the project folder."; exit 1 }
git --version | Out-Null
if (-not (Test-Path ".git")) {
    git init -b main
    git config core.autocrlf true          # Windows: CRLF on checkout, LF in repo
    git config core.quotepath false        # Show Korean file names correctly
    Write-Host "[OK] git init (branch: main)"
}
git add .
git commit -m "[init] FacilityPath v1.1 - modules, verify scripts, Claude Code config" 2>$null
Write-Host "[OK] initial commit"
Write-Host ""
Write-Host "Next (choose one):"
Write-Host "  A) GitHub CLI :  gh auth login  ;  gh repo create droneid-facilitypath --private --source . --push"
Write-Host "  B) Manual     :  create a private repo on github.com, then"
Write-Host "                   git remote add origin https://github.com/<org>/droneid-facilitypath.git"
Write-Host "                   git push -u origin main"
