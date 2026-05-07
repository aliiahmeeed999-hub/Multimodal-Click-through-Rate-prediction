# Requires: project files present under ./WWW2025_MMCTR_Challenge (OneDrive: use
# "Always keep on this device" on that folder and wait for sync to finish).
$ErrorActionPreference = 'Stop'
$Git = 'C:\Program Files\Git\mingw64\bin\git.exe'
if (-not (Test-Path $Git)) { $Git = 'git' }

Set-Location $PSScriptRoot
git config user.name 'aliiahmeeed999-hub'
git config user.email 'aliiahmeeed999-hub@users.noreply.github.com'
$env:GIT_AUTHOR_NAME = 'aliiahmeeed999-hub'
$env:GIT_AUTHOR_EMAIL = 'aliiahmeeed999-hub@users.noreply.github.com'
$env:GIT_COMMITTER_NAME = $env:GIT_AUTHOR_NAME
$env:GIT_COMMITTER_EMAIL = $env:GIT_AUTHOR_EMAIL

& $Git add -A
$status = & $Git status --porcelain
if (-not $status) {
    Write-Host 'Nothing to commit (working tree clean or no files under WWW2025_MMCTR_Challenge).'
    exit 0
}
& $Git commit -m 'Add full WWW2025 MM-CTR challenge pipeline' --no-verify
& $Git push -u origin main
