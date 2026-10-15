[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('2023', '2024')]
    [string] $Fold,
    [switch] $Worker,
    [string] $LaunchToken
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-NewJson {
    param([string] $Path, [object] $Value)
    $text = ($Value | ConvertTo-Json -Depth 20) + [Environment]::NewLine
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    try {
        $bytes = $encoding.GetBytes($text)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush()
    }
    finally {
        $stream.Dispose()
    }
}

function Assert-SafeOutputDirectories {
    param([string] $Study, [string] $Background)
    foreach ($directory in @($Study, $Background)) {
        if (-not (Test-Path -LiteralPath $directory -PathType Container) -or
            (((Get-Item -LiteralPath $directory).Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw 'Supervisor output directories must exist and cannot be links or junctions'
        }
    }
}

function Invoke-TrackedChild {
    param(
        [string] $Executable,
        [string] $Arguments,
        [string] $WorkingDirectory,
        [string] $Stdout,
        [string] $Stderr,
        [string] $StartedReceipt,
        [string] $ExitReceipt,
        [hashtable] $Identity
    )
    foreach ($path in @($Stdout, $Stderr, $StartedReceipt, $ExitReceipt)) {
        if (Test-Path -LiteralPath $path) { throw "Process output already exists: $path" }
    }
    $started = [DateTimeOffset]::UtcNow.ToString('o')
    $child = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
    try {
        $handle = $child.Handle
        $common = @{}
        foreach ($key in $Identity.Keys) { $common[$key] = $Identity[$key] }
        $common['child_pid'] = $child.Id
        $common['supervisor_pid'] = $PID
        $common['executable'] = $Executable
        $common['arguments'] = $Arguments
        $common['started_utc'] = $started
        $common['stdout_path'] = $Stdout
        $common['stderr_path'] = $Stderr
        $startError = $null
        try { Write-NewJson -Path $StartedReceipt -Value (@{ status = 'child_started' } + $common) }
        catch { $startError = $_.Exception.Message }
        $child.WaitForExit()
        $child.Refresh()
        if ($null -eq $child.ExitCode) { throw 'A waited child process did not expose an exit code' }
        $receipt = @{ status = 'child_exited'; exit_code = [int] $child.ExitCode
            finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
            exit_source = 'System.Diagnostics.Process.WaitForExit/ExitCode'
            root_observed_exit = $false; started_receipt_error = $startError } + $common
        Write-NewJson -Path $ExitReceipt -Value $receipt
        return $receipt
    }
    finally {
        try { if (-not $child.HasExited) { $child.WaitForExit() } }
        finally { $child.Dispose() }
    }
}

function Assert-PreparedStudy {
    param([string] $Repository, [string] $Study, [string] $Python, [string] $ExpectedProtocol, [string] $WorkerFold)
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Fixed AShareGPU interpreter is missing' }
    if (-not (Test-Path -LiteralPath $Study -PathType Container)) { throw 'Repair study has not been prepared' }
    $item = Get-Item -LiteralPath $Study
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Study root cannot be a link or junction' }
    $protocolPath = Join-Path $Study 'protocol.json'
    $statusPath = Join-Path $Study 'status.json'
    if (-not (Test-Path -LiteralPath $protocolPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $statusPath -PathType Leaf)) { throw 'Prepared protocol/status are missing' }
    if ((Get-FileHash -LiteralPath $protocolPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedProtocol) {
        throw 'Prepared registration SHA differs from the root-approved study'
    }
    $status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($status.status -ne 'prepared' -or $status.test_evaluated -ne $false -or $status.actionable -ne $false) {
        throw 'Study is not in its prepared, protected state'
    }
    if (Test-Path -LiteralPath (Join-Path $Study ('worker_screen_' + $WorkerFold + '.json'))) {
        throw 'This fold already has a framework worker receipt'
    }
    $runs = Join-Path $Study 'runs'
    if ((Test-Path -LiteralPath $runs) -and
        @(Get-ChildItem -LiteralPath $runs -Directory -Filter ('screen_' + $WorkerFold + '_*')).Count -gt 0) {
        throw 'This fold already has a complete or partial physical run'
    }
    $guard = @'
import sys
from src.data.preprocess import read_config
from tools.research_timing_phase_budget_repaired import check_frozen
check_frozen(read_config(sys.argv[1]))
'@
    Push-Location -LiteralPath $Repository
    try {
        & $Python -B -c $guard $protocolPath
        if ($LASTEXITCODE -ne 0) { throw "Read-only frozen registration guard failed: $LASTEXITCODE" }
    }
    finally { Pop-Location }
}

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$study = [System.IO.Path]::GetFullPath((Join-Path $repository 'reports\runs\ppo_timing_phase_state_repair_20261015'))
$original = [System.IO.Path]::GetFullPath((Join-Path $repository 'reports\runs\ppo_timing_phase_budget_20261015'))
if ($study -eq $original -or $study.StartsWith($original + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase) -or $original.StartsWith($study + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Repair output overlaps the original study' }
$python = 'D:\ProgramFiles\Anaconda3\envs\AShareGPU\python.exe'
$registrationSha = '24e7e54419bbc98c97c232187d765ff041be02cbe10f4709fc7db86205785a46'
$launcherPath = $PSCommandPath
$launcherSha = (Get-FileHash -LiteralPath $launcherPath -Algorithm SHA256).Hash.ToLowerInvariant()
$driver = Join-Path $repository 'tools\research_timing_phase_budget_repaired.py'
$arguments = '-B -u "{0}" --stage screen --output "{1}" --fold {2}' -f $driver, $study, $Fold
$background = Join-Path $study 'background'
$prefix = 'screen_' + $Fold
$claimPath = Join-Path $background ($prefix + '.launch.json')

if (-not $Worker) {
    if ($LaunchToken) { throw 'LaunchToken belongs only to the supervisor worker' }
    Assert-PreparedStudy -Repository $repository -Study $study -Python $python -ExpectedProtocol $registrationSha -WorkerFold $Fold
    if (-not (Test-Path -LiteralPath $background)) {
        $null = New-Item -ItemType Directory -Path $background
    }
    Assert-SafeOutputDirectories -Study $study -Background $background
    if (@(Get-ChildItem -LiteralPath $background -Filter ($prefix + '.*')).Count -gt 0) {
        throw 'This fold already has a launch, log, lock or receipt; explicit review is required'
    }
    $token = [Guid]::NewGuid().ToString('D')
    Write-NewJson -Path $claimPath -Value @{ schema_version = 1; status = 'launch_claimed'; stage = 'screen'; fold = $Fold
        launch_token = $token; launcher_sha256 = $launcherSha; registration_sha256 = $registrationSha
        requested_utc = [DateTimeOffset]::UtcNow.ToString('o'); executable = $python; arguments = $arguments }
    $shell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $supervisorArgs = '-NoProfile -NonInteractive -File "{0}" -Fold {1} -Worker -LaunchToken {2}' -f $launcherPath, $Fold, $token
    $supervisor = Start-Process -FilePath $shell -ArgumentList $supervisorArgs -WorkingDirectory $repository -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $background ($prefix + '.supervisor.stdout.log')) `
        -RedirectStandardError (Join-Path $background ($prefix + '.supervisor.stderr.log')) -PassThru
    try {
        $receiptPath = Join-Path $background ($prefix + '.supervisor.json')
        $receipt = @{ schema_version = 1; status = 'supervisor_started'; stage = 'screen'; fold = $Fold
            supervisor_pid = $supervisor.Id; launch_token = $token; launcher_sha256 = $launcherSha
            registration_sha256 = $registrationSha; created_utc = [DateTimeOffset]::UtcNow.ToString('o')
            launch_claim_path = $claimPath; child_exit_observed = $false }
        Write-NewJson -Path $receiptPath -Value $receipt
        $receipt['receipt_path'] = $receiptPath
        $receipt | ConvertTo-Json -Depth 8
    }
    finally { $supervisor.Dispose() }
    return
}

$lock = $null
try {
    Assert-SafeOutputDirectories -Study $study -Background $background
    $parsedToken = [Guid]::Empty
    if (-not [Guid]::TryParse($LaunchToken, [ref] $parsedToken)) { throw 'Supervisor requires its parent launch token' }
    $claim = Get-Content -LiteralPath $claimPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($claim.launch_token -ne $LaunchToken -or $claim.fold -ne $Fold -or $claim.stage -ne 'screen' -or
        $claim.launcher_sha256 -ne $launcherSha -or $claim.registration_sha256 -ne $registrationSha -or
        $claim.executable -ne $python -or $claim.arguments -ne $arguments) { throw 'Supervisor claim identity mismatch' }
    Assert-PreparedStudy -Repository $repository -Study $study -Python $python -ExpectedProtocol $registrationSha -WorkerFold $Fold
    $lockPath = Join-Path $background ($prefix + '.' + $LaunchToken + '.lock')
    $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $identity = @{ schema_version = 1; stage = 'screen'; fold = $Fold; launch_token = $LaunchToken
        launcher_sha256 = $launcherSha; registration_sha256 = $registrationSha }
    $result = Invoke-TrackedChild -Executable $python -Arguments $arguments -WorkingDirectory $repository `
        -Stdout (Join-Path $background ($prefix + '.stdout.log')) -Stderr (Join-Path $background ($prefix + '.stderr.log')) `
        -StartedReceipt (Join-Path $background ($prefix + '.started.json')) `
        -ExitReceipt (Join-Path $background ($prefix + '.exit.json')) -Identity $identity
    exit $result.exit_code
}
catch {
    $failure = $_
    try {
        Assert-SafeOutputDirectories -Study $study -Background $background
        $errorPath = Join-Path $background ($prefix + '.supervisor_error.json')
        Write-NewJson -Path $errorPath -Value @{ schema_version = 1; status = 'supervisor_failed'; stage = 'screen'; fold = $Fold
            supervisor_pid = $PID; launch_token = $LaunchToken; launcher_sha256 = $launcherSha
            registration_sha256 = $registrationSha; error = $failure.Exception.Message; child_exit_code = $null
            created_utc = [DateTimeOffset]::UtcNow.ToString('o'); root_observed_exit = $false }
    }
    catch { [Console]::Error.WriteLine('Supervisor failure receipt was not written: ' + $_.Exception.Message) }
    throw $failure
}
finally {
    if ($null -ne $lock) { $lock.Dispose() }
}
