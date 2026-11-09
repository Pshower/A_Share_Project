<#
.SYNOPSIS
    Launch opencode for this repository through the local Clash proxy.

.DESCRIPTION
    opencode is a Bun binary.  It honours the HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
    environment variables but ignores the Windows system proxy in
    HKCU\...\Internet Settings, which is what browsers and .NET use.  On a machine
    where Clash is enabled only as the system proxy, opencode therefore goes direct.

    This script exports the proxy variables for its own child process only -- nothing
    is written to the user environment, the registry, or the shared background
    service -- and then launches opencode.

    One nuance drives the --standalone injection below: model traffic is issued by the
    opencode *server* process, not by the CLI front end.  Without --standalone the CLI
    hands the request to the already-running background service, which was started
    without these variables, so exporting them would have no effect.  With
    --standalone the private server is a child of this script and inherits them.

.PARAMETER Direct
    Do not set any proxy variable; run opencode exactly as it would run normally.

.PARAMETER Check
    Resolve the proxy, test that the port accepts connections, print the result, exit.

.PARAMETER Quiet
    Suppress the one-line notice printed before launching.

.PARAMETER Proxy
    Explicit proxy URL.  Defaults to the Windows system proxy, then 127.0.0.1:7897.

.EXAMPLE
    .\tools\opencode_proxy.ps1
    .\tools\opencode_proxy.ps1 run --archived --model opencode-go/deepseek-v4.1-flash "hi"
    .\tools\opencode_proxy.ps1 -Check
    .\tools\opencode_proxy.ps1 -Direct run "hi"
#>

$ErrorActionPreference = 'Stop'

# Hand-rolled argument parsing: a param() block would try to bind opencode's own
# flags (--archived, --model, ...) as PowerShell parameters and fail.
$forwarded = @()
$proxyUrl = $null
$direct = $false
$quiet = $false
$check = $false

$i = 0
while ($i -lt $args.Count) {
    $token = [string]$args[$i]
    switch -Regex ($token) {
        '^--?(direct|noproxy)$' { $direct = $true; $i += 1; break }
        '^--?(quiet)$' { $quiet = $true; $i += 1; break }
        '^--?(check)$' { $check = $true; $i += 1; break }
        '^--?(proxy)(=|$)' {
            if ($token -match '=(.+)$') { $proxyUrl = $Matches[1]; $i += 1 }
            else { $proxyUrl = [string]$args[$i + 1]; $i += 2 }
            break
        }
        default { $forwarded += $token; $i += 1 }
    }
}

function Resolve-SystemProxy {
    <#
        Read the WinINET proxy that Clash writes when "system proxy" is enabled.
        ProxyServer is either "host:port" or "http=host:port;https=host:port".
    #>
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
    $settings = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
    if (-not $settings) { return $null }
    if ($settings.PSObject.Properties.Name -notcontains 'ProxyEnable') { return $null }
    if ([int]$settings.ProxyEnable -ne 1) { return $null }

    $raw = [string]$settings.ProxyServer
    if ([string]::IsNullOrWhiteSpace($raw)) { return $null }

    $candidate = $null
    foreach ($part in ($raw -split ';')) {
        $entry = $part.Trim()
        if ([string]::IsNullOrWhiteSpace($entry)) { continue }
        if ($entry -match '^\s*(https|http|socks)\s*=\s*(.+)$') {
            $scheme = $Matches[1].ToLowerInvariant()
            if (-not $candidate -or $scheme -eq 'http') { $candidate = $Matches[2].Trim() }
        }
        elseif (-not $candidate) {
            $candidate = $entry
        }
    }
    if ([string]::IsNullOrWhiteSpace($candidate)) { return $null }
    return $candidate
}

function Format-ProxyUrl {
    param([string] $Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    $text = $Value.Trim()
    if ($text -match '^[a-zA-Z][a-zA-Z0-9+.-]*://') { return $text }
    return "http://$text"
}

function Test-ProxyPort {
    param([string] $Url, [int] $TimeoutMs = 1500)
    try {
        $uri = [System.Uri] $Url
    }
    catch {
        return $false
    }
    if ($uri.Port -le 0) { return $false }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($uri.Host, $uri.Port)
        if (-not $task.Wait($TimeoutMs)) { return $false }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

$resolved = $null
if (-not $direct) {
    if (-not [string]::IsNullOrWhiteSpace($proxyUrl)) {
        $resolved = Format-ProxyUrl $proxyUrl
    }
    else {
        $resolved = Format-ProxyUrl (Resolve-SystemProxy)
        if (-not $resolved) { $resolved = 'http://127.0.0.1:7897' }
    }
}

if ($check) {
    if ($direct) {
        Write-Output 'proxy: none (-Direct)'
        exit 0
    }
    Write-Output "proxy: $resolved"
    if (Test-ProxyPort -Url $resolved) {
        Write-Output 'port: accepting connections'
        exit 0
    }
    Write-Output 'port: NOT reachable -- is Clash running with the system proxy on?'
    exit 1
}

# Prefer the real binary over the npm .ps1 shim, and never resolve back to the
# opencode.cmd launcher that sits next to this repository (that would recurse).
$repoRoot = Split-Path -Parent $PSScriptRoot
$candidates = @(Get-Command opencode -All -ErrorAction SilentlyContinue | Where-Object {
        $path = if ($_.Path) { $_.Path } else { $_.Source }
        $path -and -not $path.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)
    })
$opencode = $candidates | Where-Object { $_.CommandType -eq 'Application' } | Select-Object -First 1
if (-not $opencode) { $opencode = $candidates | Select-Object -First 1 }
if (-not $opencode) {
    Write-Error 'opencode was not found on PATH.'
    exit 127
}

# The private server must inherit the variables, so --standalone is injected for the
# commands that start one.  Local-only subcommands keep their arguments untouched.
if (-not $direct -and $resolved) {
    $first = if ($forwarded.Count -gt 0) { [string]$forwarded[0] } else { '' }
    $startsServer = ($first -eq '') -or (@('run', 'mini', 'acp', 'serve') -contains $first)
    $alreadyStandalone = $forwarded -contains '--standalone'
    $alreadyServer = ($forwarded -contains '--server')
    if ($startsServer -and -not $alreadyStandalone -and -not $alreadyServer) {
        $forwarded = @('--standalone') + $forwarded
    }

    $env:HTTP_PROXY = $resolved
    $env:HTTPS_PROXY = $resolved
    $env:ALL_PROXY = $resolved
    $env:http_proxy = $resolved
    $env:https_proxy = $resolved
    $env:all_proxy = $resolved
    # Keep loopback traffic (the CLI talking to its own server) out of the proxy.
    $env:NO_PROXY = 'localhost,127.0.0.1,::1'
    $env:no_proxy = 'localhost,127.0.0.1,::1'

    if (-not (Test-ProxyPort -Url $resolved)) {
        Write-Warning "proxy $resolved is not accepting connections; opencode will hang on its first request. Run with -Direct to bypass."
    }
    if (-not $quiet) {
        $how = if ($forwarded -contains '--standalone') { 'private server' } else { 'existing server' }
        Write-Host "opencode -> $resolved ($how)" -ForegroundColor DarkGray
    }
}

$exe = if ($opencode.Path) { $opencode.Path } else { $opencode.Source }
& $exe @forwarded
exit $LASTEXITCODE
