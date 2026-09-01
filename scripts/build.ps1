<#
.SYNOPSIS
  Configure and build the engine on Windows, from any shell.

.DESCRIPTION
  The Machine B counterpart to scripts/build.sh.

  WHY THIS SCRIPT EXISTS

  On Windows, nvcc does not compile host code itself -- it hands it to cl.exe. So a
  CUDA build needs the MSVC environment (INCLUDE, LIB, PATH) present, which normally
  means remembering to open an "x64 Native Tools Command Prompt for VS 2022" rather
  than a plain PowerShell window.

  Forget, and the failure is actively misleading: CMake reports that it cannot find a
  working CUDA compiler. The obvious readings of that message are "CUDA is not
  installed" or "the toolkit is broken", and neither is true -- cl.exe is simply not on
  PATH. That is an easy hour to lose, and it recurs every time you open a new terminal.

  So this script locates the Visual Studio installation with vswhere, imports the
  vcvars64 environment into the current process if cl.exe is missing, then configures
  and builds. It works from a plain PowerShell prompt, from Windows Terminal, and from
  an IDE terminal.

  It also checks up front that this nvcc can emit the requested architecture, because
  the alternative is discovering at configure time that the toolkit predates sm_89.

.PARAMETER BuildType
  CMAKE_BUILD_TYPE. RelWithDebInfo by default: optimised, and still has the symbols
  that make a profile readable. Never benchmark a Debug build.

.PARAMETER Arch
  ENGINE_CUDA_ARCH. 89 is the RTX 4090 (Ada). See docs/adr/0005-cuda-arch-explicit.md.

.PARAMETER SyncCheck
  Synchronise after every kernel launch so a fault names the launch that caused it.
  Use a separate -BuildDir: this invalidates every benchmark number.

.EXAMPLE
  scripts\build.ps1
  scripts\build.ps1 -Test
  scripts\build.ps1 -Test -Bench
  scripts\build.ps1 -BuildDir build-sync -BuildType Debug -SyncCheck -Test
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
  [ValidateSet('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
  [string] $BuildType = 'RelWithDebInfo',
  [string] $BuildDir  = 'build',
  [string] $Arch      = '89',
  [switch] $SyncCheck,
  [switch] $WError,
  [switch] $Clean,
  [switch] $Test,
  [switch] $Bench
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Step { param([string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Warn { param([string] $Message) Write-Host "!!! $Message" -ForegroundColor Yellow }

# PowerShell's $ErrorActionPreference does not apply to native executables: cmake can
# exit 1 and the script will happily continue to the next line. Every external call
# goes through here so a failure actually stops the run.
function Invoke-Native {
  param(
    [Parameter(Mandatory)] [string]   $Exe,
    [Parameter(ValueFromRemainingArguments)] [string[]] $Arguments = @()
  )
  Write-Host "`$ $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Exe exited with code $LASTEXITCODE"
  }
}

#-----------------------------------------------------------------------------------
# MSVC environment
#-----------------------------------------------------------------------------------
function Import-MsvcEnvironment {
  if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    Write-Step 'cl.exe already on PATH (native tools prompt, or already imported)'
    return
  }

  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) {
    throw @"
cl.exe is not on PATH and vswhere.exe was not found at
  $vswhere

That means Visual Studio 2022 (or its Build Tools) is not installed. Install the
"Desktop development with C++" workload -- on Windows there is no CUDA path that
does not go through MSVC, and MinGW is not a substitute. See
docs/01-dev-environment.md, Machine B setup.
"@
  }

  # -products * matters: without it vswhere ignores Build Tools installations and only
  # finds the full IDE, which is the more common setup on a machine used purely for
  # compiling.
  $installPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
  if (-not $installPath) {
    throw 'vswhere found no Visual Studio installation with the C++ x64 toolset.'
  }

  $vcvars = Join-Path $installPath 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path $vcvars)) {
    throw "Found VS at '$installPath' but no vcvars64.bat -- the C++ workload is missing."
  }

  Write-Step "importing MSVC environment from $installPath"

  # vcvars64.bat can only set variables in a cmd process, so run it there, dump the
  # resulting environment with `set`, and copy it into this process. This is the
  # standard way to get the MSVC environment into a shell that is not cmd.
  $imported = 0
  cmd.exe /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
      try {
        Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] -ErrorAction Stop
        $imported++
      } catch {
        # A handful of cmd-only pseudo-variables cannot be set from PowerShell.
        # They are not ones any build needs.
      }
    }
  }
  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "Imported $imported variables from vcvars64.bat but cl.exe is still not on PATH."
  }
  Write-Step 'cl.exe found'
}

#-----------------------------------------------------------------------------------
# CUDA toolkit check
#-----------------------------------------------------------------------------------
function Test-CudaArch {
  param([string] $Requested)

  $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  if (-not $nvcc) {
    Write-Warn @"
nvcc is not on PATH. The build will configure as CPU-only, which builds and tests the
reference path but compiles none of the kernels -- on this machine that is a failure,
not a fallback. Install the CUDA Toolkit 12.x and reopen the shell.
"@
    return
  }

  $version = (& nvcc --version | Select-String -Pattern 'release ([0-9.]+)').Matches.Groups[1].Value
  Write-Step "nvcc $version"

  $arches = & nvcc --list-gpu-arch
  if ($arches -notcontains "compute_$Requested") {
    Write-Warn @"
This nvcc cannot emit compute_$Requested. sm_89 (Ada / RTX 4090) needs CUDA >= 11.8.

Do not work around this by passing a lower architecture. sm_86 code runs on a 4090
through PTX JIT, so it will appear to work while giving you neither Ada's instruction
set nor a benchmark that means anything. Upgrade the toolkit.
"@
  }
}

#-----------------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------------
Push-Location $RepoRoot
try {
  if ($BuildType -eq 'Debug') {
    Write-Warn 'Debug build: correct for debugging, useless for timing.'
  }
  if ($SyncCheck) {
    Write-Warn 'Sync-check ON: every launch synchronises, so benchmarks from this directory are invalid.'
  }

  Import-MsvcEnvironment
  Test-CudaArch -Requested $Arch

  if ($Clean -and (Test-Path $BuildDir)) {
    Write-Step "removing $BuildDir"
    Remove-Item -Recurse -Force $BuildDir
  }

  # Ninja only when creating the directory. Changing generators on an existing build
  # directory is a hard CMake error with an unhelpful message.
  $generatorArgs = @()
  if (-not (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))) {
    if (Get-Command ninja.exe -ErrorAction SilentlyContinue) {
      $generatorArgs = @('-G', 'Ninja')
    } else {
      Write-Warn 'ninja not found; falling back to the default generator (slower).'
    }
  }

  $jobs = if ($env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS } else { '4' }

  Write-Step "configuring ($BuildType) in $BuildDir"
  Invoke-Native cmake @(
    '-B', $BuildDir, '-S', '.'
    $generatorArgs
    "-DCMAKE_BUILD_TYPE=$BuildType"
    "-DENGINE_CUDA_ARCH=$Arch"
    "-DENGINE_SYNC_CHECK_KERNELS=$(if ($SyncCheck) {'ON'} else {'OFF'})"
    "-DENGINE_WERROR=$(if ($WError) {'ON'} else {'OFF'})"
  )

  Write-Step "building with $jobs jobs"
  Invoke-Native cmake @('--build', $BuildDir, '-j', $jobs)

  if ($Test) {
    # The golden files are gitignored generated data, so a fresh clone has none and
    # the numeric suites would skip -- reporting a green run that compared nothing.
    $golden = Join-Path $RepoRoot 'tests\golden'
    if (-not (Test-Path (Join-Path $golden '*.bin'))) {
      $python = @('python', 'python3', 'py') |
        Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } |
        Select-Object -First 1
      if ($python) {
        Write-Step 'generating reference data'
        Invoke-Native $python @('tools\gen_reference.py')
      } else {
        Write-Warn 'No Python on PATH: reference data cannot be generated and the cpu_ref and kernels suites will skip.'
      }
    }
    Write-Step 'ctest'
    Invoke-Native ctest @('--test-dir', $BuildDir, '--output-on-failure')
  }

  if ($Bench) {
    if ($BuildType -eq 'Debug') {
      throw 'Refusing to benchmark a Debug build. Re-run without -BuildType Debug.'
    }
    Write-Step 'bench_cpu_ref (same-machine baseline)'
    Invoke-Native (Join-Path $BuildDir 'bin\bench_cpu_ref.exe')

    $benchKernels = Join-Path $BuildDir 'bin\bench_kernels.exe'
    if (Test-Path $benchKernels) {
      Write-Step 'bench_kernels'
      Invoke-Native $benchKernels
    } else {
      Write-Warn 'bench_kernels was not built -- CUDA was not detected. Read the configure summary above.'
    }
    Write-Host ''
    Write-Host 'Lock the clocks before quoting any of these numbers, and paste the tables'
    Write-Host 'into docs\lab-notebook.md. Rows marked (!) are not trustworthy.'
  }

  Write-Step "done -- binaries in $BuildDir\bin"
} finally {
  Pop-Location
}
