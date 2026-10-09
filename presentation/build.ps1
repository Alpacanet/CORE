param(
    [string]$RuntimeRoot = (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'),
    [string]$SkillRoot = $env:PRESENTATION_SKILL_DIR
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeNode = Join-Path $RuntimeRoot 'node\bin\node.exe'
$RuntimeModules = Join-Path $RuntimeRoot 'node\node_modules'
$RuntimePython = Join-Path $RuntimeRoot 'python\python.exe'
if (-not $SkillRoot) {
    $SkillVersions = Join-Path $env:USERPROFILE '.codex\plugins\cache\openai-primary-runtime\presentations'
    $LatestSkill = Get-ChildItem -LiteralPath $SkillVersions -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'skills\presentations\container_tools\artifact_tool_utils.mjs') } |
        Sort-Object { [version]$_.Name } -Descending |
        Select-Object -First 1
    if (-not $LatestSkill) { throw 'Presentations runtime not found. Pass -SkillRoot with the installed presentations skill path.' }
    $SkillRoot = Join-Path $LatestSkill.FullName 'skills\presentations'
}
foreach ($RequiredPath in @($RuntimeNode, $RuntimeModules, $RuntimePython, (Join-Path $SkillRoot 'container_tools\artifact_tool_utils.mjs'))) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) { throw "Missing presentation dependency: $RequiredPath" }
}
$BuildRoot = Join-Path $PSScriptRoot '.build'
$ModuleLink = Join-Path $BuildRoot 'node_modules'

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
if (-not (Test-Path -LiteralPath $ModuleLink)) {
    New-Item -ItemType Junction -Path $ModuleLink -Target $RuntimeModules | Out-Null
}

$env:PRESENTATION_SKILL_DIR = $SkillRoot
$env:PRESENTATION_RUNTIME_PYTHON = $RuntimePython
$env:PRESENTATION_PROJECT_ROOT = $ProjectRoot
$env:RUNTIME_NODE_MODULES = $RuntimeModules
$env:RUNTIME_BIN_DIR = Join-Path $RuntimeRoot 'bin\override'

$BuildScript = Join-Path $BuildRoot 'build.mjs'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build.mjs') -Destination $BuildScript -Force
& $RuntimeNode $BuildScript
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
