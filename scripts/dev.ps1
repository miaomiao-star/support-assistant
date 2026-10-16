$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
# 载入已保存的用户路径配置，兼容尚未重启的终端。
foreach ($name in @('UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'UV_PYTHON_CACHE_DIR', 'UV_PROJECT_ENVIRONMENT', 'TEMP', 'TMP')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if ($value) { [Environment]::SetEnvironmentVariable($name, $value, 'Process') }
}
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) { $uvPath = $uvCommand.Source }
else { $uvPath = Join-Path ([Environment]::GetEnvironmentVariable('UV_INSTALL_DIR', 'User')) 'uv.exe' }
& $uvPath run uvicorn app.main:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
