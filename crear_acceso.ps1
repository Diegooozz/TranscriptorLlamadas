$dir = $PSScriptRoot
$py = (python -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))").Trim()
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) "Tole Tole - Transcriptor.lnk"))
$lnk.TargetPath = $py
$lnk.Arguments = "`"$dir\transcriptor.py`""
$lnk.WorkingDirectory = $dir
$lnk.IconLocation = "$dir\assets\tole.ico"
$lnk.Description = "Transcriptor de llamadas - Tole Tole"
$lnk.Save()
Write-Output $lnk.FullName
