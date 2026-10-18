' Freehand Windows launcher: no console is created in normal mode.
' Keep this file beside main.py, in the Hands directory.
Option Explicit

Const ForReading = 1
Dim fs, shell, root, pyDir, exe, guiPython, consolePython, mainScript
Dim appData, preferencesFile, debugEnabled, startupFolder, shortcutFile, link
Dim wscriptExe, iconFile, commandLine

Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fs.GetParentFolderName(WScript.ScriptFullName)
mainScript = fs.BuildPath(root, "main.py")
pyDir = fs.BuildPath(fs.BuildPath(root, ".venv"), "Scripts")
guiPython = fs.BuildPath(pyDir, "pythonw.exe")
consolePython = fs.BuildPath(pyDir, "python.exe")

If Not fs.FileExists(mainScript) Then
    MsgBox "Freehand could not find main.py next to launch_windows.vbs.", vbCritical, "Freehand Launcher"
    WScript.Quit 1
End If
If Not fs.FileExists(guiPython) Or Not fs.FileExists(consolePython) Then
    MsgBox "Freehand's Python environment is missing. Run install_windows.bat once, then start Freehand with launch_windows.vbs.", vbExclamation, "Freehand Launcher"
    WScript.Quit 1
End If

' Read the same per-user preference as main.py. Use old local settings before migration.
appData = shell.ExpandEnvironmentStrings("%APPDATA%")
preferencesFile = fs.BuildPath(fs.BuildPath(appData, "Freehand"), "ui_settings.json")
If Not fs.FileExists(preferencesFile) Then
    preferencesFile = fs.BuildPath(root, "ui_settings.json")
End If

debugEnabled = ReadDebugPreference(preferencesFile)

' Upgrade this application's Start Menu shortcut without running the installer.
' WScript (not CMD or python.exe) launches the app on subsequent starts.
On Error Resume Next
startupFolder = fs.BuildPath(appData, "Microsoft\Windows\Start Menu\Programs")
If fs.FolderExists(startupFolder) Then
    shortcutFile = fs.BuildPath(startupFolder, "Freehand" & ChrW(&HFF1A) & " Spatial Computing.lnk")
    Set link = shell.CreateShortcut(shortcutFile)
    wscriptExe = fs.BuildPath(shell.ExpandEnvironmentStrings("%SystemRoot%"), "System32\wscript.exe")
    link.TargetPath = wscriptExe
    link.Arguments = Quote(WScript.ScriptFullName)
    link.WorkingDirectory = root
    link.Description = "Freehand: Spatial Computing"
    iconFile = fs.BuildPath(root, "Freehand_logo_small.ico")
    If fs.FileExists(iconFile) Then link.IconLocation = iconFile & ",0"
    link.Save
End If
Err.Clear
On Error GoTo 0

If debugEnabled Then
    ' A console is created only by this explicit, user-selected mode.
    exe = consolePython
Else
    ' pythonw.exe has the Windows GUI subsystem: no CMD window exists at all.
    exe = guiPython
End If

shell.CurrentDirectory = root
commandLine = Quote(exe) & " " & Quote(mainScript)
' SW_SHOWNORMAL is safe for Qt's fullscreen window; pythonw has no console.
shell.Run commandLine, 1, False

Function Quote(value)
    Quote = Chr(34) & Replace(CStr(value), Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function

Function ReadDebugPreference(filePath)
    Dim inputFile, jsonText, matcher
    ReadDebugPreference = False
    If Not fs.FileExists(filePath) Then Exit Function
    On Error Resume Next
    Set inputFile = fs.OpenTextFile(filePath, ForReading, False)
    If Err.Number <> 0 Then
        Err.Clear
        Exit Function
    End If
    jsonText = inputFile.ReadAll
    inputFile.Close
    If Err.Number <> 0 Then
        Err.Clear
        Exit Function
    End If
    On Error GoTo 0
    Set matcher = CreateObject("VBScript.RegExp")
    matcher.IgnoreCase = False
    matcher.Global = False
    matcher.Pattern = Chr(34) & "debug_console" & Chr(34) & "\s*:\s*true\b"
    ReadDebugPreference = matcher.Test(jsonText)
End Function
