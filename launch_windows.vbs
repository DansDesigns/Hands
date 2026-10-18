' Freehand Windows launcher: windowless by default, console only for debugging.
' Installer shortcut mode: cscript //nologo launch_windows.vbs --install-shortcut
Option Explicit

Const ForReading = 1
Dim fs, shell, root, pyDir, exe, guiPython, consolePython, mainScript
Dim appData, preferencesFile, debugEnabled, commandLine

Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fs.GetParentFolderName(WScript.ScriptFullName)
mainScript = fs.BuildPath(root, "main.py")
pyDir = fs.BuildPath(fs.BuildPath(root, ".venv"), "Scripts")
guiPython = fs.BuildPath(pyDir, "pythonw.exe")
consolePython = fs.BuildPath(pyDir, "python.exe")

' The installer uses this mode to create and VERIFY the shortcut without
' launching Freehand. Normal double-click startup remains unchanged.
If WScript.Arguments.Count > 0 Then
    If LCase(CStr(WScript.Arguments(0))) = "--install-shortcut" Then
        If InstallStartMenuShortcut() Then
            WScript.Echo "Freehand Start Menu shortcut installed successfully."
            WScript.Quit 0
        Else
            WScript.Echo "ERROR: Could not create/verify the Freehand Start Menu shortcut."
            WScript.Quit 1
        End If
    End If
End If

If Not fs.FileExists(mainScript) Then
    MsgBox "Freehand could not find main.py next to launch_windows.vbs.", vbCritical, "Freehand Launcher"
    WScript.Quit 1
End If
If Not fs.FileExists(guiPython) Or Not fs.FileExists(consolePython) Then
    MsgBox "Freehand's Python environment is missing. Run install_windows.bat first.", vbExclamation, "Freehand Launcher"
    WScript.Quit 1
End If

' Repair the shortcut quietly on normal launches if a profile was changed.
Call InstallStartMenuShortcut()

' Read the same per-user Debugging setting as main.py.
appData = shell.ExpandEnvironmentStrings("%APPDATA%")
preferencesFile = fs.BuildPath(fs.BuildPath(appData, "Freehand"), "ui_settings.json")
If Not fs.FileExists(preferencesFile) Then
    preferencesFile = fs.BuildPath(root, "ui_settings.json")
End If

debugEnabled = ReadDebugPreference(preferencesFile)
If debugEnabled Then
    exe = consolePython
Else
    exe = guiPython
End If

shell.CurrentDirectory = root
commandLine = Quote(exe) & " " & Quote(mainScript)
shell.Run commandLine, 1, False

Function Quote(value)
    Quote = Chr(34) & Replace(CStr(value), Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function

Function InstallStartMenuShortcut()
    Dim programs, shortcutFile, link, verifyLink, wscriptExe, iconFile
    Dim oldUnicode, oldPrototype
    InstallStartMenuShortcut = False
    On Error Resume Next
    Err.Clear
    ' The Windows shell resolves the actual per-user Programs directory,
    ' including redirected/roaming Start Menu locations.
    programs = shell.SpecialFolders("Programs")
    If Err.Number <> 0 Or Len(programs) = 0 Then
        Err.Clear
        Exit Function
    End If
    If Not fs.FolderExists(programs) Then
        Err.Clear
        Exit Function
    End If

    ' Use a plain ASCII file name: it reliably appears in Windows Start search.
    ' Windows prohibits ':' inside a file name.
    shortcutFile = fs.BuildPath(programs, "Freehand - Spatial Computing.lnk")
    wscriptExe = fs.BuildPath(shell.ExpandEnvironmentStrings("%SystemRoot%"), "System32\wscript.exe")
    Set link = shell.CreateShortcut(shortcutFile)
    If Err.Number <> 0 Then
        Err.Clear
        Exit Function
    End If
    link.TargetPath = wscriptExe
    link.Arguments = Quote(WScript.ScriptFullName)
    link.WorkingDirectory = root
    link.Description = "Freehand: Spatial Computing"
    iconFile = fs.BuildPath(root, "Freehand_logo_small.ico")
    If fs.FileExists(iconFile) Then link.IconLocation = iconFile & ",0"
    Err.Clear
    link.Save
    If Err.Number <> 0 Then
        Err.Clear
        Exit Function
    End If
    If Not fs.FileExists(shortcutFile) Then Exit Function

    Err.Clear
    Set verifyLink = shell.CreateShortcut(shortcutFile)
    If Err.Number <> 0 Then
        Err.Clear
        Exit Function
    End If
    If LCase(CStr(verifyLink.Arguments)) <> LCase(Quote(WScript.ScriptFullName)) Then Exit Function
    If LCase(CStr(verifyLink.TargetPath)) <> LCase(wscriptExe) Then Exit Function
    InstallStartMenuShortcut = True

    ' Remove superseded names only after the new link has been verified.
    oldUnicode = fs.BuildPath(programs, "Freehand" & ChrW(&HFF1A) & " Spatial Computing.lnk")
    oldPrototype = fs.BuildPath(programs, "Spatial Desktop Prototype.lnk")
    If fs.FileExists(oldUnicode) Then fs.DeleteFile oldUnicode, True
    If fs.FileExists(oldPrototype) Then fs.DeleteFile oldPrototype, True
    Err.Clear
    On Error GoTo 0
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
