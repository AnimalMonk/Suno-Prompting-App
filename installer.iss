; Inno Setup Script for Suno Prompt Generator

[Setup]
AppName=Suno Prompt Generator
AppVersion=1.1.1
AppPublisher=AnimalMonk
AppPublisherURL=https://github.com/AnimalMonk/Suno-Prompting-App
DefaultDirName={autopf}\SunoPromptGenerator
DefaultGroupName=Suno Prompt Generator
OutputBaseFilename=SunoPromptGenerator_Setup
OutputDir=Output
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

[Files]
Source: "dist\SunoPromptGenerator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Suno Prompt Generator"; Filename: "{app}\SunoPromptGenerator.exe"
Name: "{autodesktop}\Suno Prompt Generator"; Filename: "{app}\SunoPromptGenerator.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\SunoPromptGenerator.exe"; Description: "Launch Suno Prompt Generator"; Flags: nowait postinstall skipifsilent

[Code]
var
  ApiKeyPage: TInputQueryWizardPage;

procedure InitializeWizard();
begin
  ApiKeyPage := CreateInputQueryPage(wpSelectDir,
    'API Configuration',
    'Enter your OpenRouter API key to use the application.',
    'You need an OpenRouter API key. Get one at https://openrouter.ai/keys'#13#10 +
    'This will be saved locally in the install directory.');
  ApiKeyPage.Add('OpenRouter API Key:', False);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = ApiKeyPage.ID then
  begin
    if Trim(ApiKeyPage.Values[0]) = '' then
    begin
      MsgBox('Please enter your OpenRouter API key to continue.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  EnvContent: String;
begin
  if CurStep = ssPostInstall then
  begin
    EnvContent := 'OPENROUTER_API_KEY=' + Trim(ApiKeyPage.Values[0]);
    SaveStringToFile(ExpandConstant('{app}\.env'), EnvContent, False);
  end;
end;
