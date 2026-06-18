param(
    [Parameter(Mandatory=$true)]
    [string]$Url,
    [string]$Browser = "firefox",
    [string]$OutputDir = "$env:USERPROFILE\Downloads"
)

yt-dlp `
    --cookies-from-browser $Browser `
    --extract-audio `
    --audio-format mp3 `
    --audio-quality 0 `
    --output "$OutputDir\%(title)s.%(ext)s" `
    $Url
