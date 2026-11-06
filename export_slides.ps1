$pptxPath = "c:\Users\Dinkesh\Downloads\BTP\BTP_Defense_Presentation_Dinkesh_Rithvik.pptx"
$imgDir = "c:\Users\Dinkesh\Downloads\BTP\slides_img"
if (-not (Test-Path $imgDir)) { New-Item -ItemType Directory -Path $imgDir | Out-Null }

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($pptxPath, [Microsoft.Office.Core.MsoTriState]::msoTrue, [Microsoft.Office.Core.MsoTriState]::msoFalse, [Microsoft.Office.Core.MsoTriState]::msoFalse)
$pres.Export($imgDir, "PNG", 1920, 1080)
$pres.Close()
$ppt.Quit()
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()
Write-Host "Exported slide images to $imgDir successfully!"
