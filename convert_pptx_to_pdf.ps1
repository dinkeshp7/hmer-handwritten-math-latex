$pptxPath = "c:\Users\Dinkesh\Downloads\BTP\BTP_Defense_Presentation_Dinkesh_Rithvik.pptx"
$pdfPath = "c:\Users\Dinkesh\Downloads\BTP\BTP_Defense_Presentation_Dinkesh_Rithvik.pdf"

Write-Host "Converting PPTX to PDF using PowerPoint COM..."
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    # Open presentation without visible window
    $pres = $ppt.Presentations.Open($pptxPath, [Microsoft.Office.Core.MsoTriState]::msoTrue, [Microsoft.Office.Core.MsoTriState]::msoFalse, [Microsoft.Office.Core.MsoTriState]::msoFalse)
    # 32 = ppSaveAsPDF
    $pres.SaveAs($pdfPath, 32)
    $pres.Close()
    $ppt.Quit()
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
    Write-Host "SUCCESS: PDF generated at $pdfPath"
} catch {
    Write-Error $_.Exception.Message
}
