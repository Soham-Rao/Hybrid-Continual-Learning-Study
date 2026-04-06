param(
    [string]$Device = "cuda",
    [string]$MiniSeeds = "42,123,456,789,1024",
    [string]$TinySeeds = "42,123,456,789,1024",
    [switch]$RunMiniAll,
    [switch]$RunMiniTop,
    [switch]$RunTinyTop,
    [switch]$UseViT
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $root

$miniScript = Join-Path $repoRoot "Project\notebooks\local_mini_imagenet.py"
$tinyScript = Join-Path $repoRoot "Project\notebooks\local_tiny_imagenet.py"

if (-not ($RunMiniAll -or $RunMiniTop -or $RunTinyTop)) {
    Write-Host "Nothing selected. Use -RunMiniAll, -RunMiniTop, and/or -RunTinyTop."
    exit 1
}

if ($RunMiniAll) {
    $miniModel = if ($UseViT) { "vit_small" } else { "slim_resnet18" }
    conda run -n genai python $miniScript `
        --device $Device `
        --model $miniModel `
        --methods all `
        --seeds $MiniSeeds
}

if ($RunMiniTop) {
    $miniModel = if ($UseViT) { "vit_small" } else { "slim_resnet18" }
    conda run -n genai python $miniScript `
        --device $Device `
        --model $miniModel `
        --methods "fine_tune,ewc,lwf,der,xder,icarl,er_ewc,progress_compress,si_der" `
        --seeds $MiniSeeds
}

if ($RunTinyTop) {
    $tinyModel = if ($UseViT) { "vit_small" } else { "slim_resnet18" }
    conda run -n genai python $tinyScript `
        --device $Device `
        --model $tinyModel `
        --methods "fine_tune,der,xder,si_der,icarl" `
        --seeds $TinySeeds
}
