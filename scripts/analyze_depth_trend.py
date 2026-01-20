
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.analysis import run_analysis
from core.models import AnalysisConfig, PhysicalConfig, DetectionParams
from core.image_ops import to_gray16

console = Console()

def analyze_file(filepath: Path, pixel_scale_um: float = 3.012):
    console.print(f"[bold blue]Analyzing {filepath.name}...[/bold blue]")
    
    # Load image using PIL (works with LZW-compressed TIFF without imagecodecs)
    try:
        from PIL import Image
        with Image.open(filepath) as img:
            image = np.array(img)
    except Exception as e:
        console.print(f"[red]Failed to load {filepath}: {e}[/red]")
        return None
        
    if image.ndim == 3 and (image.shape[-1] == 1 or image.shape[0] == 1):
        image = np.squeeze(image)
    elif image.ndim == 3 and image.shape[-1] > 1:
        # Convert RGB to gray if needed, but to_gray16 does this too
        pass
            
    gray_image = to_gray16(image)
    
    # Config - match default_config.json
    config = AnalysisConfig(
        physical=PhysicalConfig(pixel_scale_um_per_px=pixel_scale_um),
        detection=DetectionParams(
            min_diameter_um=9.0,
            max_diameter_um=18.0,
            background_threshold=6000.0  # Match default config
        ),
        min_valid_balls=5
    )
    
    # Use full pipeline
    result = run_analysis(gray_image, config)
    
    if not result.valid_balls:
        console.print("[yellow]No valid balls detected![/yellow]")
        return None
    
    # Extract Data
    # Depth in um. Note: m.z_px is in upsampled pixels if upsample_factor > 1.
    # The actual scale for upsampled pixels is pixel_scale_um / upsample_factor.
    actual_px_scale = pixel_scale_um / result.upsample_factor
    
    valid_data = []
    for m in result.valid_balls:
        if m.resolution_um is not None:
            depth_um = m.z_px * actual_px_scale
            valid_data.append((depth_um, m.resolution_um))
            
    if not valid_data:
         console.print("[yellow]No valid data after extraction![/yellow]")
         return None
         
    return valid_data

def main():
    data_dir = PROJECT_ROOT / "data_samples"
    files = ["10um_cropped.tiff", "50um_cropped.tiff"]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red']
    
    for fname, color in zip(files, colors):
        fpath = data_dir / fname
        if not fpath.exists():
            console.print(f"[red]File not found: {fpath}[/red]")
            continue
            
        data = analyze_file(fpath)
        if data:
            zs, res = zip(*data)
            plt.scatter(res, zs, label=fname, c=color, alpha=0.6, edgecolors='none')
            
    plt.gca().invert_yaxis() # Depth increases downwards
    plt.xlabel("Lateral Resolution ($\\mu m$)")
    plt.ylabel("Depth ($Z$) [$\\mu m$]")
    plt.title("Resolution vs Depth Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = PROJECT_ROOT / "analysis_results" / "depth_trend.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    console.print(f"[bold green]Plot saved to {output_path}[/bold green]")

if __name__ == "__main__":
    main()
