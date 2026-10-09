from pathlib import Path
from scripts.utils import list_images, read_image, write_image, link_or_copy


def main() -> None:
    out_dir = Path(snakemake.output[0])
    out_dir.mkdir(parents=True, exist_ok=False)

    inputs = list(snakemake.input)
    if not inputs:
        raise ValueError("No input provided to read_data")

    dapi_param = snakemake.params.get("dapi_channel", None)
    dapi_path = Path(str(dapi_param)) if dapi_param is not None else None
    
    # Ugly

    if dapi_path is not None and not dapi_path.is_file():
        dapi_path = None

    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        files = list_images(Path(inputs[0]))
    else:
        files = [Path(p) for p in inputs]

    if dapi_path is not None:
        files = [f for f in files if f.resolve() != dapi_path.resolve()]
        files = [dapi_path] + files # If dapi is in some other other folder for some reason
        if not files:
            raise ValueError("No image channels available to write")
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Input path not found: {f}")
        dst = out_dir / f.name
        link_or_copy(f, dst)

if __name__ == "__main__":
    main()
