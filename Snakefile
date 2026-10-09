configfile: "config/config.yaml"
SAMPLES = config["samples"]
SAMPLE_NAMES = list(SAMPLES)

POSTPROCESS = config.get("enable_postprocess", True)
EVALUATE = config.get("enable_evaluation", False)

FINAL_MASK = "mask_final.tif" if POSTPROCESS else "mask_raw.tif"
FINAL_PROJ = "max_proj.png"
RAW_MASK = temp("results/{sample}/mask_raw.tif") if POSTPROCESS else "results/{sample}/mask_raw.tif"


def sample_entry(wildcards, key=None, default=None):
    sample_conf = SAMPLES.get(wildcards.sample)
    if sample_conf is None:
        raise KeyError(f"Sample '{wildcards.sample}' not defined in config['samples']")
    if isinstance(sample_conf, str):
        sample_conf = {"path": sample_conf}
    if not isinstance(sample_conf, dict):
        raise TypeError(f"Sample '{wildcards.sample}' must be a string or dict")
    if key is None:
        return sample_conf
    if key in sample_conf:
        return sample_conf[key]
    if default is not None:
        return default
    raise KeyError(f"Sample '{wildcards.sample}' has no '{key}' entry in config['samples']")


rule all:
    input:
        expand("results/{sample}/" + FINAL_PROJ, sample=SAMPLE_NAMES),
        expand("results/{sample}/" + FINAL_MASK, sample=SAMPLE_NAMES),
        expand("results/{sample}/cell_features.csv", sample=SAMPLE_NAMES),
        expand("results/{sample}/cell_features.h5ad", sample=SAMPLE_NAMES),
        expand("results/{sample}/evaluation.csv", sample=SAMPLE_NAMES) if EVALUATE else [],

rule read_data:
    input:
        lambda wc: sample_entry(wc, "path")
    output:
        raw_folder = directory(temp("results/{sample}/raw_imgs/"))
    params:
        dapi_channel=lambda wc: sample_entry(wc).get("dapi_channel")
    script:
        "scripts/read_data.py"


rule normalize:
    input:
        raw_folder ="results/{sample}/raw_imgs/"
    output:
        norm_folder = directory(temp("results/{sample}/norm_imgs/"))
    params:
        artifact_removal=config["normalization"].get("artifact_removal", {}),
        intensity=config["normalization"].get("intensity", {}),
        dapi_channel=lambda wc: sample_entry(wc).get("dapi_channel")
    script:
        "scripts/normalize.py"


rule max_projection:
    input:
        "results/{sample}/norm_imgs/"
    output:
        "results/{sample}/" + FINAL_PROJ
    params:
        marker_selection=config["max_projection"].get("marker_selection", []),
        dapi_channel=lambda wc: sample_entry(wc).get("dapi_channel")
    script:
        "scripts/max_projection.py"

rule segment:
    input:
        "results/{sample}/" + FINAL_PROJ
    output:
        RAW_MASK
    params:
        model=config["segmentation"].get("model", "cellpose"),
        hyperparameters=config["segmentation"].get("hyperparameters", {}),
        tiling=config["segmentation"].get("tiling", {})
    script:
        "scripts/segment.py"


rule postprocess:
    input:
        "results/{sample}/mask_raw.tif"
    output:
        "results/{sample}/" + FINAL_MASK
    params:
        enabled=POSTPROCESS,
        remove_small=config["postprocess"].get("remove_small", 0),
        fill_holes=config["postprocess"].get("fill_holes", 0),
    script:
        "scripts/postprocess.py"


rule extract_export:
    input:
       mask =  "results/{sample}/" + FINAL_MASK,
       raw_image_folder = "results/{sample}/raw_imgs/"
    output:
        features = "results/{sample}/cell_features.csv",
        h5ad = "results/{sample}/cell_features.h5ad"
    script:
        "scripts/extract_export.py"


rule evaluate:
    input:
        mask="results/{sample}/" + FINAL_MASK
    output:
        "results/{sample}/evaluation.csv"
    params:
        enabled=EVALUATE,
        metrics=config.get("evaluation", {}).get("metrics", []),
        gt=lambda wc: config.get("evaluation", {}).get("gt_masks", {}).get(wc.sample),
        iou_threshold=config.get("evaluation", {}).get("iou_threshold", 0.5),
    script:
        "scripts/evaluate.py"
