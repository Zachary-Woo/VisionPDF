"""
Tier 2 (Experimental) -- SAM encoder feature clustering + text layer.

Loads only the SAM ViT-B visual encoder from DeepSeek OCR 2 (skipping the
expensive 14B+ language model decoder entirely) and uses its spatial feature
maps to segment the document page into layout regions via agglomerative
clustering.  Text content is then pulled from the native PDF text layer
within each detected region.

This is the naive baseline for SAM-based approaches; the trained detection
head in extract_det.py is expected to outperform this.

Architecture recap:
  SAM ViT-B  -->  neck 768->256  -->  stride-2 conv 256->512  -->  stride-2 conv 512->896
  Input 1024x1024  =>  feature map 16x16x896  (each cell ~ 64x64 px)

The 16x16 feature grid is clustered with a position-aware approach:
  1. Normalise feature vectors.
  2. Concatenate normalised (row, col) position to each feature vector
     so clusters are spatially contiguous.
  3. Run Agglomerative Clustering with a connectivity constraint
     (4-connected grid) so that clusters form rectangular-ish regions.
  4. Compute the bounding box of each cluster in pixel space.
  5. Map pixel boxes to PDF coordinates, extract text, reconstruct markdown.

Usage:
    python -m benchmark.tier2_hybrid.sam.extract_clustering [--input-dir path]
                                                             [--output-dir path]
                                                             [--model-path path-or-hf-id]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pypdfium2 as pdfium
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import DEEPSEEK_OCR2_MODEL, OMNIDOCBENCH_PDFS, OUTPUT_DIR, RENDER_DPI
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import pixel_to_pdf_coords, render_page
from benchmark.tier2_hybrid.sam.encoder import load_sam_encoder, preprocess_image
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier2_sam_clustering"


# ---------------------------------------------------------------------------
# Feature-map clustering and region extraction
# ---------------------------------------------------------------------------

def cluster_features(
    features: np.ndarray,
    n_clusters: int = 12,
    spatial_weight: float = 2.0,
) -> np.ndarray:
    """
    Cluster the spatial feature grid into document regions.

    Parameters
    ----------
    features : (H, W, C) numpy array of encoder features.
    n_clusters : int
        Target number of clusters (document regions).
    spatial_weight : float
        How much to weight spatial (row, col) position relative to
        the feature dimensions.  Higher values produce more spatially
        contiguous clusters.

    Returns
    -------
    labels : (H, W) integer cluster labels.
    """
    H, W, C = features.shape

    norms = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
    feat_normed = features / norms

    rows = np.arange(H)[:, None].repeat(W, axis=1).astype(np.float32) / max(H - 1, 1)
    cols = np.arange(W)[None, :].repeat(H, axis=0).astype(np.float32) / max(W - 1, 1)
    pos = np.stack([rows, cols], axis=-1) * spatial_weight

    combined = np.concatenate([feat_normed, pos], axis=-1).reshape(H * W, -1)

    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(
        np.stack([rows.ravel(), cols.ravel()], axis=-1),
        n_neighbors=4, mode="connectivity", include_self=False,
    )

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
    )
    labels = clustering.fit_predict(combined).reshape(H, W)
    return labels


def labels_to_bboxes(
    labels: np.ndarray,
    cell_px_h: float,
    cell_px_w: float,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Convert a (H, W) label map into pixel-space bounding boxes.

    Returns a list of (cluster_id, x1, y1, x2, y2) in pixel coords.
    """
    unique_labels = np.unique(labels)
    bboxes = []
    for cid in unique_labels:
        ys, xs = np.where(labels == cid)
        y1 = float(ys.min()) * cell_px_h
        y2 = float(ys.max() + 1) * cell_px_h
        x1 = float(xs.min()) * cell_px_w
        x2 = float(xs.max() + 1) * cell_px_w
        bboxes.append((int(cid), x1, y1, x2, y2))
    return bboxes


# ---------------------------------------------------------------------------
# Text extraction and markdown reconstruction
# ---------------------------------------------------------------------------

def extract_text_for_regions(
    pdf_path: str,
    page_index: int,
    bboxes: List[Tuple[int, float, float, float, float]],
    render_scale: float,
) -> str:
    """
    For each region bounding box (in pixel space of the rendered image),
    convert to PDF coordinates and extract text from the text layer.
    Regions are sorted top-to-bottom, left-to-right.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]
    text_page = page.get_textpage()

    sorted_boxes = sorted(
        bboxes,
        key=lambda b: ((b[2] + b[4]) / 2.0, (b[1] + b[3]) / 2.0),
    )

    parts: List[str] = []
    for _cid, px1, py1, px2, py2 in sorted_boxes:
        pdf_x1, pdf_y1, pdf_x2, pdf_y2 = pixel_to_pdf_coords(
            px1, py1, px2, py2, render_scale
        )
        text = text_page.get_text_bounded(pdf_x1, pdf_y1, pdf_x2, pdf_y2).strip()
        if text:
            parts.append(text)

    text_page.close()
    page.close()
    doc.close()

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_dir: Path, output_base: Path, model_path: str):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM encoder from {model_path}...")
    encoder = load_sam_encoder(model_path, device)

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        with timed(use_cuda=True) as t:
            image, render_scale = render_page(str(pdf_path), dpi=RENDER_DPI)

            tensor, img_scale, new_w, new_h = preprocess_image(image, target_size=1024)
            tensor = tensor.to(device=device, dtype=torch.float16)

            with torch.no_grad():
                feat = encoder(tensor)

            feat_np = feat[0].float().cpu().numpy()
            feat_np = feat_np.transpose(1, 2, 0)
            H_feat, W_feat, _ = feat_np.shape

            n_clusters = min(20, max(4, H_feat * W_feat // 8))
            labels = cluster_features(feat_np, n_clusters=n_clusters)

            cell_px_h = (1024.0 / H_feat) / img_scale
            cell_px_w = (1024.0 / W_feat) / img_scale
            bboxes = labels_to_bboxes(labels, cell_px_h, cell_px_w)

            markdown = extract_text_for_regions(
                str(pdf_path), 0, bboxes, render_scale
            )

        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(timing_csv, METHOD_NAME, page_id, t.wall_seconds, t.cuda_seconds)

    write_summary(out_dir, METHOD_NAME, {"total_pages": len(pdf_files)})
    print(f"{METHOD_NAME}: processed {len(pdf_files)} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEEPSEEK_OCR2_MODEL,
        help="HuggingFace model ID or local path to DeepSeek OCR 2 checkpoint",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()
