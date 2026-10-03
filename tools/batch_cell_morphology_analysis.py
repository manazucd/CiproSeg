#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tools/batch_cell_morphology_analysis.py
#  Purpose       : Perform batch connected-component analysis of binary
#                  cell masks, extract per-cell morphological measurements,
#                  and generate tabulated results and annotated mask images.
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-10-03
#  Last modified : 2026-10-03, MK
#  Version       : 0.2.1
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
"""Per-cell morphology tables and annotated images for a folder of masks."""

import argparse
import logging
import platform
import sys
from pathlib import Path

import pandas as pd
import skimage
from matplotlib import patheffects
from matplotlib.figure import Figure
from skimage.io import imread
from skimage.measure import label, regionprops_table

# MASK_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
MASK_EXTENSIONS = (".tif", ".tiff", ".png", ".bmp")
DEFAULT_OUTPUT_SUBDIR = "cell_analysis_output"
TABLE_SUFFIX = "_cell_analysis.txt"
ANNOTATED_SUFFIX = "_annotated.png"

# Same threshold as the rest of the pipeline. The mask generation tool
# leaves faint values (1-50) around cell edges, and counting those as
# foreground inflates areas and can join neighbouring cells.
FOREGROUND_THRESHOLD = 128

# 8-connectivity, so two cells touching only at a corner count as one object.
PIXEL_CONNECTIVITY = 2

RGB_CHANNELS = 3
RGBA_CHANNELS = 4

REGION_PROPERTIES = (
    "label",
    "centroid",
    "bbox",
    "area",
    "axis_major_length",
    "axis_minor_length",
)

FIGURE_SIZE_INCHES = (12, 8)
FIGURE_DPI = 300
LABEL_FONT_SIZE = 8
LABEL_OUTLINE_WIDTH = 2

LOG_FORMAT = "%(levelname)s: %(message)s"

logger = logging.getLogger(__name__)


def load_mask(path):
    """
    Read a mask image as a 2-D array.

    RGB and RGBA exports are reduced to their brightest colour channel, so
    coloured masks still work and any alpha channel is ignored.

    Parameters:
        path: Path to the mask image

    Returns:
        2-D array of the original pixel values

    Raises:
        ValueError: if the file holds anything other than a single 2-D mask
    """
    image = imread(path)
    if image.ndim == 3 and image.shape[-1] in (RGB_CHANNELS, RGBA_CHANNELS):
        image = image[..., :RGB_CHANNELS].max(axis=-1)
    if image.ndim != 2:
        raise ValueError(
            f"{path} has shape {image.shape}, expected a single 2-D mask. "
            "Save stacks as one image per frame."
        )
    return image


def binarise(image, threshold=FOREGROUND_THRESHOLD):
    """
    Turn a 2-D mask image into a boolean foreground array.

    Masks stored as 0/1 are used as they are. Anything else, such as 0/255,
    is thresholded so that faint edge pixels count as background.

    Parameters:
        image: 2-D mask array
        threshold: Lowest pixel value that counts as foreground

    Returns:
        2-D boolean array, True for foreground
    """
    if image.max() <= 1:
        return image > 0
    return image >= threshold


def measure_components(mask):
    """
    Label the connected components of a binary mask and measure each one.

    Parameters:
        mask: 2-D boolean array, True for foreground

    Returns:
        DataFrame with one row per component. Positions use scikit-image
        order: centroid-0 is the row and centroid-1 the column, and bbox is
        (min_row, min_col, max_row, max_col) with exclusive maxima. Lengths
        are in pixels and areas are pixel counts.

    Raises:
        ValueError: if mask is not a 2-D boolean array
    """
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError(
            f"Expected a 2-D boolean mask, got {mask.dtype} with shape "
            f"{mask.shape}. Pass the image through binarise() first."
        )

    labels = label(mask, connectivity=PIXEL_CONNECTIVITY)
    measurements = regionprops_table(labels, properties=REGION_PROPERTIES)
    table = pd.DataFrame(measurements)

    # A one-pixel-wide object has a zero minor axis. Report NaN, not inf.
    minor_axis = table["axis_minor_length"].replace(0, float("nan"))
    table["aspect_ratio"] = table["axis_major_length"] / minor_axis
    return table


def write_measurements(table, mask_path, table_path):
    """
    Write one mask's measurements to disk as an aligned text table.

    Parameters:
        table: Measurements from measure_components()
        mask_path: Mask the measurements came from, noted in the header
        table_path: Text file to write
    """
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(f"Input mask: {mask_path}\n")
        handle.write(f"Foreground threshold: {FOREGROUND_THRESHOLD}\n")
        handle.write(f"Pixel connectivity: {PIXEL_CONNECTIVITY} (8-connectivity)\n")
        handle.write(f"Number of connected components: {len(table)}\n\n")

        if table.empty:
            handle.write("No connected components were detected.\n")
        else:
            handle.write(table.to_string(index=False) + "\n")


def save_annotated_mask(mask, table, title, image_path):
    """
    Save the mask as a PNG with each component's ID at its centroid.

    Parameters:
        mask: 2-D boolean array that was measured
        table: Measurements from measure_components()
        title: Figure title
        image_path: PNG file to write
    """
    # A bare Figure needs no pyplot, so the global matplotlib backend is
    # left alone and there is nothing to close afterwards.
    figure = Figure(figsize=FIGURE_SIZE_INCHES)
    axes = figure.subplots()
    axes.imshow(mask, cmap="gray")

    # IDs often overhang thin cells, and on curved filaments the centroid
    # can fall outside the cell altogether. The white outline keeps them
    # readable against the black background.
    outline = patheffects.withStroke(
        linewidth=LABEL_OUTLINE_WIDTH, foreground="white"
    )
    centroids = zip(table["label"], table["centroid-0"], table["centroid-1"])
    for label_id, row, col in centroids:
        axes.text(
            col, row, str(label_id),
            ha="center", va="center",
            fontsize=LABEL_FONT_SIZE, fontweight="bold",
            path_effects=[outline],
        )

    axes.set_title(title)
    axes.axis("off")
    figure.savefig(image_path, dpi=FIGURE_DPI, bbox_inches="tight")


def analyse_mask(mask_path, output_dir):
    """
    Measure one mask and save its table and annotated image.

    Parameters:
        mask_path: Mask image to analyse
        output_dir: Existing directory for the results
    """
    mask = binarise(load_mask(mask_path))
    table = measure_components(mask)

    table_path = output_dir / f"{mask_path.stem}{TABLE_SUFFIX}"
    image_path = output_dir / f"{mask_path.stem}{ANNOTATED_SUFFIX}"
    title = f"Connected-component IDs: {mask_path.name}"

    write_measurements(table, mask_path, table_path)
    save_annotated_mask(mask, table, title, image_path)
    logger.info("Found %d connected components", len(table))


def find_masks(input_dir):
    """Return the mask images directly inside input_dir, sorted by name."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
    )


def log_environment():
    """Log the software versions behind the measurements, for traceability."""
    logger.info(
        "Python %s on %s, scikit-image %s",
        platform.python_version(),
        platform.platform(),
        skimage.__version__,
    )


def parse_arguments():
    """
    Parse and check the command-line arguments.

    Returns:
        Namespace with input_dir and output_dir as Paths. output_dir
        defaults to a subdirectory of input_dir.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the binary mask images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the results. "
            f"Default: <input_dir>/{DEFAULT_OUTPUT_SUBDIR}"
        ),
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if not args.input_dir.is_dir():
        parser.error(f"input path is not a directory: {args.input_dir}")
    if args.output_dir is None:
        args.output_dir = args.input_dir / DEFAULT_OUTPUT_SUBDIR
    return args


def main():
    """Run the analysis over every mask in the input directory."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_arguments()
    log_environment()

    mask_paths = find_masks(args.input_dir)
    if not mask_paths:
        sys.exit(
            f"No mask images ({', '.join(MASK_EXTENSIONS)}) "
            f"found in {args.input_dir}"
        )
    logger.info("Found %d mask(s) in %s", len(mask_paths), args.input_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s", args.output_dir)

    for index, mask_path in enumerate(mask_paths, start=1):
        logger.info(
            "Analysing %s (%d/%d)", mask_path.name, index, len(mask_paths)
        )
        analyse_mask(mask_path, args.output_dir)

    logger.info("Done. Processed %d mask(s).", len(mask_paths))


if __name__ == "__main__":
    main()
