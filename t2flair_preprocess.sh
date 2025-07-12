#!/bin/bash

# ===============================
# Preprocessing for T2-FLAIR images
# Based on the pipeline described in the paper:
# 1. Rigid registration to GG-FLAIR-366
# 2. Brain extraction (BET)
# 3. Lesion masking
# 4. Z-score normalization
# 5. Set non-brain voxels to -1
# ===============================

# ==== Input arguments ====
INPUT_IMG="$1"         # T2-FLAIR image (e.g., T2FLAIR.nii.gz)
TEMPLATE_IMG="$2"      # Template image (e.g., GG-FLAIR-366.nii.gz)
LESION_MASK="$3"       # Binary lesion mask (e.g., from DWI segmentation)
OUTPUT_DIR="$4"        # Output directory (e.g., ./preproc_output)

# ==== Prepare output ====
mkdir -p "${OUTPUT_DIR}"

echo "Step 1: Rigid-body registration to standard template"
flirt -in "${INPUT_IMG}" \
      -ref "${TEMPLATE_IMG}" \
      -out "${OUTPUT_DIR}/aligned.nii.gz" \
      -omat "${OUTPUT_DIR}/flirt.mat" \
      -dof 6 \
      -interp trilinear

echo "Step 2: Brain extraction using BET"
bet "${OUTPUT_DIR}/aligned.nii.gz" \
    "${OUTPUT_DIR}/brain.nii.gz" \
    -f 0.3 -m

echo "Step 3: Lesion masking (remove infarct voxels)"
fslmaths "${OUTPUT_DIR}/brain.nii.gz" \
         -mas "${LESION_MASK}" \
         "${OUTPUT_DIR}/brain_masked.nii.gz"

echo "Step 4: Intensity normalization (Z-score)"
mean=$(fslstats "${OUTPUT_DIR}/brain_masked.nii.gz" -M)
std=$(fslstats "${OUTPUT_DIR}/brain_masked.nii.gz" -S)

fslmaths "${OUTPUT_DIR}/brain_masked.nii.gz" \
         -sub ${mean} -div ${std} \
         "${OUTPUT_DIR}/brain_zscore.nii.gz"

echo "Step 5: Set background voxels to -1"
fslmaths "${OUTPUT_DIR}/brain_masked.nii.gz" -bin "${OUTPUT_DIR}/brain_mask.nii.gz"
fslmaths "${OUTPUT_DIR}/brain_mask.nii.gz" -binv -mul -1 \
         -add "${OUTPUT_DIR}/brain_zscore.nii.gz" \
         "${OUTPUT_DIR}/final_preproc.nii.gz"

echo "Done. Final preprocessed image:"
echo "${OUTPUT_DIR}/final_preproc.nii.gz"
