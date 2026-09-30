import argparse
import logging
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz
from skimage import measure

# Keep the official MedSAM2 inference setup unchanged.
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def getLargestCC(segmentation):
    """Keep the largest connected component from a binary 3D mask."""
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation.astype(np.uint8)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.

    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.

    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.uint8)

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert('RGB')
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array

    return resized_array


def load_ct(ct_path):
    """Load a CT volume with SimpleITK and return the image object and array."""
    ct_image = sitk.ReadImage(ct_path)
    ct_data = sitk.GetArrayFromImage(ct_image)
    return ct_image, ct_data


def load_mask(mask_path):
    """Load a tumor mask with SimpleITK and return the mask image object and array."""
    mask_image = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(mask_image)
    return mask_image, mask_data


def preprocess_ct(ct_data):
    """Apply the dataset-specific CT preprocessing used before MedSAM2 inference."""
    ct_data = np.asarray(ct_data, dtype=np.float32)
    ct_data = np.clip(ct_data, -1000, 400)
    ct_data = (ct_data + 1000.0) / 1400.0 * 255.0
    ct_data = np.clip(ct_data, 0, 255)
    ct_data = np.uint8(ct_data)
    return ct_data


def find_key_slice(mask_data):
    """Find the slice with the largest tumor area in the volume."""
    if mask_data.size == 0 or mask_data.ndim != 3:
        return None

    areas = mask_data.sum(axis=(1, 2))
    if np.all(areas == 0):
        return None

    return int(np.argmax(areas))


def compute_bbox(mask_data, key_slice):
    """Compute a 2D bounding box from the key-slice tumor mask."""
    if key_slice is None:
        return None

    key_slice_mask = mask_data[key_slice]
    ys, xs = np.where(key_slice_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    bbox = np.array([
        xs.min(),
        ys.min(),
        xs.max(),
        ys.max(),
    ], dtype=np.int32)
    return bbox


def run_inference(predictor, ct_data, key_slice, bbox):
    """
    Run the official MedSAM2 video propagation inference pipeline.
    This keeps predictor.init_state(), add_new_points_or_box(),
    propagate_in_video(), and reset_state() exactly as in the original script.
    """
    key_slice_img = ct_data[key_slice, :, :]
    video_height = key_slice_img.shape[0]
    video_width = key_slice_img.shape[1]

    img_3D_ori = ct_data
    assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'

    img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()

    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std

    segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, video_height, video_width)

        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=key_slice,
            obj_id=1,
            box=bbox,
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

        predictor.reset_state(inference_state)

        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=key_slice,
            obj_id=1,
            box=bbox,
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

        predictor.reset_state(inference_state)

    if np.max(segs_3D) > 0:
        segs_3D = getLargestCC(segs_3D)
        segs_3D = np.uint8(segs_3D)

    return segs_3D


def save_prediction(ct_image, prediction, output_path):
    """Save the predicted mask using the original CT image metadata."""
    sitk_mask = sitk.GetImageFromArray(prediction)
    sitk_mask.CopyInformation(ct_image)
    sitk.WriteImage(sitk_mask, output_path)


def main():
    parser = argparse.ArgumentParser(description='MedSAM2 inference on NSCLC-Radiomics CT volumes.')
    parser.add_argument('--ct_dir', type=str, required=True, help='Directory containing *_ct.nii.gz files.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing *_tumor_mask.nii.gz files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save prediction masks.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/MedSAM2_latest.pt', help='MedSAM2 checkpoint path.')
    parser.add_argument('--cfg', type=str, default='configs/sam2.1_hiera_t512.yaml', help='MedSAM2 model config.')
    args = parser.parse_args()

    ct_dir = args.ct_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    checkpoint = args.checkpoint
    model_cfg = args.cfg

    os.makedirs(output_dir, exist_ok=True)

    ct_files = sorted(glob(os.path.join(ct_dir, '*_ct.nii.gz')))
    logging.info(f'Found {len(ct_files)} CT volumes to process.')

    predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

    for ct_path in ct_files:
        patient_name = os.path.basename(ct_path).replace('_ct.nii.gz', '')
        logging.info(f'Processing {patient_name}...')

        mask_path = os.path.join(mask_dir, f'{patient_name}_tumor_mask.nii.gz')
        if not os.path.exists(mask_path):
            logging.info(f'Skipping {patient_name}: missing tumor mask.')
            continue

        ct_image, ct_data = load_ct(ct_path)
        _, mask_data = load_mask(mask_path)

        if mask_data.size == 0 or np.all(mask_data == 0):
            logging.info('Skipping patient with empty mask...')
            continue

        ct_data_pre = preprocess_ct(ct_data)
        key_slice = find_key_slice(mask_data)
        if key_slice is None:
            logging.info('Skipping patient with empty mask...')
            continue

        bbox = compute_bbox(mask_data, key_slice)
        if bbox is None:
            logging.info('Skipping patient with empty mask...')
            continue

        prediction = run_inference(predictor, ct_data_pre, key_slice, bbox)

        if np.max(prediction) > 0:
            output_path = os.path.join(output_dir, f'{patient_name}_prediction.nii.gz')
            logging.info('Saving prediction...')
            save_prediction(ct_image, prediction.astype(np.uint8), output_path)


if __name__ == '__main__':
    main()

