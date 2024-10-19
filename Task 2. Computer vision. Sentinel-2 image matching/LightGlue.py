from LightGlue.lightglue import viz2d
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import numpy_image_to_torch, rbd


class cfg:
    """Configuration class for setting up parameters related to the LightGlue framework,
    feature extraction, and keypoint matching."""

    lightglue = {
        "extractor": "SuperPoint",  # Feature extractor type: SuperPoint or DISK
        "device": "cpu",  # Device for running the extractor and matcher (e.g., "cpu" or "cuda")
        "max_kpts": 2048,  # Maximum number of keypoints to extract
        "homography": {
            "method": cv2.RANSAC,  # Method used for homography estimation (RANSAC algorithm)
            "ransacReprojThreshold": 3.0  # Reprojection threshold for RANSAC
        }
    }


def preprocess_lightglue(img):
    """Preprocess the input image by converting it from a NumPy array to a PyTorch tensor."""
    img = numpy_image_to_torch(img)
    return img


def match_lightglue(img0, img1, cfg):
    """
    Extract and match keypoints between two images using the LightGlue framework and a specified feature extractor.

    Args:
        img0 (np.array): The first input image.
        img1 (np.array): The second input image.
        cfg (dict): Configuration dictionary that contains the feature extractor type ('SuperPoint' or 'DISK'),
                    the maximum number of keypoints, and the device for processing (e.g., 'cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the following information:
            - "points0" (torch.Tensor): The matched keypoints in the first image.
            - "points1" (torch.Tensor): The matched keypoints in the second image.
            - "matches01" (dict): The full match information including matches and pruning.
            - "matches" (torch.Tensor): The indices of matched keypoints between img0 and img1.
            - "kpts0" (torch.Tensor): The keypoints extracted from img0.
            - "kpts1" (torch.Tensor): The keypoints extracted from img1.
            - "img0" (torch.Tensor): The preprocessed img0.
            - "img1" (torch.Tensor): The preprocessed img1.
    """
    # Preprocess both images
    img0 = preprocess_lightglue(img0)
    img1 = preprocess_lightglue(img1)

    # Load the appropriate feature extractor and matcher based on the configuration
    if cfg["extractor"] == "SuperPoint":
        extractor = SuperPoint(max_num_keypoint=cfg["max_kpts"]).eval().to(cfg["device"])
        matcher = LightGlue(features='superpoint').eval().to(cfg["device"])

    if cfg["extractor"] == "DISK":
        extractor = DISK(max_num_keypoints=cfg["max_kpts"]).eval().to(cfg["device"])
        matcher = LightGlue(features='disk').eval().to(cfg["device"])

    # Extract local features from both images
    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    # Match the extracted features
    matches01 = matcher({'image0': feats0, 'image1': feats1})

    # Remove batch dimension from the feature and match tensors
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Get keypoints and matched points
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01['matches']  # indices with shape (K, 2)
    points0 = kpts0[matches[..., 0]]  # matched coordinates in img0
    points1 = kpts1[matches[..., 1]]  # matched coordinates in img1

    # Return all results in a dictionary
    return {
        "points0": points0,
        "points1": points1,
        "matches01": matches01,
        "matches": matches,
        "kpts0": kpts0,
        "kpts1": kpts1,
        "img0": img0,
        "img1": img1
    }


def visualize_lightglue(img0, img1, points0, points1, kpts0, kpts1, matches01, plt_key=True, rotation=0, **kwargs):
    """Visualize the keypoints and matches between two images using the LightGlue framework."""
    # Plot the two images side by side
    axes = viz2d.plot_images([img0, img1], dpi=70)

    # Plot the matches between keypoints
    viz2d.plot_matches(points0, points1, color='lime', lw=0.2)

    # Add labels to the images
    viz2d.add_text(0, f'Img1', fs=20)
    viz2d.add_text(1, f'Img2: Rotation {rotation} deg.', fs=20)

    # Optionally plot the keypoints with color pruning
    if plt_key:
        kpc0, kpc1 = viz2d.cm_prune(matches01['prune0']), viz2d.cm_prune(matches01['prune1'])
        viz2d.plot_images([img0, img1], dpi=70)
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)