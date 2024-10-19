from lightglue import match_pair
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd
import kornia as K  # Kornia for image processing in PyTorch


def numpy_to_torch_image(np_img, device=torch.device('cpu')):
    """Convert a NumPy array to a PyTorch tensor."""
    torch_img = torch.from_numpy(np_img).float() / 255.0
    torch_img = torch_img.permute(2, 0, 1).unsqueeze(0)  # Change to (C, H, W) and add batch dim
    return torch_img.to(device)


def perform_matching(img1, img2, device):
    """Extract features and match keypoints between two images."""
    # Convert images to PyTorch tensors
    image0 = numpy_to_torch_image(img1, device=device)
    image1 = numpy_to_torch_image(img2, device=device)

    # Initialize the ALIKED extractor and LightGlue matcher
    extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.01).eval().to(device)
    matcher = LightGlue(features='aliked').eval().to(device)

    # Perform matching
    with torch.inference_mode():
        feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)

    # Print the number of matched keypoints
    num_matches = len(matches01["matches"])
    print(f'Number of matched keypoints: {num_matches}')

    return feats0, feats1, matches01, image0, image1, num_matches


def visualize_matches(feats0, feats1, matches01, image0, image1):
    """Visualize the matches between two images using keypoints."""
    # Extract keypoints and matches
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Convert images from GPU to CPU and NumPy
    image0_np = image0.squeeze().permute(1, 2, 0).cpu().numpy()
    image1_np = image1.squeeze().permute(1, 2, 0).cpu().numpy()

    # Plot the images
    axes = viz2d.plot_images([image0_np, image1_np])

    # Plot the matches between keypoints
    viz2d.plot_matches(m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy(), color="lime", lw=0.2)

    # Add text to the plot
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    # Plot keypoints with color maps
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0_np, image1_np])
    viz2d.plot_keypoints([kpts0.cpu().numpy(), kpts1.cpu().numpy()], colors=[kpc0, kpc1], ps=10)
    plt.show()


def match_and_visualize(img1, img2):
    """Perform matching and visualize the results for two images."""
    # Set the device (use GPU if available)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Perform matching
    feats0, feats1, matches01, image0, image1, num_matches = perform_matching(img1, img2, device)

    # Visualize the matches
    visualize_matches(feats0, feats1, matches01, image0, image1)
