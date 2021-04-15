import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations
from scipy.ndimage import binary_fill_holes
from skimage import img_as_ubyte
from skimage.color import rgb2gray, rgb2hsv
from skimage.draw import polygon, circle_perimeter
from skimage.feature import canny
from skimage.filters import median
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import disk, square
from skimage.morphology import remove_small_holes
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

plt.rc('axes', axisbelow=True, grid=True)
plt.rc('grid', c='w', ls=':')
plt.rc('font', family='serif', size=11)
plt.rc('image', cmap='gray')
plt.rc('axes.spines', bottom=False, left=False, top=False, right=False)
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True


def to1channel(img, mode='r'):
    '''
    Args:
        img, array: (N, M, 3) 8-bit RGB image
        mode, str: how to perform conversion:
          'r' - red channel only
          'g' - grayscale
          's' - saturation
          'v' - value
    Returns:
        img_output: (N, M) 8-bit image
    '''
    if mode == 'r':
        return img[:, :, 0]
    elif mode == 'g':
        return img_as_ubyte(rgb2gray(img))
    elif mode == 's':
        return img_as_ubyte(rgb2hsv(img)[:, :, 1])
    elif mode == 'v':
        return img_as_ubyte(rgb2hsv(img)[:, :, 2])


def points2dists(triangle_vertices, sqrt=False):
    trs = triangle_vertices.copy()
    trs[:, :2] -= trs[:, 2][:, None]
    trs[:, 2] = trs[:, 0] - trs[:, 1]
    sqr_dists = (trs ** 2).sum(axis=-1)
    if sqrt:
        return np.sqrt(sqr_dists)
    else:
        return sqr_dists


def farthest3(points):
    inds = np.array(list(combinations(range(len(points)), 3)))
    dists = points2dists(points[inds]).sum(axis=1)
    far_inds = inds[dists.argmax()]
    return points[far_inds]


def poly2inds(vertices):
    inds = polygon(vertices[:, 0], vertices[:, 1])
    return inds


def reverse_erosion_triangle(corners, center, erosion_radius):
    dirs = (corners - center) / np.sqrt(((corners - center) ** 2).sum(axis=1, keepdims=True))
    return np.round(corners + dirs * erosion_radius * 2).astype(int)


def iou_sets(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


def mask_objects(img, sigma=1, closing_radius=3,
                 erosion_radius=20, remove_border=True):
    img = to1channel(img)
    img = canny(img, sigma=sigma)
    img = binary_closing(img, disk(closing_radius))
    img = binary_fill_holes(img, disk(1))
    if remove_border:
        img = clear_border(img)
    img = binary_erosion(img, disk(erosion_radius))
    return img


def build_triangles(bin_mask, iou_theshold=0.8, erosion_radius=20):
    labels = label(bin_mask)
    centers, vertices = [], []
    for prop in regionprops(labels):
        if prop.area < 50:
            continue
        center = prop.centroid
        cs = prop.coords
        bbox = prop.bbox
        border_mask = (
            (cs[:, 0] == bbox[0]) | (cs[:, 0] == bbox[2] - 1) | 
            (cs[:, 1] == bbox[1]) | (cs[:, 1] == bbox[3] - 1)
        )
        corners = farthest3(cs[border_mask])
        iou = iou_sets(set((row, col) for row, col in cs), # obj
                       set(zip(*poly2inds(corners))))  # circumscribed triangle
        if (iou > iou_theshold):
            centers.append(center)
            big_corners = reverse_erosion_triangle(corners, center, erosion_radius)
            vertices.append(big_corners)
    return np.array(centers), np.array(vertices)


def triangles2mask(shape, vertices, erosion_radius=0):
    mask = np.zeros(shape, dtype=bool)
    for triangle in vertices:
        inds = poly2inds(triangle)
        mask[inds] = 1
    if erosion_radius > 0:
        mask = binary_erosion(mask, disk(erosion_radius))
    return mask


def find_dots(img, vertices, median_kernel=5, sigma=1, low=10, high=40, closing_size=1):
    img = median(img, square(median_kernel))
    mask = triangles2mask(img.shape, vertices, 2)
    img = canny(img, sigma=sigma, low_threshold=low, high_threshold=high, mask=mask)
    img = binary_closing(img, disk(closing_size))
    dots = np.logical_xor(remove_small_holes(img, area_threshold=200), img)
    centers = np.array([prop.centroid for prop in regionprops(label(dots))])
    return centers


def count_dots(vertices, centers, dots):
    counts = []
    dot_set = set((row, col) for row, col in np.round(dots).astype(int))
    for triangle in vertices:
        cur_dots = set(zip(*poly2inds(triangle))) & dot_set
        cur_dots = np.array(list(cur_dots))
        cur_cnt = ((triangle - cur_dots[:, None]) ** 2).sum(axis=-1).argmin(axis=1)
        counts.append([(cur_cnt == i).sum() for i in range(3)])
    return np.array(counts)


def plot(main_image, bin_mask, centers, vertices, gr_image):
    plt.figure('TRIMINO', figsize=(8, 6))

    plt.subplot(221)
    plt.axis('off')
    plt.imshow(main_image)

    plt.subplot(222)
    # plt.axis('off')
    plt.imshow(bin_mask)

    plt.subplot(223)
    # plt.axis('off')
    plt.imshow(main_image)
    plt.scatter(centers[:, 1], centers[:, 0], marker='*', c='lime')
    for triangle in vertices:
        triangle_plot = plt.Polygon(triangle[:, [1, 0]], fill=False,
                                    color='lime', lw=1)
        plt.gca().add_patch(triangle_plot)

    plt.subplot(224)
    plt.axis('off')

    plt.imshow(np.zeros_like(gr_image))
    for triangle in vertices:
        triangle_plot = plt.Polygon(triangle[:, [1, 0]], fill=False,
                                    color='cyan', lw=1)
        plt.gca().add_patch(triangle_plot)
    plt.scatter(dots[:, 1], dots[:, 0], marker='o', color='w', s=1)
    plt.tight_layout()
    plt.savefig('output_figure.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Segmentation and classification of trimino pieces.')
    parser.add_argument('input_path',
        help='path to the input image (RGB), .bmp and .png are supported')
    args = parser.parse_args()

    main_image = np.array(Image.open(args.input_path))
    print('Finding triminoes...')
    bin_mask = mask_objects(main_image)
    centers, vertices = build_triangles(bin_mask)
    gr_image = to1channel(main_image, 's')
    print('Counting dots...')
    dots = find_dots(gr_image, vertices)
    counts = count_dots(vertices, centers, dots)
    output_list = [f'{len(centers)}\n']
    for c, cnt in zip(np.round(centers).astype(int), counts):
        output_list.append(f'{c[1]}, {c[0]}; {cnt[0]}, {cnt[1]}, {cnt[2]}\n')
    with open('output.txt', 'w') as fout:
        fout.writelines(output_list)
    print('Result is written to output.txt!')
    print('Drawing...')
    plot(main_image, bin_mask, centers, vertices, gr_image)
