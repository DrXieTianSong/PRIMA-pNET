import numpy as np
from scipy.stats import entropy as scipy_entropy, skew, kurtosis, linregress
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import (distance_transform_edt, binary_erosion,
                           binary_dilation, generate_binary_structure,
                           maximum_filter, label as ndimage_label,
                           center_of_mass, map_coordinates)
from sklearn.decomposition import PCA

TUMOR_LABEL = 6
PANCREAS_LABELS = {1, 2, 3, 4}
HEAD_LABEL = 3
BODY_LABEL = 2
TAIL_LABEL = 4
DUCT_LABEL = 5
AORTA_LABEL = 7
PANCREAS_ORGAN_LABELS = {1, 2, 3, 4, 5, 6}

MAIN_VESSELS = {"ca": 8, "sma": 10, "vein": 11}
SIMPLE_VESSELS = {"aorta": 7, "ivc": 9}

CYSTIC_HU_THRESHOLD = 35
HETERO_MIN_VOXELS = 10
DUCT_MIN_VOXELS = 50
CONTACT_DISTANCE = 2.0
GAP_THRESHOLD_DUCT = 5.0
MIN_LENGTH_FOR_GRADIENT = 5.0
MIN_BINS_FOR_GRADIENT = 3
TUMOR_VICINITY_MM = 5.0
CONTACT_THRESHOLD = 2.0
PLANE_SIZE = 64
PLANE_SPACING = 0.5
MIN_VESSEL_AREA = 5
GAP_THRESHOLD_VESSEL = 5
PRESENCE_RADIUS = 2


def compute_heterogeneity(hu_array, prefix):
    keys = [f"{prefix}_cv", f"{prefix}_iqr", f"{prefix}_entropy",
            f"{prefix}_skewness", f"{prefix}_kurtosis", f"{prefix}_p10p90"]
    if len(hu_array) < HETERO_MIN_VOXELS:
        return {k: np.nan for k in keys}

    mean_val = float(np.mean(hu_array))
    std_val = float(np.std(hu_array))
    cv = round(std_val / abs(mean_val), 4) if abs(mean_val) > 1e-6 else np.nan

    q75, q25 = np.percentile(hu_array, [75, 25])
    iqr = round(float(q75 - q25), 2)

    hu_min = int(np.floor(hu_array.min()))
    hu_max = int(np.ceil(hu_array.max()))
    if hu_max > hu_min:
        bins = np.arange(hu_min, hu_max + 5, 5)
        hist, _ = np.histogram(hu_array, bins=bins)
        hist_prob = hist / hist.sum()
        ent = round(float(scipy_entropy(hist_prob + 1e-10, base=2)), 4)
    else:
        ent = 0.0

    sk = round(float(skew(hu_array)), 4)
    kt = round(float(kurtosis(hu_array, fisher=True)), 4)
    p90, p10 = np.percentile(hu_array, [90, 10])
    p10p90 = round(float(p90 - p10), 2)

    return {
        f"{prefix}_cv": cv,
        f"{prefix}_iqr": iqr,
        f"{prefix}_entropy": ent,
        f"{prefix}_skewness": sk,
        f"{prefix}_kurtosis": kt,
        f"{prefix}_p10p90": p10p90,
    }


def compute_3d_diameters(tumor_mask):
    if tumor_mask.sum() < 2:
        return 0.0, 0.0

    surface = tumor_mask & ~binary_erosion(tumor_mask)
    coords = np.array(np.where(surface)).T.astype(float)

    if len(coords) < 2:
        coords = np.array(np.where(tumor_mask)).T.astype(float)
    if len(coords) < 2:
        return 0.0, 0.0

    if len(coords) >= 4:
        try:
            hull = ConvexHull(coords)
            pts = coords[hull.vertices]
        except Exception:
            pts = coords
    else:
        pts = coords

    dist_mat = squareform(pdist(pts))
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    max_diam = float(dist_mat[i, j])

    if max_diam < 1e-6:
        return 0.0, 0.0

    direction = (pts[j] - pts[i]) / max_diam
    proj = pts - np.outer(pts @ direction, direction)
    perp_diam = float(pdist(proj).max()) if len(proj) >= 2 else 0.0

    return max_diam, perp_diam


def compute_axial_diameters(mask_2d):
    coords = np.array(np.where(mask_2d)).T.astype(float)
    if len(coords) < 2:
        return 0.0, 0.0

    struct_2d = np.ones((3, 3), dtype=bool)
    boundary = mask_2d & ~binary_erosion(mask_2d, structure=struct_2d)
    bcoords = np.array(np.where(boundary)).T.astype(float)
    if len(bcoords) < 2:
        bcoords = coords

    if len(bcoords) >= 3:
        try:
            hull = ConvexHull(bcoords)
            pts = bcoords[hull.vertices]
        except Exception:
            pts = bcoords
    else:
        pts = bcoords

    if len(pts) < 2:
        return 0.0, 0.0

    dist_mat = squareform(pdist(pts))
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    long_diam = float(dist_mat[i, j])

    if long_diam < 1e-6:
        return 0.0, 0.0

    direction = (pts[j] - pts[i]) / long_diam
    perp = np.array([-direction[1], direction[0]])
    proj = pts @ perp
    short_diam = float(proj.max() - proj.min())

    return long_diam, short_diam


def extract_prior_features(ct_data, mask_data):
    tumor_mask = mask_data == TUMOR_LABEL
    tumor_voxels = int(tumor_mask.sum())

    if tumor_voxels == 0:
        return {}

    tumor_hu = ct_data[tumor_mask]
    features = {}

    max_3d, perp_3d = compute_3d_diameters(tumor_mask)
    features["prior_max_3d_diameter"] = round(max_3d, 2)
    features["prior_perp_3d_diameter"] = round(perp_3d, 2)

    axial_areas = tumor_mask.sum(axis=(0, 1))
    max_z = int(np.argmax(axial_areas))
    axial_slice = tumor_mask[:, :, max_z]
    axial_long, axial_short = compute_axial_diameters(axial_slice)
    features["prior_axial_long_diameter"] = round(axial_long, 2)
    features["prior_axial_short_diameter"] = round(axial_short, 2)

    cystic_mask_1d = tumor_hu < CYSTIC_HU_THRESHOLD
    solid_count = int((~cystic_mask_1d).sum())
    cystic_count = int(cystic_mask_1d.sum())

    features["prior_solid_volume_mm3"] = float(solid_count)
    features["prior_cystic_volume_mm3"] = float(cystic_count)
    features["prior_solid_proportion"] = round(solid_count / tumor_voxels, 4)
    features["prior_cystic_proportion"] = round(cystic_count / tumor_voxels, 4)
    features["prior_solid_cystic_ratio"] = (
        round(solid_count / cystic_count, 4) if cystic_count > 0 else np.nan)

    axial_ct = ct_data[:, :, max_z]
    axial_tumor_hu = axial_ct[axial_slice]
    axial_total = int(axial_slice.sum())
    axial_cystic = int((axial_tumor_hu < CYSTIC_HU_THRESHOLD).sum())
    axial_solid = axial_total - axial_cystic

    features["prior_axial_total_area_mm2"] = float(axial_total)
    features["prior_axial_solid_area_mm2"] = float(axial_solid)
    features["prior_axial_cystic_area_mm2"] = float(axial_cystic)

    if axial_total > 0:
        features["prior_axial_solid_proportion"] = round(axial_solid / axial_total, 4)
        features["prior_axial_cystic_proportion"] = round(axial_cystic / axial_total, 4)
    else:
        features["prior_axial_solid_proportion"] = np.nan
        features["prior_axial_cystic_proportion"] = np.nan

    features["prior_axial_solid_cystic_ratio"] = (
        round(axial_solid / axial_cystic, 4) if axial_cystic > 0 else np.nan)

    solid_hu = tumor_hu[~cystic_mask_1d]

    if len(solid_hu) > 0:
        solid_mean = float(np.mean(solid_hu))
        features["prior_solid_mean_hu"] = round(solid_mean, 2)

        pancreas_hu = ct_data[np.isin(mask_data, list(PANCREAS_LABELS))]
        if len(pancreas_hu) > 0:
            pmean = float(np.mean(pancreas_hu))
            features["prior_tumor_to_pancreas_ratio"] = (
                round(solid_mean / pmean, 4) if abs(pmean) > 1e-6 else np.nan)
        else:
            features["prior_tumor_to_pancreas_ratio"] = np.nan

        aorta_hu = ct_data[mask_data == AORTA_LABEL]
        if len(aorta_hu) > 0:
            amean = float(np.mean(aorta_hu))
            features["prior_tumor_to_aorta_ratio"] = (
                round(solid_mean / amean, 4) if abs(amean) > 1e-6 else np.nan)
        else:
            features["prior_tumor_to_aorta_ratio"] = np.nan
    else:
        features["prior_solid_mean_hu"] = np.nan
        features["prior_tumor_to_pancreas_ratio"] = np.nan
        features["prior_tumor_to_aorta_ratio"] = np.nan

    cystic_hu = tumor_hu[cystic_mask_1d]
    features.update(compute_heterogeneity(solid_hu, "prior_solid"))
    features.update(compute_heterogeneity(cystic_hu, "prior_cystic"))
    features.update(compute_heterogeneity(tumor_hu, "prior_whole"))

    return features


def compute_medial_axis_points(mask, dt):
    footprint = np.zeros((3, 3, 3), dtype=bool)
    footprint[1, 1, 1] = True
    footprint[0, 1, 1] = footprint[2, 1, 1] = True
    footprint[1, 0, 1] = footprint[1, 2, 1] = True
    footprint[1, 1, 0] = footprint[1, 1, 2] = True

    local_max = maximum_filter(dt, footprint=footprint)
    medial_mask = (dt == local_max) & mask & (dt > 0)

    coords = np.array(np.where(medial_mask)).T
    dt_vals = dt[medial_mask]

    if len(coords) == 0:
        coords = np.array(np.where(mask)).T
        dt_vals = dt[mask]

    return coords, dt_vals


def compute_head_tail_axis(mask_data):
    head_mask = mask_data == HEAD_LABEL
    tail_mask = mask_data == TAIL_LABEL

    if head_mask.sum() == 0 or tail_mask.sum() == 0:
        return None, None, None

    head_coords = np.array(np.where(head_mask)).T.astype(float)
    tail_coords = np.array(np.where(tail_mask)).T.astype(float)

    head_centroid = head_coords.mean(axis=0)
    tail_centroid = tail_coords.mean(axis=0)

    axis = tail_centroid - head_centroid
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return None, None, None

    return axis / length, head_centroid, tail_centroid


def assign_duct_to_region(duct_coords, mask_data):
    region_labels = [HEAD_LABEL, BODY_LABEL, TAIL_LABEL]
    distances = np.full((len(duct_coords), 3), np.inf)

    for idx, rl in enumerate(region_labels):
        region_mask = mask_data == rl
        if region_mask.sum() == 0:
            continue
        dt_region = distance_transform_edt(~region_mask)
        for i, c in enumerate(duct_coords):
            distances[i, idx] = dt_region[int(c[0]), int(c[1]), int(c[2])]

    min_idx = np.argmin(distances, axis=1)
    regions = np.array([region_labels[mi] for mi in min_idx])
    all_inf = np.all(np.isinf(distances), axis=1)
    regions[all_inf] = 0

    return regions


def extract_duct_features(ct_data, mask_data):
    duct_mask = mask_data == DUCT_LABEL
    tumor_mask = mask_data == TUMOR_LABEL
    n_voxels = int(duct_mask.sum())

    features = {}
    features["duct_visible"] = 1 if n_voxels >= DUCT_MIN_VOXELS else 0
    features["duct_volume_mm3"] = float(n_voxels)

    if features["duct_visible"] == 0:
        nan_keys = [
            "duct_length_mm", "duct_max_diameter", "duct_mean_diameter",
            "duct_head_max_diameter", "duct_head_mean_diameter",
            "duct_body_max_diameter", "duct_body_mean_diameter",
            "duct_tail_max_diameter", "duct_tail_mean_diameter",
            "duct_head_body_ratio", "duct_body_tail_ratio", "duct_head_tail_ratio",
            "duct_tumor_min_distance", "duct_tumor_contact", "duct_tumor_contact_length",
            "duct_n_segments", "duct_truncated_by_tumor",
            "duct_upstream_max_diameter", "duct_downstream_max_diameter",
            "duct_upstream_downstream_ratio", "duct_diameter_gradient",
        ]
        for k in nan_keys:
            features[k] = np.nan
        return features

    coords = np.array(np.where(duct_mask)).T.astype(float)
    if len(coords) >= 2:
        pca = PCA(n_components=1)
        proj = pca.fit_transform(coords)
        features["duct_length_mm"] = round(float(proj.max() - proj.min()), 2)
    else:
        features["duct_length_mm"] = 0.0

    dt = distance_transform_edt(duct_mask)
    medial_coords, medial_dt = compute_medial_axis_points(duct_mask, dt)

    features["duct_max_diameter"] = round(float(dt.max()) * 2, 2)
    features["duct_mean_diameter"] = (
        round(float(medial_dt.mean()) * 2, 2) if len(medial_dt) > 0 else np.nan)

    axis, head_centroid, tail_centroid = compute_head_tail_axis(mask_data)

    duct_coords = np.array(np.where(duct_mask)).T
    if axis is not None:
        regions_all = assign_duct_to_region(duct_coords, mask_data)
        regions_medial = assign_duct_to_region(medial_coords, mask_data)
    else:
        regions_all = np.zeros(len(duct_coords), dtype=int)
        regions_medial = np.zeros(len(medial_coords), dtype=int)

    region_names = {HEAD_LABEL: "head", BODY_LABEL: "body", TAIL_LABEL: "tail"}
    region_max_diams = {}

    for rl, name in region_names.items():
        mask_medial = regions_medial == rl
        mask_all = regions_all == rl

        if mask_all.sum() > 0:
            region_dt_all = dt[duct_coords[mask_all, 0],
                               duct_coords[mask_all, 1],
                               duct_coords[mask_all, 2]]
            max_d = round(float(region_dt_all.max()) * 2, 2)
        else:
            max_d = np.nan

        if mask_medial.sum() > 0:
            mean_d = round(float(medial_dt[mask_medial].mean()) * 2, 2)
        else:
            mean_d = np.nan

        features[f"duct_{name}_max_diameter"] = max_d
        features[f"duct_{name}_mean_diameter"] = mean_d
        region_max_diams[name] = max_d

    def safe_ratio(a, b):
        if np.isnan(a) or np.isnan(b) or abs(b) < 1e-6:
            return np.nan
        return round(a / b, 4)

    features["duct_head_body_ratio"] = safe_ratio(
        region_max_diams["head"], region_max_diams["body"])
    features["duct_body_tail_ratio"] = safe_ratio(
        region_max_diams["body"], region_max_diams["tail"])
    features["duct_head_tail_ratio"] = safe_ratio(
        region_max_diams["head"], region_max_diams["tail"])

    if tumor_mask.sum() == 0:
        features["duct_tumor_min_distance"] = np.nan
        features["duct_tumor_contact"] = 0
        features["duct_tumor_contact_length"] = 0.0
    else:
        tumor_dt = distance_transform_edt(~tumor_mask)
        dist_to_tumor = tumor_dt[duct_coords[:, 0], duct_coords[:, 1], duct_coords[:, 2]]
        min_dist = float(dist_to_tumor.min())
        features["duct_tumor_min_distance"] = round(min_dist, 2)
        features["duct_tumor_contact"] = 1 if min_dist < CONTACT_DISTANCE else 0

        if features["duct_tumor_contact"] == 1 and axis is not None:
            contact_mask = dist_to_tumor < CONTACT_DISTANCE
            contact_coords = duct_coords[contact_mask].astype(float)
            if len(contact_coords) >= 2:
                proj = (contact_coords - head_centroid) @ axis
                features["duct_tumor_contact_length"] = round(float(proj.max() - proj.min()), 2)
            else:
                features["duct_tumor_contact_length"] = 0.0
        else:
            features["duct_tumor_contact_length"] = 0.0

    labeled, n_segments = ndimage_label(duct_mask)
    features["duct_n_segments"] = int(n_segments)

    if n_segments >= 2 and tumor_mask.sum() > 0:
        tumor_dt_trunc = distance_transform_edt(~tumor_mask)
        near_tumor_count = 0
        for seg_id in range(1, n_segments + 1):
            seg_coords = np.array(np.where(labeled == seg_id)).T
            seg_dist = tumor_dt_trunc[seg_coords[:, 0], seg_coords[:, 1], seg_coords[:, 2]]
            if seg_dist.min() < GAP_THRESHOLD_DUCT:
                near_tumor_count += 1
        features["duct_truncated_by_tumor"] = 1 if near_tumor_count >= 2 else 0
    else:
        features["duct_truncated_by_tumor"] = 0

    if axis is not None and tumor_mask.sum() > 0:
        tumor_coords_f = np.array(np.where(tumor_mask)).T.astype(float)
        tumor_centroid = tumor_coords_f.mean(axis=0)
        tumor_proj = float((tumor_centroid - head_centroid) @ axis)

        duct_coords_f = duct_coords.astype(float)
        duct_proj = (duct_coords_f - head_centroid) @ axis
        duct_dt_vals = dt[duct_coords[:, 0], duct_coords[:, 1], duct_coords[:, 2]]

        upstream_mask = duct_proj > tumor_proj
        downstream_mask = duct_proj < tumor_proj

        features["duct_upstream_max_diameter"] = (
            round(float(duct_dt_vals[upstream_mask].max()) * 2, 2)
            if upstream_mask.sum() > 0 else np.nan)
        features["duct_downstream_max_diameter"] = (
            round(float(duct_dt_vals[downstream_mask].max()) * 2, 2)
            if downstream_mask.sum() > 0 else np.nan)

        up_d = features["duct_upstream_max_diameter"]
        down_d = features["duct_downstream_max_diameter"]
        if not np.isnan(up_d) and not np.isnan(down_d) and abs(down_d) > 1e-6:
            features["duct_upstream_downstream_ratio"] = round(up_d / down_d, 4)
        else:
            features["duct_upstream_downstream_ratio"] = np.nan

        duct_length = float(duct_proj.max() - duct_proj.min())
        if duct_length >= MIN_LENGTH_FOR_GRADIENT:
            proj_min = duct_proj.min()
            proj_range = duct_proj.max() - proj_min
            bin_indices = np.clip(
                ((duct_proj - proj_min) / proj_range * 10).astype(int), 0, 9)
            bin_means, bin_positions = [], []
            for b in range(10):
                bmask = bin_indices == b
                if bmask.sum() > 0:
                    bin_means.append(float(duct_dt_vals[bmask].mean()) * 2)
                    bin_positions.append(b)
            if len(bin_means) >= MIN_BINS_FOR_GRADIENT:
                slope, _, _, _, _ = linregress(bin_positions, bin_means)
                features["duct_diameter_gradient"] = round(float(slope), 4)
            else:
                features["duct_diameter_gradient"] = np.nan
        else:
            features["duct_diameter_gradient"] = np.nan
    else:
        features["duct_upstream_max_diameter"] = np.nan
        features["duct_downstream_max_diameter"] = np.nan
        features["duct_upstream_downstream_ratio"] = np.nan
        features["duct_diameter_gradient"] = np.nan

    return features


def extract_pancreas_features(mask_data, duct_row):
    features = {}
    panc_mask = np.isin(mask_data, list(PANCREAS_LABELS))
    tumor_mask = mask_data == TUMOR_LABEL

    if panc_mask.sum() == 0:
        for k in ["panc_max_thickness", "panc_mean_thickness",
                   "panc_head_thickness", "panc_body_thickness",
                   "panc_tail_thickness", "panc_thickness_at_tumor"]:
            features[k] = np.nan
    else:
        dt = distance_transform_edt(panc_mask)
        features["panc_max_thickness"] = round(float(dt.max()) * 2, 2)

        medial_coords, medial_dt = compute_medial_axis_points(panc_mask, dt)
        features["panc_mean_thickness"] = round(float(medial_dt.mean()) * 2, 2)

        for lbl, name in [(HEAD_LABEL, "head"), (BODY_LABEL, "body"), (TAIL_LABEL, "tail")]:
            region_mask = mask_data == lbl
            if region_mask.sum() == 0:
                features[f"panc_{name}_thickness"] = np.nan
            else:
                dt_region = distance_transform_edt(region_mask)
                features[f"panc_{name}_thickness"] = round(float(dt_region.max()) * 2, 2)

        if tumor_mask.sum() > 0:
            tumor_dt = distance_transform_edt(~tumor_mask)
            near_tumor = panc_mask & (tumor_dt <= TUMOR_VICINITY_MM)
            if near_tumor.sum() > 0:
                features["panc_thickness_at_tumor"] = round(float(np.median(dt[near_tumor])) * 2, 2)
            else:
                features["panc_thickness_at_tumor"] = np.nan
        else:
            features["panc_thickness_at_tumor"] = np.nan

    def safe_ratio(a, b):
        if a is None or b is None:
            return np.nan
        if np.isnan(a) or np.isnan(b) or abs(b) < 1e-6:
            return np.nan
        return round(a / b, 4)

    features["panc_duct_thickness_ratio"] = safe_ratio(
        features.get("panc_max_thickness"), duct_row.get("duct_max_diameter"))
    for name, duct_col in [("head", "duct_head_max_diameter"),
                            ("body", "duct_body_max_diameter"),
                            ("tail", "duct_tail_max_diameter")]:
        features[f"panc_duct_{name}_ratio"] = safe_ratio(
            features.get(f"panc_{name}_thickness"), duct_row.get(duct_col))

    panc_vol = float(panc_mask.sum())
    tumor_vol = float(tumor_mask.sum())
    features["panc_parenchyma_volume_mm3"] = panc_vol
    features["tumor_panc_volume_ratio"] = (
        round(tumor_vol / panc_vol, 4) if panc_vol > 0 and tumor_vol > 0 else np.nan)

    if tumor_mask.sum() == 0:
        features["tumor_surface_area_mm2"] = np.nan
        features["tumor_panc_contact_area_mm2"] = np.nan
        features["tumor_panc_contact_ratio"] = np.nan
        features["tumor_protrusion_ratio"] = np.nan
    else:
        struct_6conn = generate_binary_structure(3, 1)
        eroded = binary_erosion(tumor_mask, structure=struct_6conn)
        surface_mask = tumor_mask & ~eroded
        total_surface = int(surface_mask.sum())
        features["tumor_surface_area_mm2"] = float(total_surface)

        if total_surface == 0:
            features["tumor_panc_contact_area_mm2"] = 0.0
            features["tumor_panc_contact_ratio"] = np.nan
            features["tumor_protrusion_ratio"] = np.nan
        else:
            struct_26conn = generate_binary_structure(3, 3)
            panc_dilated = binary_dilation(panc_mask, structure=struct_26conn)
            contact_count = int((surface_mask & panc_dilated).sum())

            pancreas_organ_mask = np.isin(mask_data, list(PANCREAS_ORGAN_LABELS))
            non_organ_dilated = binary_dilation(~pancreas_organ_mask, structure=struct_6conn)
            protrusion_count = int((surface_mask & non_organ_dilated).sum())

            features["tumor_panc_contact_area_mm2"] = float(contact_count)
            features["tumor_panc_contact_ratio"] = round(contact_count / total_surface, 4)
            features["tumor_protrusion_ratio"] = round(protrusion_count / total_surface, 4)

    return features


def extract_skeleton(vessel_mask):
    from skimage.morphology import skeletonize
    return skeletonize(vessel_mask).astype(np.uint8)


def keep_largest_component(mask):
    labeled, n = ndimage_label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = np.argmax(counts)
    return (labeled == largest).astype(np.uint8)


def skeleton_to_ordered_path(skel_mask):
    import networkx as nx

    coords = np.argwhere(skel_mask > 0)
    if len(coords) < 3:
        return coords.astype(float)

    idx_map = {tuple(c): i for i, c in enumerate(coords)}
    G = nx.Graph()
    for i, c in enumerate(coords):
        G.add_node(i, xyz=tuple(c))

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets.append((dz, dy, dx, np.sqrt(dz * dz + dy * dy + dx * dx)))

    for c in coords:
        i = idx_map[tuple(c)]
        for dz, dy, dx, w in offsets:
            nb = (c[0] + dz, c[1] + dy, c[2] + dx)
            j = idx_map.get(nb)
            if j is not None:
                G.add_edge(i, j, weight=w)

    if len(G) == 0:
        return coords.astype(float)

    start = next(iter(G.nodes))
    lengths = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    a = max(lengths, key=lengths.get)
    lengths = nx.single_source_dijkstra_path_length(G, a, weight="weight")
    b = max(lengths, key=lengths.get)

    try:
        path_nodes = nx.shortest_path(G, a, b, weight="weight")
    except nx.NetworkXNoPath:
        path_nodes = list(G.nodes)

    return np.array([G.nodes[n]["xyz"] for n in path_nodes], dtype=float)


def compute_local_frames(path_xyz):
    n_pts = len(path_xyz)
    if n_pts < 2:
        return [(np.array([0, 0, 1.0]), np.array([1, 0, 0.0]), np.array([0, 1, 0.0]))]

    tangents = np.diff(path_xyz, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])

    frames = []
    for t in tangents:
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-8:
            t = np.array([0, 0, 1.0])
        else:
            t = t / norm_t
        if abs(t[0]) < 0.9:
            base = np.array([1.0, 0, 0])
        else:
            base = np.array([0, 1.0, 0])
        n = np.cross(t, base)
        n = n / (np.linalg.norm(n) + 1e-10)
        b = np.cross(t, n)
        b = b / (np.linalg.norm(b) + 1e-10)
        frames.append((t, n, b))

    return frames


def reslice_plane(volume, center, normal, binormal,
                  size=PLANE_SIZE, spacing=PLANE_SPACING):
    half = size // 2
    uu, vv = np.meshgrid(
        np.arange(-half, half) * spacing,
        np.arange(-half, half) * spacing,
        indexing="xy")
    uu = uu.ravel()
    vv = vv.ravel()

    coords = (center[:, None]
              + uu[None, :] * normal[:, None]
              + vv[None, :] * binormal[:, None])

    sampled = map_coordinates(volume.astype(float), coords, order=1, mode="nearest")
    return sampled.reshape(size, size)


def compute_wrap_angle(vessel_2d, tumor_2d, min_area=MIN_VESSEL_AREA):
    vessel_bin = vessel_2d > 0.5
    tumor_bin = tumor_2d > 0.5

    if vessel_bin.sum() < min_area or tumor_bin.sum() == 0:
        return 0.0

    centroid = center_of_mass(vessel_bin)
    H, W = vessel_bin.shape
    interface_pts = []

    for x in range(1, H - 1):
        for y in range(1, W - 1):
            if not vessel_bin[x, y]:
                continue
            if vessel_bin[x - 1:x + 2, y - 1:y + 2].all():
                continue
            x0, x1 = max(0, x - 2), min(H, x + 3)
            y0, y1 = max(0, y - 2), min(W, y + 3)
            if tumor_bin[x0:x1, y0:y1].any():
                interface_pts.append((x, y))

    if len(interface_pts) < 2:
        return 0.0

    bp = np.array(interface_pts, dtype=float)
    angles = np.arctan2(bp[:, 1] - centroid[1], bp[:, 0] - centroid[0])
    angles_deg = np.degrees(angles) % 360
    sorted_angles = np.sort(np.unique(np.round(angles_deg, 0)))

    if len(sorted_angles) < 2:
        return 0.0

    gaps = np.diff(sorted_angles)
    wrap_gap = 360.0 - sorted_angles[-1] + sorted_angles[0]
    gaps = np.append(gaps, wrap_gap)
    largest_gap = float(gaps.max())

    return max(0.0, 360.0 - largest_gap)


def detect_vessel_occlusion(vessel_mask, path_xyz, radius=PRESENCE_RADIUS,
                            min_gap=GAP_THRESHOLD_VESSEL):
    if len(path_xyz) < 5:
        return False, 1.0

    shape = vessel_mask.shape
    trace = []
    for p in path_xyz:
        z, y, x = int(round(p[0])), int(round(p[1])), int(round(p[2]))
        present = False
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dz * dz + dy * dy + dx * dx > radius * radius:
                        continue
                    nz, ny, nx_ = z + dz, y + dy, x + dx
                    if (0 <= nz < shape[0] and 0 <= ny < shape[1]
                            and 0 <= nx_ < shape[2]):
                        if vessel_mask[nz, ny, nx_]:
                            present = True
                            break
                if present:
                    break
            if present:
                break
        trace.append(1 if present else 0)

    trace = np.array(trace)
    max_gap = 0
    current_gap = 0
    for v in trace:
        if v == 0:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0

    return max_gap >= min_gap, max_gap


def compute_narrowing_ratio(vessel_mask, path_xyz, tumor_dt,
                            contact_dist=5.0, ref_dist=15.0):
    vessel_dt = distance_transform_edt(vessel_mask)
    path_int = path_xyz.astype(int)
    valid = ((path_int[:, 0] >= 0) & (path_int[:, 0] < vessel_mask.shape[0]) &
             (path_int[:, 1] >= 0) & (path_int[:, 1] < vessel_mask.shape[1]) &
             (path_int[:, 2] >= 0) & (path_int[:, 2] < vessel_mask.shape[2]))
    path_int = path_int[valid]

    if len(path_int) < 5:
        return np.nan

    cl_vessel_dt = vessel_dt[path_int[:, 0], path_int[:, 1], path_int[:, 2]]
    cl_tumor_dist = tumor_dt[path_int[:, 0], path_int[:, 1], path_int[:, 2]]

    contact_mask = cl_tumor_dist < contact_dist
    ref_mask = cl_tumor_dist > ref_dist

    if contact_mask.sum() == 0:
        return np.nan

    contact_min_diam = float(cl_vessel_dt[contact_mask].min()) * 2
    if ref_mask.sum() > 0:
        ref_median_diam = float(np.median(cl_vessel_dt[ref_mask])) * 2
    else:
        ref_median_diam = float(cl_vessel_dt.max()) * 2

    if ref_median_diam < 0.5:
        return np.nan

    return round(contact_min_diam / ref_median_diam, 4)


def compute_main_vessel_features(tumor_mask, tumor_dt, vessel_mask, prefix):
    features = {}
    nan_keys = ["contact", "min_distance", "max_contact_angle",
                "encasement_grade", "narrowing_ratio", "interrupted"]

    if vessel_mask.sum() == 0 or tumor_mask.sum() == 0:
        for k in nan_keys:
            features[f"{prefix}_{k}"] = np.nan
        return features

    vessel_coords = np.argwhere(vessel_mask)
    dist_to_tumor = tumor_dt[vessel_coords[:, 0], vessel_coords[:, 1], vessel_coords[:, 2]]
    min_dist = float(dist_to_tumor.min())
    features[f"{prefix}_min_distance"] = round(min_dist, 2)
    features[f"{prefix}_contact"] = 1 if min_dist < CONTACT_THRESHOLD else 0

    if not features[f"{prefix}_contact"]:
        features[f"{prefix}_max_contact_angle"] = 0.0
        features[f"{prefix}_encasement_grade"] = 0
        features[f"{prefix}_narrowing_ratio"] = np.nan
        features[f"{prefix}_interrupted"] = 0
        return features

    vessel_clean = keep_largest_component(vessel_mask.astype(np.uint8))
    skel = extract_skeleton(vessel_clean > 0)

    if skel.sum() < 3:
        features[f"{prefix}_max_contact_angle"] = 0.0
        features[f"{prefix}_encasement_grade"] = 0
        features[f"{prefix}_narrowing_ratio"] = np.nan
        features[f"{prefix}_interrupted"] = 0
        return features

    path_xyz = skeleton_to_ordered_path(skel)
    frames = compute_local_frames(path_xyz)

    vessel_vol = vessel_mask.astype(float)
    tumor_vol = tumor_mask.astype(float)

    max_angle = 0.0
    for i, (pt, frm) in enumerate(zip(path_xyz, frames)):
        pz, py, px = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
        if (0 <= pz < tumor_dt.shape[0] and 0 <= py < tumor_dt.shape[1]
                and 0 <= px < tumor_dt.shape[2]):
            if tumor_dt[pz, py, px] > 20:
                continue
        t, n, b = frm
        v2d = reslice_plane(vessel_vol, pt, n, b)
        t2d = reslice_plane(tumor_vol, pt, n, b)
        angle = compute_wrap_angle(v2d, t2d)
        max_angle = max(max_angle, angle)

    features[f"{prefix}_max_contact_angle"] = round(max_angle, 1)

    if max_angle <= 0:
        grade = 0
    elif max_angle <= 90:
        grade = 1
    elif max_angle <= 180:
        grade = 2
    else:
        grade = 3
    features[f"{prefix}_encasement_grade"] = grade

    features[f"{prefix}_narrowing_ratio"] = compute_narrowing_ratio(
        vessel_mask, path_xyz, tumor_dt)

    interrupted, _ = detect_vessel_occlusion(vessel_mask, path_xyz)
    features[f"{prefix}_interrupted"] = 1 if interrupted else 0

    return features


def compute_simple_vessel_features(tumor_dt, vessel_mask, prefix):
    features = {}
    if vessel_mask.sum() == 0:
        features[f"{prefix}_contact"] = np.nan
        features[f"{prefix}_min_distance"] = np.nan
        return features

    vessel_coords = np.argwhere(vessel_mask)
    dist_to_tumor = tumor_dt[vessel_coords[:, 0], vessel_coords[:, 1], vessel_coords[:, 2]]
    min_dist = float(dist_to_tumor.min())
    features[f"{prefix}_min_distance"] = round(min_dist, 2)
    features[f"{prefix}_contact"] = 1 if min_dist < CONTACT_THRESHOLD else 0
    return features


def extract_vessel_features(mask_data):
    tumor_mask = mask_data == TUMOR_LABEL

    if tumor_mask.sum() == 0:
        features = {}
        for prefix in MAIN_VESSELS:
            for k in ["contact", "min_distance", "max_contact_angle",
                       "encasement_grade", "narrowing_ratio", "interrupted"]:
                features[f"{prefix}_{k}"] = np.nan
        for prefix in SIMPLE_VESSELS:
            features[f"{prefix}_contact"] = np.nan
            features[f"{prefix}_min_distance"] = np.nan
        return features

    tumor_dt = distance_transform_edt(~tumor_mask)
    features = {}

    for prefix, lbl in MAIN_VESSELS.items():
        vessel_mask = mask_data == lbl
        features.update(compute_main_vessel_features(
            tumor_mask, tumor_dt, vessel_mask, prefix))

    for prefix, lbl in SIMPLE_VESSELS.items():
        vessel_mask = mask_data == lbl
        features.update(compute_simple_vessel_features(
            tumor_dt, vessel_mask, prefix))

    return features
