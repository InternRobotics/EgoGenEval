import os
import glob
import numpy as np
import cv2
import json
from PIL import Image
from tqdm import tqdm
import math
import pickle

try:
    from .eval_assets import load_eval_asset_scene
except ImportError:
    from eval_assets import load_eval_asset_scene

try:
    import h5py
    H5PY_OK = True
except Exception:
    h5py = None
    H5PY_OK = False

class FrameInfoItem:
    def __init__(self, frame_idx, frame_name, image, depth, pose, 
                 K_color, K_depth):
        self.frame_idx = frame_idx
        self.frame_name = frame_name
        self.image = image
        self.depth = depth
        self.extrinsics = pose         
        self.intrinsics =K_color
        self.pos = None
        self.direction = None
        self.pitch = None
        self.height = None


def blender2opencv_c2w(pose):
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    opencv_c2w = np.array(pose) @ blender2opencv
    return opencv_c2w.tolist()
def convert_intrinsics(meta_data):
    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = float(fx) / 4.0 # downsample by 4
    intrinsics[1, 1] = float(fy) / 4.0
    intrinsics[0, 2] = float(cx) / 4.0
    intrinsics[1, 2] = float(cy) / 4.0
    return intrinsics
def read_image_cv2_local(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def cv2_image_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV-loaded color images to RGB while preserving grayscale."""
    if img is None:
        return img
    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def align_camera_poses_to_ground(extrinsics):
    """
    Args:
        extrinsics: (N, 4, 4) camera-to-world matrices (OpenCV convention)

    Returns:
        extrinsics_aligned: (N, 4, 4) aligned camera-to-world matrices
        R_align: (3, 3) global rotation
    """
    extrinsics = np.asarray(extrinsics)

    # --- 1. camera centers ---
    centers = extrinsics[:, :3, 3]
    center_mean = centers.mean(axis=0)
    X = centers - center_mean

    # --- 2. PCA / SVD ---
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    ground_normal = Vt[2]  # smallest variance direction

    # enforce: ground normal points to -Y
    if ground_normal[0] > 0:
        ground_normal = -ground_normal

    # --- 3. build ground-aligned frame ---
    # y axis = ground normal
    y_axis = ground_normal / np.linalg.norm(ground_normal)

    # x axis: choose in-plane direction (max variance)
    x_axis = Vt[0]
    x_axis -= x_axis.dot(y_axis) * y_axis
    x_axis /= np.linalg.norm(x_axis)

    # z axis: right-handed
    z_axis = np.cross(x_axis, y_axis)

    R_world_new = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns

    # --- 4. global alignment transform ---
    T_align = np.eye(4)
    T_align[:3, :3] = R_world_new.T
    T_align[:3, 3] = -R_world_new.T @ center_mean

    # --- 5. apply to all poses ---
    extrinsics_aligned = np.array([T_align @ T for T in extrinsics])

    return extrinsics_aligned, R_world_new


def load_scannetpp_scene(video_id):
    private_assets = load_eval_asset_scene("scannetpp", video_id)
    if private_assets is not None:
        return private_assets
    
    scannetpp_dir = os.environ.get('SCANNETPP_ROOT', 'datasets/scannetpp_processed')

    scene = video_id
    # SCANNETPP_ROOT may point either directly at the directory that holds the
    # per-scene folders, or one level up (the manifest paths embed a
    # "scannetpp_processed/" segment). Accept both conventions.
    scene_path = os.path.join(scannetpp_dir, scene)
    if not os.path.isdir(scene_path):
        nested = os.path.join(scannetpp_dir, "scannetpp_processed", scene)
        if os.path.isdir(nested):
            scene_path = nested

    metadata_path = os.path.join(scene_path, "scene_metadata.npz")
    metadata = np.load(metadata_path)
    meta_intrinsics = metadata["intrinsics"]
    trajectories = metadata["trajectories"]
    meta_images = metadata["images"]

    scene_id = scene_path.split("/")[-1].split(".")[0]
    scene_frames = [
        {
            "file_path": os.path.join(scene_path, "images", meta_images[i].split(".")[0] + ".jpg"),
            "depth_path": os.path.join(scene_path, "depth", meta_images[i].split(".")[0] + ".png"),
            "intrinsics": meta_intrinsics[i],
            "extrinsics": trajectories[i],
        }
        # for i in range(len(meta_images))
        for i in range(len(trajectories))
    ]
    scene_frames.sort(key=lambda x: x["file_path"]) 
    meta_extrinsics = np.array([frame["extrinsics"] for frame in scene_frames])
    meta_intrinsics = np.array([frame["intrinsics"] for frame in scene_frames])
    rgb_paths = np.array([frame["file_path"] for frame in scene_frames])
    depth_paths = np.array([frame["depth_path"] for frame in scene_frames])
    num_imgs = len(meta_extrinsics)

    idxs = [i for i in range(num_imgs)]

    items = []
    
    max_depth = 0
    skipped_bad_frames = []
    skipped_bad_frame_count = 0

    for idx in idxs:
        image_filepath = rgb_paths[idx]
        depth_filepath = depth_paths[idx]

        try:
            rgb_image = np.array(Image.open(image_filepath))
            # depthmap = read_depth_cv2(depth_filepath, ceph_read=False)
            with Image.open(depth_filepath) as depth_img:
                depthmap = np.array(depth_img).astype(np.int32)
        except Exception as e:
            skipped_bad_frame_count += 1
            if len(skipped_bad_frames) < 5:
                skipped_bad_frames.append({
                    "rgb": str(image_filepath),
                    "depth": str(depth_filepath),
                    "error": str(e),
                })
            continue

        depthmap = depthmap.astype(np.float32) / 1000
        depthmap[~np.isfinite(depthmap)] = 0

        depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
        threshold = (
            np.percentile(depthmap[depthmap > 0], 98)
            if depthmap[depthmap > 0].size > 0
            else 0
        )
        depthmap[depthmap > threshold] = 0.0
        if depthmap.max() > max_depth:
            max_depth = depthmap.max()
        
        intrinsic = meta_intrinsics[idx]
        camera_pose = meta_extrinsics[idx]
        
        idx = len(items)
        item = FrameInfoItem(
            frame_idx=idx,
            frame_name = image_filepath,
            image=resize_image_half(rgb_image),
            depth=resize_image_half(depthmap).astype(np.float32),
            pose=camera_pose.astype(np.float32),
            K_color=resize_intrinsics(intrinsic[:3,:3].astype(np.float32),0.5),
            K_depth=resize_intrinsics(intrinsic[:3,:3].astype(np.float32),0.5),
        )
        items.append(item)

    if skipped_bad_frame_count > 0:
        print(
            f"[WARN][ScanNet++:{video_id}] skipped {skipped_bad_frame_count} bad RGB/depth frames. "
            f"First examples: {json.dumps(skipped_bad_frames, ensure_ascii=False)}"
        )
   
    return items, max_depth


def resize_intrinsics(K, scale):
    K_new = K.copy()
    K_new[0, 0] *= scale  # fx
    K_new[1, 1] *= scale  # fy
    K_new[0, 2] *= scale  # cx
    K_new[1, 2] *= scale  # cy
    return K_new
def resize_image_half(image):
    """
    Args:
        image: np.ndarray, shape (H, W, C) or (H, W)
    Returns:
        resized_image: np.ndarray, shape (H//2, W//2, C)
    """
    return cv2.resize(
        image,
        dsize=None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_LINEAR
    )


HYPERSIM_ROOT = os.environ.get("HYPERSIM_ROOT", "datasets/hypersim")


def hypersim_scenes_root(root=HYPERSIM_ROOT):
    candidates = [
        os.path.join(root, "evermotion_dataset", "scenes"),
        os.path.join(root, "scenes"),
        root,
    ]
    for cand in candidates:
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "ai_*")):
            return cand
    return candidates[0]


def split_hypersim_scene_camera(scene_id):
    normalized = str(scene_id).replace("\\", "/")
    if "__cam_" in normalized:
        scene_part, cam_suffix = normalized.rsplit("__cam_", 1)
        return scene_part, "cam_" + cam_suffix
    if "/" in normalized and os.path.basename(normalized).startswith("cam_"):
        scene_part, cam_name = normalized.rsplit("/", 1)
        return scene_part, cam_name
    return normalized, None


def resolve_hypersim_scene_dir(scene_id, root=HYPERSIM_ROOT):
    scene_part, cam_name = split_hypersim_scene_camera(scene_id)
    if os.path.isabs(scene_part) and os.path.isdir(scene_part):
        scene_dir = scene_part
        scene_name = os.path.basename(scene_dir.rstrip("/"))
    else:
        scene_name = scene_part.strip("/")
        scene_dir = os.path.join(hypersim_scenes_root(root), scene_name)
    return scene_dir, scene_name, cam_name


def hypersim_available_cameras(scene_dir):
    detail_dir = os.path.join(scene_dir, "_detail")
    cams = []
    for cam_dir in sorted(glob.glob(os.path.join(detail_dir, "cam_*"))):
        cam_name = os.path.basename(cam_dir)
        image_dir = os.path.join(scene_dir, "images", f"scene_{cam_name}_final_preview")
        geom_dir = os.path.join(scene_dir, "images", f"scene_{cam_name}_geometry_hdf5")
        required = [
            os.path.join(cam_dir, "camera_keyframe_positions.hdf5"),
            os.path.join(cam_dir, "camera_keyframe_orientations.hdf5"),
            image_dir,
            geom_dir,
        ]
        if all(os.path.exists(p) for p in required):
            cams.append(cam_name)
    return cams


def hypersim_intrinsics(width, height, fov_x_deg=60.0):
    fov_x = math.radians(float(fov_x_deg))
    fx = 0.5 * float(width) / max(math.tan(0.5 * fov_x), 1e-6)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fx
    K[0, 2] = 0.5 * float(width - 1)
    K[1, 2] = 0.5 * float(height - 1)
    return K


def hypersim_meters_per_asset_unit(scene_dir):
    path = os.path.join(scene_dir, "_detail", "metadata_scene.csv")
    if not os.path.exists(path):
        return 1.0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if len(parts) >= 2 and parts[0] == "meters_per_asset_unit":
                    return float(parts[1])
    except Exception:
        pass
    return 1.0


def hypersim_frame_index(path):
    stem = os.path.basename(path)
    parts = stem.split(".")
    if len(parts) < 3 or parts[0] != "frame":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def read_hypersim_hdf5(path):
    if not H5PY_OK:
        raise RuntimeError("Hypersim support requires h5py in the current Python environment.")
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            key = "dataset" if "dataset" in f else next(iter(f.keys()))
            return np.asarray(f[key][()])
    except Exception:
        return None


def clip_depth_outliers(depth):
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    valid = depth > 0
    if np.any(valid):
        depth[depth > np.percentile(depth[valid], 98)] = 0.0
    return depth


def load_hypersim_scene(scene_id, root=HYPERSIM_ROOT, fov_x_deg=60.0, pose_convention="opengl"):
    private_assets = load_eval_asset_scene("hypersim", scene_id)
    if private_assets is not None:
        return private_assets
    scene_dir, scene_name, cam_name = resolve_hypersim_scene_dir(scene_id, root=root)
    cam_names = [cam_name] if cam_name else hypersim_available_cameras(scene_dir)[:1]
    if not cam_names:
        print(f"[WARN][hypersim:{scene_id}] no usable cam_* trajectory found")
        return [], 0

    items = []
    max_depth = 0
    meters_per_asset_unit = hypersim_meters_per_asset_unit(scene_dir)

    for cam in cam_names:
        cam_dir = os.path.join(scene_dir, "_detail", cam)
        image_dir = os.path.join(scene_dir, "images", f"scene_{cam}_final_preview")
        geom_dir = os.path.join(scene_dir, "images", f"scene_{cam}_geometry_hdf5")
        positions = read_hypersim_hdf5(os.path.join(cam_dir, "camera_keyframe_positions.hdf5"))
        orientations = read_hypersim_hdf5(os.path.join(cam_dir, "camera_keyframe_orientations.hdf5"))
        if positions is None or orientations is None:
            print(f"[WARN][hypersim:{scene_name}/{cam}] missing camera_keyframe HDF5")
            continue

        positions = np.asarray(positions, dtype=np.float32) * float(meters_per_asset_unit)
        orientations = np.asarray(orientations, dtype=np.float32)

        frame_infos = []
        for image_path in glob.glob(os.path.join(image_dir, "frame.*.color.jpg")):
            idx = hypersim_frame_index(image_path)
            if idx is None:
                continue
            depth_path = os.path.join(geom_dir, f"frame.{idx:04d}.depth_meters.hdf5")
            frame_infos.append((idx, image_path, depth_path))
        frame_infos.sort(key=lambda x: x[0])

        for idx, image_path, depth_path in frame_infos:
            if idx >= len(positions) or idx >= len(orientations) or not os.path.exists(depth_path):
                continue
            rgb_image = read_image_cv2_local(image_path)
            depthmap = read_hypersim_hdf5(depth_path)
            if rgb_image is None or depthmap is None:
                continue

            depthmap = clip_depth_outliers(np.asarray(depthmap, dtype=np.float32))
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = orientations[idx]
            T[:3, 3] = positions[idx]
            if pose_convention == "opengl":
                T = np.asarray(blender2opencv_c2w(T), dtype=np.float32)

            K = hypersim_intrinsics(rgb_image.shape[1], rgb_image.shape[0], fov_x_deg)
            rgb_image = resize_image_half(rgb_image)
            depthmap = cv2.resize(
                depthmap,
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            K = resize_intrinsics(K, 0.5)

            if depthmap.max() > max_depth:
                max_depth = depthmap.max()

            items.append(FrameInfoItem(
                frame_idx=len(items),
                frame_name=image_path,
                image=rgb_image,
                depth=depthmap,
                pose=T.astype(np.float32),
                K_color=K.astype(np.float32),
                K_depth=K.astype(np.float32),
            ))

    return items, max_depth

def load_matterport3d_scene(video_id):
    private_assets = load_eval_asset_scene("matterport3d", video_id)
    if private_assets is not None:
        return private_assets
    
    max_depth = 0
    image_root = os.environ.get("MATTERPORT3D_ROOT", "datasets/matterport3d/scans")
    NEW_PKL_DIR = os.environ.get("MATTERPORT3D_METADATA", "datasets/matterport3d_metadata")
    pkl_data = pickle.load(open(f'{NEW_PKL_DIR}/{video_id}.pkl','rb'))

    image_paths = pkl_data['image_paths']
    depth_paths = pkl_data['depth_image_paths']
    extrinsics_c2w = pkl_data['extrinsics_c2w']

    depth_intrinsics = pkl_data['depth_intrinsics']

    extrinsics = pkl_data['extrinsics_c2w']
    intrinsics = pkl_data['intrinsics']
    items = []
    
    
    for idx,(image_path, depth_path, camera_pose, intrinsic) in enumerate(zip(image_paths,depth_paths,extrinsics,intrinsics)):
        rgb_image = cv2.imread(image_path.replace('matterport3d',image_root),cv2.IMREAD_UNCHANGED)
        depth_raw = cv2.imread(depth_path.replace('matterport3d',image_root),cv2.IMREAD_UNCHANGED)
        if rgb_image is None or depth_raw is None:
            continue
        rgb_image = cv2_image_to_rgb(rgb_image)
        depthmap = depth_raw.astype(np.float32) / 4000.0
        
        if depthmap.max()>max_depth:
            max_depth = depthmap.max()
      
        item = FrameInfoItem(
            frame_idx=idx,
            frame_name = image_path.replace('matterport3d',image_root),
            image=resize_image_half(rgb_image),
            depth=resize_image_half(depthmap).astype(np.float32),
            pose=camera_pose.astype(np.float32),
            K_color=resize_intrinsics(intrinsic[:3,:3].astype(np.float32),0.5),
            K_depth=resize_intrinsics(intrinsic[:3,:3].astype(np.float32),0.5),
        )
        items.append(item)
        
    return items, max_depth


def load_scannet_scene(scan_id):
    private_assets = load_eval_asset_scene("scannet", scan_id)
    if private_assets is not None:
        return private_assets
    infos_dir = os.environ.get("SCANNET_METADATA", "datasets/scannet_metadata")
    image_root = os.environ.get("SCANNET_ROOT", "datasets/scannet")
    def read_annotation_pickle(path, show_progress=True):
        """
        Returns: A dictionary. Format. scene_id : (bboxes, object_ids, object_types, visible_view_object_dict, extrinsics_c2w, axis_align_matrix, intrinsics, image_paths)
        bboxes: numpy array of bounding boxes, shape (N, 9): xyz, lwh, ypr
        object_ids: numpy array of obj ids, shape (N,)
        object_types: list of strings, each string is a type of object
        visible_view_object_dict: a dictionary {view_id: visible_instance_ids}
        extrinsics_c2w: a list of 4x4 matrices, each matrix is the extrinsic matrix of a view
        axis_align_matrix: a 4x4 matrix, the axis-aligned matrix of the scene
        intrinsics: a list of 4x4 matrices, each matrix is the intrinsic matrix of a view
        image_paths: a list of strings, each string is the path of an image in the scene
        """
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True)
        metainfo = data["metainfo"]
        object_type_to_int = metainfo["categories"]
        object_int_to_type = {v: k for k, v in object_type_to_int.items()}
        datalist = data["data_list"]
        output_data = {}
        pbar = tqdm(range(len(datalist))) if show_progress else range(len(datalist))
        for scene_idx in pbar:
            images = datalist[scene_idx]["images"]
            intrinsic = datalist[scene_idx].get("cam2img", None)  # a 4x4 matrix
            missing_intrinsic = False
            if intrinsic is None:
                missing_intrinsic = True  # each view has different intrinsic for mp3d
            depth_intrinsic = datalist[scene_idx].get(
                "cam2depth", None
            )  # a 4x4 matrix, for 3rscan
            if depth_intrinsic is None and not missing_intrinsic:
                depth_intrinsic = datalist[scene_idx][
                    "depth2img"
                ]  # a 4x4 matrix, for scannet
            axis_align_matrix = datalist[scene_idx]["axis_align_matrix"]  # a 4x4 matrix
            scene_id = images[0]["img_path"].split("/")[-2]  # str

            instances = datalist[scene_idx]["instances"]
            bboxes = []
            object_ids = []
            object_types = []
            object_type_ints = []
            for object_idx in range(len(instances)):
                bbox_3d = instances[object_idx]["bbox_3d"]  # list of 9 values
                bbox_label_3d = instances[object_idx]["bbox_label_3d"]  # int
                bbox_id = instances[object_idx]["bbox_id"]  # int
                object_type = object_int_to_type[bbox_label_3d]
                # if object_type in EXCLUDED_OBJECTS:
                #     continue
                object_type_ints.append(bbox_label_3d)
                object_types.append(object_type)
                bboxes.append(bbox_3d)
                object_ids.append(bbox_id)
            bboxes = np.array(bboxes)
            object_ids = np.array(object_ids)
            object_type_ints = np.array(object_type_ints)

            visible_view_object_dict = {}
            extrinsics_c2w = []
            intrinsics = []
            depth_intrinsics = []
            image_paths = []
            visible_list = []
            for image_idx in range(len(images)):
                img_path = images[image_idx]["img_path"]  # str
                extrinsic_id = img_path.split("/")[-1].split(".")[0]  # str
                cam2global = images[image_idx]["cam2global"]  # a 4x4 matrix
                if missing_intrinsic:
                    intrinsic = images[image_idx]["cam2img"]
                    depth_intrinsic = images[image_idx]["cam2depth"]
                visible_instance_indices = images[image_idx][
                    "visible_instance_ids"
                ]  # numpy array of int
                visible_list.append(visible_instance_indices)
                visible_instance_ids = object_ids[visible_instance_indices]
                visible_view_object_dict[extrinsic_id] = visible_instance_ids
                extrinsics_c2w.append(cam2global)
                intrinsics.append(intrinsic)
                depth_intrinsics.append(depth_intrinsic)
                image_paths.append(img_path)
            if show_progress:
                pbar.set_description(f"Processing scene {scene_id}")
            output_data[scene_id] = {
                "bboxes": bboxes,
                "object_ids": object_ids,
                "object_types": object_types,
                "object_type_ints": object_type_ints,
                "visible_list": visible_list,
                "extrinsics_c2w": extrinsics_c2w,
                "axis_align_matrix": axis_align_matrix,
                "intrinsics": intrinsics,
                "depth_intrinsics": depth_intrinsics,
                "image_paths": image_paths,
            }
        return output_data


    def get_scene_info(scene_id):
        scene_info = {}
        anno = read_annotation_pickle(os.path.join(infos_dir, f"{scene_id}.pkl"), show_progress=False)[scene_id]
        scene_info[scene_id] = {}
        scene_info[scene_id]["bboxes"] = anno["bboxes"]
        scene_info[scene_id]["object_ids"] = anno["object_ids"]
        scene_info[scene_id]["object_types"] = anno["object_types"]
        scene_info[scene_id]["visible_list"] = anno["visible_list"]
        scene_info[scene_id]["image_paths"] = anno["image_paths"]
        scene_info[scene_id]["depth_intrinsics"] = anno["depth_intrinsics"]
        scene_info[scene_id]["intrinsics"] = anno["intrinsics"]

        scene_info[scene_id]["view_ids"] = [path.split("/")[-1].split(".")[0] for path in anno["image_paths"]]
        scene_info[scene_id]["extrinsics_c2w"] = anno["extrinsics_c2w"]
        
        scene_info[scene_id]["axis_align_matrix"] = anno["axis_align_matrix"]
        scene_info[scene_id]["camera_extrinsics_c2w"] = [(anno["axis_align_matrix"] @ extrinsic) for extrinsic in
                                                            anno["extrinsics_c2w"]]
        return scene_info[scene_id]
    
    max_depth = 0
    pkl_data = get_scene_info(scan_id)
    image_paths = pkl_data['image_paths']
    depth_paths = [image_path.replace('jpg','png') for image_path in image_paths]
    extrinsics_c2w = pkl_data['camera_extrinsics_c2w']
    intrinsics = [resize_intrinsics(intrinsic,0.5) for intrinsic in pkl_data['intrinsics']]
    depth_intrinsics = pkl_data['depth_intrinsics']

    if image_paths:
        print(image_paths[0].replace('scannet',image_root))
    items = []
    for idx,(image_path, depth_path, camera_pose, intrinsic) in enumerate(zip(image_paths,depth_paths,extrinsics_c2w,intrinsics)):
        rgb_image_raw = cv2.imread(image_path.replace('scannet',image_root),cv2.IMREAD_UNCHANGED)
        depth_raw = cv2.imread(depth_path.replace('scannet',image_root),cv2.IMREAD_UNCHANGED)
        if rgb_image_raw is None or depth_raw is None:
            continue

        rgb_image_raw = cv2_image_to_rgb(rgb_image_raw)
        rgb_image = resize_image_half(rgb_image_raw)
        depthmap = depth_raw.astype(np.float32) / 1000.0
        depthmap[~np.isfinite(depthmap)] = 0

        # Resize depth map to match image dimensions
        if depthmap.shape[0] != rgb_image.shape[0] or depthmap.shape[1] != rgb_image.shape[1]:
            depthmap = cv2.resize(
                depthmap, 
                (rgb_image.shape[1], rgb_image.shape[0]),  # (width, height) for OpenCV
                interpolation=cv2.INTER_NEAREST  # Better for depth maps to preserve sharpness
            )

        depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
        threshold = (
            np.percentile(depthmap[depthmap > 0], 98)
            if depthmap[depthmap > 0].size > 0
            else 0
        )
        depthmap[depthmap > threshold] = 0.0
        
        if depthmap.max()>max_depth:
            max_depth = depthmap.max()
      
        item = FrameInfoItem(
            frame_idx=idx,
            frame_name = image_path.replace('scannet',image_root),
            image=resize_image_half(rgb_image),
            depth=resize_image_half(depthmap).astype(np.float32),
            pose=camera_pose.astype(np.float32),
            K_color=resize_intrinsics(intrinsic[:3,:3].astype(np.float32),0.5),
            K_depth=resize_intrinsics(intrinsic[:3,:3].astype(np.float32),0.5),
        )
        items.append(item)
        
    return items, max_depth


if __name__=='__main__':
    NEW_PKL_DIR = os.environ.get("MATTERPORT3D_METADATA", "datasets/matterport3d_metadata")
    grounps = {}
    for video_pkl in tqdm(os.listdir(NEW_PKL_DIR)):
        pkl_data = pickle.load(open(f'{NEW_PKL_DIR}/{video_pkl}','rb'))
        image_paths = pkl_data['image_paths']
        grounps[video_pkl[:-4]]=[image_path.replace('matterport3d','matterport3d/scans') for image_path in image_paths]
    with open('mp3d_camera_grounps.json','w') as f:
        json.dump(grounps,f,indent=4)
