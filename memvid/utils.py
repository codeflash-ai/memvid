"""
Shared utility functions for Memvid
"""

import io
import json
import qrcode
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import logging
from tqdm import tqdm
import base64
import gzip

from .config import get_default_config, codec_parameters

_QR_CAPACITIES = {
    'L': [25, 47, 77, 114, 154, 195, 224, 279, 335, 395,
          468, 535, 619, 667, 758, 854, 938, 1046, 1153, 1249,
          1352, 1460, 1588, 1704, 1853, 1990, 2132, 2223, 2369, 2520,
          2677, 2840, 3009, 3183, 3351, 3537, 3729, 3927, 4087, 4296],
    'M': [20, 38, 61, 90, 122, 154, 178, 221, 262, 311,
          366, 419, 483, 528, 600, 656, 734, 816, 909, 970,
          1035, 1134, 1248, 1326, 1451, 1542, 1637, 1732, 1839, 1994,
          2113, 2238, 2369, 2506, 2632, 2780, 2894, 3054, 3220, 3391],
    'Q': [16, 29, 47, 67, 87, 108, 125, 157, 189, 221,
          259, 296, 352, 376, 426, 470, 531, 574, 644, 702,
          742, 823, 890, 963, 1041, 1094, 1172, 1263, 1322, 1429,
          1499, 1618, 1700, 1787, 1867, 1966, 2071, 2181, 2298, 2420],
    'H': [10, 20, 35, 50, 64, 84, 93, 122, 143, 174,
          200, 227, 259, 283, 321, 365, 408, 452, 493, 557,
          587, 640, 672, 744, 779, 864, 910, 958, 1016, 1080,
          1150, 1226, 1307, 1394, 1431, 1530, 1591, 1658, 1774, 1852]
}

logger = logging.getLogger(__name__)


def encode_to_qr(data: str) -> Image.Image:
    """
    Encode data to QR code image

    Args:
        data: String data to encode
        config: Optional QR configuration

    Returns:
        PIL Image of QR code
    """

    config = get_default_config()["qr"]

    # DEBUG: Log input data characteristics
    original_data_len = len(data)
    logger.debug(f"[encode_to_qr] Input data length: {original_data_len} bytes")
    logger.debug(f"[encode_to_qr] Input data preview: {data[:100]}..." if len(data) > 100 else f"[encode_to_qr] Input data: {data}")

    # Compress data if it's large
    is_compressed = False
    if len(data) > 100:
        try:
            original_bytes = data.encode('utf-8')
            compressed_bytes = gzip.compress(original_bytes)
            encoded = base64.b64encode(compressed_bytes).decode('ascii')
            data = "GZ:" + encoded  # Prefix to indicate compression
            is_compressed = True

            # DEBUG: Log compression results
            compression_ratio = len(compressed_bytes) / len(original_bytes) * 100
            logger.debug(f"[encode_to_qr] Compression enabled")
            logger.debug(f"[encode_to_qr] Original size: {len(original_bytes)} bytes")
            logger.debug(f"[encode_to_qr] Compressed size: {len(compressed_bytes)} bytes ({compression_ratio:.1f}%)")
            logger.debug(f"[encode_to_qr] Base64 encoded size: {len(encoded)} bytes")
            logger.debug(f"[encode_to_qr] Final data size (with GZ: prefix): {len(data)} bytes")
        except Exception as e:
            logger.error(f"[encode_to_qr] Compression failed: {e}")
            # Fall back to uncompressed data
            is_compressed = False

    # Dynamically adjust QR code version for large data
    # Use automatic version (None) for small data, or calculate based on data size
    # The config version is a maximum, not a fixed value
    estimated_version = (len(data) // 80) + 1
    qr_version = min(max(1, estimated_version), config["version"], 40)

    # For small data, use automatic version detection (more reliable)
    if len(data) < 500:
        qr_version = None  # Let qrcode library choose optimal version
        logger.debug(f"[encode_to_qr] Using automatic version for small data ({len(data)} bytes)")
    else:
        logger.debug(f"[encode_to_qr] Using QR version: {qr_version} (estimated for {len(data)} bytes)")

    # DEBUG: Log QR code parameters
    if qr_version:
        logger.debug(f"[encode_to_qr] QR version: {qr_version} (max capacity ~{_get_qr_capacity(qr_version, config['error_correction'])} bytes)")
    logger.debug(f"[encode_to_qr] Error correction: {config['error_correction']}")
    logger.debug(f"[encode_to_qr] Box size: {config['box_size']}, Border: {config['border']}")

    try:
        qr = qrcode.QRCode(
            version=qr_version,
            error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{config['error_correction']}"),
            box_size=config["box_size"],
            border=config["border"],
        )

        qr.add_data(data)
        qr.make(fit=True)

        # DEBUG: Log actual QR version used after fit
        actual_version = qr.version
        if actual_version != qr_version:
            logger.warning(f"[encode_to_qr] QR version adjusted from {qr_version} to {actual_version} after fit")

        img = qr.make_image(fill_color=config["fill_color"], back_color=config["back_color"])

        # Convert to PIL.Image if it's a custom image class
        if not isinstance(img, Image.Image):
            img = img.convert('RGB')

        # DEBUG: Log image characteristics
        logger.debug(f"[encode_to_qr] Generated QR image size: {img.size}")
        logger.debug(f"[encode_to_qr] Image mode: {img.mode}")
        logger.debug(f"[encode_to_qr] Encoding successful (compressed={is_compressed})")

        return img

    except Exception as e:
        logger.error(f"[encode_to_qr] QR encoding failed: {e}")
        logger.error(f"[encode_to_qr] Data length: {len(data)}, QR version: {qr_version}")
        raise RuntimeError(f"Failed to encode data to QR code: {e}") from e


def _get_qr_capacity(version: int, error_correction: str) -> int:
    """
    Estimate QR code capacity based on version and error correction level.
    These are approximate values for alphanumeric mode.
    """
    if version < 1 or version > 40:
        return 0

    return _QR_CAPACITIES.get(error_correction, _QR_CAPACITIES['M'])[version - 1]


def decode_qr(image: np.ndarray, timeout: int = 5) -> Optional[str]:
    """
    Decode QR code from image

    Args:
        image: OpenCV image array
        timeout: Maximum processing time in seconds

    Returns:
        Decoded string or None if decode fails
    """
    import signal

    def handler(signum, frame):
        raise TimeoutError("QR decode timed out")

    # DEBUG: Log input image characteristics
    logger.debug(f"[decode_qr] Input image shape: {image.shape}")
    logger.debug(f"[decode_qr] Image dtype: {image.dtype}")
    logger.debug(f"[decode_qr] Image value range: [{image.min()}, {image.max()}]")

    try:
        # Set signal handler
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

        # Ensure image is grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug(f"[decode_qr] Converted to grayscale from {image.shape[2]} channels")
        else:
            gray = image
            logger.debug(f"[decode_qr] Image already grayscale")

        # Initialize OpenCV QR code detector
        detector = cv2.QRCodeDetector()

        # Detect and decode with multiple methods
        methods = [
            ("Direct decode", lambda x: detector.detectAndDecode(x)),
            ("Histogram equalization", lambda x: detector.detectAndDecode(cv2.equalizeHist(x))),
            ("Adaptive threshold", lambda x: detector.detectAndDecode(
                cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            )),
        ]

        for method_name, method in methods:
            try:
                logger.debug(f"[decode_qr] Trying method: {method_name}")
                data, bbox, straight_qrcode = method(gray)

                if data:
                    logger.debug(f"[decode_qr] Method '{method_name}' succeeded")
                    logger.debug(f"[decode_qr] Raw decoded data length: {len(data)} bytes")
                    logger.debug(f"[decode_qr] Data preview: {data[:100]}..." if len(data) > 100 else f"[decode_qr] Data: {data}")

                    # Check if data was compressed
                    if data.startswith("GZ:"):
                        try:
                            logger.debug(f"[decode_qr] Detected compressed data (GZ: prefix)")
                            compressed_data_b64 = data[3:]
                            logger.debug(f"[decode_qr] Base64 data length: {len(compressed_data_b64)} bytes")

                            compressed_data = base64.b64decode(compressed_data_b64)
                            logger.debug(f"[decode_qr] Compressed data length: {len(compressed_data)} bytes")

                            decompressed = gzip.decompress(compressed_data)
                            logger.debug(f"[decode_qr] Decompressed data length: {len(decompressed)} bytes")

                            data = decompressed.decode('utf-8')
                            logger.debug(f"[decode_qr] Final decoded data length: {len(data)} bytes")
                            logger.debug(f"[decode_qr] Decompression successful")

                        except base64.binascii.Error as e:
                            logger.error(f"[decode_qr] Base64 decode failed: {e}")
                            logger.error(f"[decode_qr] Bad base64 data: {compressed_data_b64[:100]}...")
                            return None
                        except gzip.BadGzipFile as e:
                            logger.error(f"[decode_qr] Gzip decompress failed: {e}")
                            logger.error(f"[decode_qr] Compressed data (hex): {compressed_data[:50].hex()}")
                            return None
                        except UnicodeDecodeError as e:
                            logger.error(f"[decode_qr] UTF-8 decode failed: {e}")
                            return None
                        except Exception as e:
                            logger.error(f"[decode_qr] Decompression error: {e}")
                            return None

                    logger.debug(f"[decode_qr] Decode successful, returning data")
                    return data
                else:
                    logger.debug(f"[decode_qr] Method '{method_name}' returned no data")

            except Exception as e:
                logger.warning(f"[decode_qr] Method '{method_name}' failed: {e}")
                continue

        logger.warning(f"[decode_qr] All decode methods failed")

    except TimeoutError:
        logger.warning(f"[decode_qr] QR decode timed out after {timeout}s")
    except Exception as e:
        logger.error(f"[decode_qr] QR decode error: {e}")
        import traceback
        logger.error(f"[decode_qr] Traceback: {traceback.format_exc()}")
    finally:
        # Cancel the alarm
        signal.alarm(0)

    return None


def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert QR PIL image to video frame

    Args:
        qr_image: PIL Image of QR code
        frame_size: Target frame size (width, height)

    Returns:
        OpenCV frame array
    """
    # DEBUG: Log input image characteristics
    logger.debug(f"[qr_to_frame] Input QR image size: {qr_image.size}")
    logger.debug(f"[qr_to_frame] Input QR image mode: {qr_image.mode}")
    logger.debug(f"[qr_to_frame] Target frame size: {frame_size}")

    original_size = qr_image.size

    # Resize to fit frame while maintaining aspect ratio
    # Use NEAREST for QR codes to preserve sharp edges (better than LANCZOS for binary images)
    qr_image = qr_image.resize(frame_size, Image.Resampling.NEAREST)
    logger.debug(f"[qr_to_frame] Resized from {original_size} to {qr_image.size}")

    # Convert to RGB mode if necessary (handles L, P, etc. modes)
    if qr_image.mode != 'RGB':
        original_mode = qr_image.mode
        qr_image = qr_image.convert('RGB')
        logger.debug(f"[qr_to_frame] Converted from {original_mode} to RGB")

    # Convert to numpy array and ensure proper dtype
    img_array = np.array(qr_image, dtype=np.uint8)
    logger.debug(f"[qr_to_frame] Numpy array shape: {img_array.shape}, dtype: {img_array.dtype}")
    logger.debug(f"[qr_to_frame] Pixel value range: [{img_array.min()}, {img_array.max()}]")

    # Verify QR code contrast
    unique_values = np.unique(img_array)
    logger.debug(f"[qr_to_frame] Unique pixel values count: {len(unique_values)}")
    if len(unique_values) < 2:
        logger.warning(f"[qr_to_frame] WARNING: Image has insufficient contrast! Only {len(unique_values)} unique values")

    # Convert to OpenCV format
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    logger.debug(f"[qr_to_frame] Final frame shape: {frame.shape}")

    return frame


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract single frame from video
    
    Args:
        video_path: Path to video file
        frame_number: Frame index to extract
        
    Returns:
        OpenCV frame array or None
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            return frame
    finally:
        cap.release()
    return None


@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    """
    Extract and decode frame with caching
    """
    frame = extract_frame(video_path, frame_number)
    if frame is not None:
        return decode_qr(frame)
    return None


def batch_extract_frames(video_path: str, frame_numbers: List[int], 
                        max_workers: int = 4) -> List[Tuple[int, Optional[np.ndarray]]]:
    """
    Extract multiple frames in parallel
    
    Args:
        video_path: Path to video file
        frame_numbers: List of frame indices
        max_workers: Number of parallel workers
        
    Returns:
        List of (frame_number, frame) tuples
    """
    results = []
    
    # Sort frame numbers for sequential access
    sorted_frames = sorted(frame_numbers)
    
    cap = cv2.VideoCapture(video_path)
    try:
        for frame_num in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            results.append((frame_num, frame if ret else None))
    finally:
        cap.release()
    
    return results


def parallel_decode_qr(frames: List[Tuple[int, np.ndarray]], 
                      max_workers: int = 4) -> List[Tuple[int, Optional[str]]]:
    """
    Decode multiple QR frames in parallel
    
    Args:
        frames: List of (frame_number, frame) tuples
        max_workers: Number of parallel workers
        
    Returns:
        List of (frame_number, decoded_data) tuples
    """
    def decode_frame(item):
        frame_num, frame = item
        if frame is not None:
            data = decode_qr(frame)
            return (frame_num, data)
        return (frame_num, None)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(decode_frame, frames))
    
    return results


def batch_extract_and_decode(video_path: str, frame_numbers: List[int], 
                            max_workers: int = 4, show_progress: bool = False) -> Dict[int, str]:
    """
    Extract and decode multiple frames efficiently
    
    Args:
        video_path: Path to video file
        frame_numbers: List of frame indices
        max_workers: Number of parallel workers
        show_progress: Show progress bar
        
    Returns:
        Dict mapping frame_number to decoded data
    """
    # Extract frames
    frames = batch_extract_frames(video_path, frame_numbers)
    
    # Decode in parallel
    if show_progress:
        frames = tqdm(frames, desc="Decoding QR frames")
    
    decoded = parallel_decode_qr(frames, max_workers)
    
    # Build result dict
    result = {}
    for frame_num, data in decoded:
        if data is not None:
            result[frame_num] = data
    
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary or word boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_space = chunk.rfind(' ')

            # Prefer period break if it occurs in last 20% of chunk
            if last_period > chunk_size * 0.8:
                end = start + last_period + 1
            # Otherwise, break at last space
            elif last_space > chunk_size * 0.5:
                end = start + last_space

            chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    # Ensure last chunk is not too small
    if chunks and len(chunks[-1]) < chunk_size * 0.5:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


def save_index(index_data: Dict[str, Any], output_path: str):
    """Save index data to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def load_index(index_path: str) -> Dict[str, Any]:
    """Load index data from JSON file"""
    with open(index_path, 'r') as f:
        return json.load(f)