import os
from opendm.net import download
from opendm import log
import zipfile
import time
import sys
import tempfile
from contextlib import contextmanager
import rawpy
import cv2

try:
    import fcntl
except ImportError:
    fcntl = None

MODEL_CACHE_ENV = "ODM_AI_MODELS_PATH"


def _abs_path(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _legacy_model_cache_dir():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    if sys.platform == 'win32':
        program_data = os.getenv('PROGRAMDATA')
        if program_data:
            base_dir = os.path.join(program_data, "ODM")

    return os.path.join(os.path.abspath(base_dir), "storage", "models")


def _user_model_cache_dir():
    if sys.platform == 'win32':
        return None

    cache_dir = os.environ.get("XDG_CACHE_HOME")
    if not cache_dir:
        home = os.path.expanduser("~")
        if not home or home == "~":
            return None
        cache_dir = os.path.join(home, ".cache")

    return os.path.join(_abs_path(cache_dir), "odm", "models")


def _candidate_model_cache_dirs():
    candidates = []
    env_cache = os.environ.get(MODEL_CACHE_ENV)
    if env_cache:
        candidates.append(_abs_path(env_cache))

    candidates.append(_legacy_model_cache_dir())

    user_cache = _user_model_cache_dir()
    if user_cache:
        candidates.append(user_cache)

    candidates.append(os.path.join(tempfile.gettempdir(), "odm-ai-models"))

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)

    return deduped


def _can_prepare_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test_%s" % os.getpid())
        with open(test_file, "w"):
            pass
        os.remove(test_file)
        return True
    except Exception:
        return False


def _select_model_cache_dir(namespace, version, name):
    candidates = _candidate_model_cache_dirs()

    for base_dir in candidates:
        versioned_dir = os.path.join(base_dir, namespace, version)
        if os.path.isfile(os.path.join(versioned_dir, name)):
            return versioned_dir

    skipped = []
    for base_dir in candidates:
        versioned_dir = os.path.join(base_dir, namespace, version)
        if _can_prepare_dir(versioned_dir):
            if skipped:
                log.ODM_WARNING("Cannot write AI model cache directory %s, using %s" % (skipped[-1], versioned_dir))
            return versioned_dir
        skipped.append(versioned_dir)

    log.ODM_WARNING("Cannot prepare any AI model cache directory. Tried: %s" % ", ".join(skipped))
    return None


@contextmanager
def _model_download_lock(versioned_dir):
    lock_file = None
    try:
        lock_path = os.path.join(versioned_dir, ".download.lock")
        lock_file = open(lock_path, "w")
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        if lock_file is not None:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                except Exception:
                    pass
            lock_file.close()


def read_image(img_path):
    if img_path[-4:].lower() in [".dng", ".raw", ".nef"]:
        try:
            with rawpy.imread(img_path) as r:
                img = r.postprocess(output_bps=8, use_camera_wb=True, use_auto_wb=False)
        except:
            return None
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def get_model(namespace, url, version, name = "model.onnx"):
    version = version.replace(".", "_")

    versioned_dir = _select_model_cache_dir(namespace, version, name)
    if versioned_dir is None:
        return None

    model_file = os.path.join(versioned_dir, name)

    if os.path.isfile(model_file):
        return model_file

    with _model_download_lock(versioned_dir):
        if os.path.isfile(model_file):
            return model_file

        log.ODM_INFO("Using AI model cache %s" % versioned_dir)
        log.ODM_INFO("Downloading AI model from %s ..." % url)

        last_update = 0

        def callback(progress):
            nonlocal last_update

            time_has_elapsed = time.time() - last_update >= 2

            if time_has_elapsed or int(progress) == 100:
                log.ODM_INFO("Downloading: %s%%" % int(progress))
                last_update = time.time()

        try:
            downloaded_file = download(url, versioned_dir, progress_callback=callback)
        except Exception as e:
            log.ODM_WARNING("Cannot download %s: %s" % (url, str(e)))
            return None

        if os.path.basename(downloaded_file).lower().endswith(".zip"):
            log.ODM_INFO("Extracting %s ..." % downloaded_file)
            with zipfile.ZipFile(downloaded_file, 'r') as z:
                z.extractall(versioned_dir)
            os.remove(downloaded_file)
        
        if not os.path.isfile(model_file):
            log.ODM_WARNING("Cannot find %s (is the URL to the AI model correct?)" % model_file)
            return None
        else:
            return model_file
