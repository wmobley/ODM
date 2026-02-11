import time
import datetime
import os
import threading
import zipfile
import glob
import hashlib
import json
import shutil
import requests
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from opendm import log
from opendm import system
from opendm import config
from pyodm import Node, exceptions
from pyodm.utils import AtomicCounter
from pyodm.types import TaskInfo, TaskStatus
from opendm.osfm import OSFMContext, get_submodel_args_dict, get_submodel_argv
from opendm.utils import double_quote


def hash_file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

try:
    import queue
except ImportError:
    import Queue as queue

class LocalRemoteExecutor:
    """
    A class for performing OpenSfM reconstructions and full ODM pipeline executions
    using a mix of local and remote processing. Tasks are executed locally one at a time
    and remotely until a node runs out of available slots for processing. This allows us
    to use the processing power of the current machine as well as offloading tasks to a 
    network node.
    """
    def __init__(self, nodeUrl, rolling_shutter = False, rerun = False):
        # Ensure token is present if provided via environment and not already on the URL.
        # Default to https when targeting port 443 and scheme is missing.
        if "://" not in nodeUrl:
            # Heuristic: use https for port 443, else http.
            try:
                hostpart = nodeUrl
                # Extract port if present
                port_guess = None
                if ":" in nodeUrl:
                    hostpart, port_str = nodeUrl.rsplit(":", 1)
                    if port_str.isdigit():
                        port_guess = int(port_str)
                scheme_guess = "https" if port_guess == 443 else "http"
                nodeUrl = f"{scheme_guess}://{nodeUrl}"
            except Exception:
                nodeUrl = f"http://{nodeUrl}"

        parsed = urlparse(nodeUrl)
        # If caller provided http on port 443, upgrade to https for proper access.
        if parsed.scheme == "http" and parsed.port == 443:
            parsed = parsed._replace(scheme="https")
        query = parse_qs(parsed.query)
        if "token" not in query or not query.get("token"):
            env_token = os.environ.get("ODM_NODE_TOKEN")
            # Try port-specific token first (ODM_NODE_TOKEN_<PORT>)
            port = parsed.port
            if port:
                env_token = os.environ.get(f"ODM_NODE_TOKEN_{port}", env_token)
            if env_token:
                query["token"] = [env_token]
                new_query = urlencode(query, doseq=True)
                parsed = parsed._replace(query=new_query)
                # Preserve original scheme/netloc if provided; if we injected http://, strip it back out
                rebuilt = urlunparse(parsed)
                if nodeUrl.startswith("http://") or nodeUrl.startswith("https://"):
                    nodeUrl = rebuilt
                else:
                    # remove the injected scheme when caller passed host:port
                    nodeUrl = rebuilt.split("://", 1)[-1]
        self.node = Node.from_url(nodeUrl)
        self.params = {
            'tasks': [],
            'threads': [],
            'rolling_shutter': rolling_shutter,
            'rerun': rerun
        }
        self.node_online = True

        log.ODM_INFO("LRE: Initializing using cluster node %s:%s" % (self.node.host, self.node.port))
        try:
            info = self.node.info()
            log.ODM_INFO("LRE: Node is online and running %s version %s"  % (info.engine, info.engine_version))
        except exceptions.NodeConnectionError:
            log.ODM_WARNING("LRE: The node seems to be offline! We'll still process the dataset, but it's going to run entirely locally.")
            self.node_online = False
        except Exception as e:
            raise system.ExitException("LRE: An unexpected problem happened while opening the node connection: %s" % str(e))

    def set_projects(self, paths):
        self.project_paths = paths

    def run_reconstruction(self):
        self.run(ReconstructionTask)

    def run_toolchain(self):
        self.run(ToolchainTask)

    def run(self, taskClass):
        if not self.project_paths:
            return

        # Shared variables across threads
        class nonloc:
            error = None
            local_processing = False
            max_remote_tasks = None
        
        calculate_task_limit_lock = threading.Lock()
        finished_tasks = AtomicCounter(0)
        remote_running_tasks = AtomicCounter(0)

        # Create queue
        q = queue.Queue()
        for pp in self.project_paths:
            log.ODM_INFO("LRE: Adding to queue %s" % pp)
            q.put(taskClass(pp, self.node, self.params))

        def remove_task_safe(task):
            try:
                removed = task.remove()
            except exceptions.OdmError:
                removed = False
            return removed
        
        def cleanup_remote_tasks():
            if self.params['tasks']:
                log.ODM_WARNING("LRE: Attempting to cleanup remote tasks")
            else:
                log.ODM_INFO("LRE: No remote tasks left to cleanup")

            for task in self.params['tasks']:
                log.ODM_INFO("LRE: Removing remote task %s... %s" % (task.uuid, 'OK' if remove_task_safe(task) else 'NO'))

        def handle_result(task, local, error = None, partial=False):
            def cleanup_remote():
                if not partial and task.remote_task:
                    log.ODM_INFO("LRE: Cleaning up remote task (%s)... %s" % (task.remote_task.uuid, 'OK' if remove_task_safe(task.remote_task) else 'NO'))
                    self.params['tasks'].remove(task.remote_task)
                    task.remote_task = None

            if error:
                log.ODM_WARNING("LRE: %s failed with: %s" % (task, str(error)))
                
                # Special case in which the error is caused by a SIGTERM signal
                # this means a local processing was terminated either by CTRL+C or 
                # by canceling the task.
                if str(error) == "Child was terminated by signal 15":
                    system.exit_gracefully()

                task_limit_reached = isinstance(error, NodeTaskLimitReachedException)
                if task_limit_reached:
                    # Estimate the maximum number of tasks based on how many tasks
                    # are currently running
                    with calculate_task_limit_lock:
                        if nonloc.max_remote_tasks is None:
                            node_task_limit = 0
                            for t in self.params['tasks']:
                                try:
                                    info = t.info(with_output=-3)
                                    if info.status == TaskStatus.RUNNING and info.processing_time >= 0 and len(info.output) >= 3:
                                        node_task_limit += 1
                                except exceptions.OdmError:
                                    pass

                            nonloc.max_remote_tasks = max(1, node_task_limit)
                            log.ODM_INFO("LRE: Node task limit reached. Setting max remote tasks to %s" % node_task_limit)
                                

                # Retry, but only if the error is not related to a task failure
                if task.retries < task.max_retries and not isinstance(error, exceptions.TaskFailedError):
                    # Put task back in queue
                    # Don't increment the retry counter if this task simply reached the task
                    # limit count.
                    if not task_limit_reached:
                        task.retries += 1
                    task.wait_until = datetime.datetime.now() + datetime.timedelta(seconds=task.retries * task.retry_timeout)
                    cleanup_remote()
                    q.task_done()

                    log.ODM_INFO("LRE: Re-queueing %s (retries: %s)" % (task, task.retries))
                    q.put(task)
                    if not local: remote_running_tasks.increment(-1)
                    return
                else:
                    nonloc.error = error
                    finished_tasks.increment()
                    if not local: remote_running_tasks.increment(-1)
            else:
                if not partial:
                    log.ODM_INFO("LRE: %s finished successfully" % task)
                    finished_tasks.increment()
                    if not local: remote_running_tasks.increment(-1)

            cleanup_remote()
            if not partial: q.task_done()
            
        def local_worker():
            while True:
                # Block until a new queue item is available
                task = q.get()

                if task is None or nonloc.error is not None:
                    q.task_done()
                    break

                # Process local
                try:
                    nonloc.local_processing = True
                    task.process(True, handle_result)
                except Exception as e:
                    handle_result(task, True, e)
                finally:
                    nonloc.local_processing = False


        def remote_worker():
            while True:
                # Block until a new queue item is available
                task = q.get()

                if task is None or nonloc.error is not None:
                    q.task_done()
                    break
                
                # Yield to local processing
                if not nonloc.local_processing:
                    log.ODM_INFO("LRE: Yielding to local processing, sending %s back to the queue" % task)
                    q.put(task)
                    q.task_done()
                    time.sleep(0.05)
                    continue

                # If we've found an estimate of the limit on the maximum number of tasks
                # a node can process, we block until some tasks have completed
                if nonloc.max_remote_tasks is not None and remote_running_tasks.value >= nonloc.max_remote_tasks:
                    q.put(task)
                    q.task_done()
                    time.sleep(2)
                    continue

                # Process remote
                try:
                    remote_running_tasks.increment()
                    task.process(False, handle_result)
                except Exception as e:
                    handle_result(task, False, e)
        
        # Create queue thread
        local_thread = threading.Thread(target=local_worker)
        if self.node_online:
            remote_thread = threading.Thread(target=remote_worker)

        system.add_cleanup_callback(cleanup_remote_tasks)

        # Start workers
        local_thread.start()
        if self.node_online:
            remote_thread.start()

        # block until all tasks are done (or CTRL+C)
        try:
            while finished_tasks.value < len(self.project_paths) and nonloc.error is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.ODM_WARNING("LRE: CTRL+C")
            system.exit_gracefully()
        
        # stop workers
        q.put(None)
        if self.node_online:
            q.put(None)

        # Wait for queue thread
        local_thread.join()
        if self.node_online:
            remote_thread.join()

        # Wait for all remains threads
        for thrds in self.params['threads']:
            thrds.join()
        
        system.remove_cleanup_callback(cleanup_remote_tasks)
        cleanup_remote_tasks()

        if nonloc.error is not None:
            # Try not to leak access token
            if isinstance(nonloc.error, exceptions.NodeConnectionError):
                raise exceptions.NodeConnectionError("A connection error happened. Check the connection to the processing node and try again.")
            else:
                raise nonloc.error
        

class NodeTaskLimitReachedException(Exception):
    pass

class Task:
    def __init__(self, project_path, node, params, max_retries=5, retry_timeout=10):
        self.project_path = project_path
        self.node = node
        self.params = params
        self.wait_until = datetime.datetime.now() # Don't run this task until a certain time
        self.max_retries = max_retries
        self.retries = 0
        self.retry_timeout = retry_timeout
        self.remote_task = None

    def process(self, local, done):
        def handle_result(error = None, partial=False):
            done(self, local, error, partial)

        log.ODM_INFO("LRE: About to process %s %s" % (self, 'locally' if local else 'remotely'))
        
        if local:
            self._process_local(handle_result) # Block until complete
        else:
            now = datetime.datetime.now()
            if self.wait_until > now:
                wait_for = (self.wait_until - now).seconds + 1
                log.ODM_INFO("LRE: Waiting %s seconds before processing %s" % (wait_for, self))
                time.sleep(wait_for)

            # TODO: we could consider uploading multiple tasks
            # in parallel. But since we are using the same node
            # perhaps this wouldn't be a big speedup.
            self._process_remote(handle_result) # Block until upload is complete

    def path(self, *paths):
        return os.path.join(self.project_path, *paths)

    def touch(self, file):
        with open(file, 'w') as fout:
            fout.write("Done!\n")

    def create_seed_payload(self, paths, touch_files=[], max_attempts=2):
        """
        Build a seed.zip with required inputs. Validate the archive after writing
        to catch corruption early (ZipFile.testzip) and retry a limited number of times.
        """
        attempts = 0
        paths = list(filter(os.path.exists, map(lambda p: self.path(p), paths)))
        outfile = self.path("seed.zip")

        def wait_for_stable_file(path, label, max_wait=30, interval=0.5, stable_checks=3):
            last_size = -1
            last_mtime = -1
            stable_count = 0
            start = time.time()

            while (time.time() - start) <= max_wait:
                try:
                    stats = os.stat(path)
                    size = stats.st_size
                    mtime = stats.st_mtime
                    if size > 0 and size == last_size and mtime == last_mtime:
                        stable_count += 1
                        if stable_count >= stable_checks:
                            return True
                    else:
                        stable_count = 0
                    last_size = size
                    last_mtime = mtime
                except Exception:
                    stable_count = 0
                time.sleep(interval)

            log.ODM_WARNING("LRE: Timed out waiting for stable seed archive (%s) at %s" % (self, label))
            return False

        def describe_inputs():
            files = []
            for p in paths:
                if os.path.isdir(p):
                    for root, _, filenames in os.walk(p):
                        for filename in filenames:
                            full = os.path.join(root, filename)
                            try:
                                size = os.path.getsize(full)
                            except Exception:
                                size = -1
                            files.append((full, size))
                else:
                    try:
                        size = os.path.getsize(p)
                    except Exception:
                        size = -1
                    files.append((p, size))
            return files

        input_desc = describe_inputs()

        while attempts < max_attempts:
            attempts += 1
            try:
                with zipfile.ZipFile(outfile, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                    for p in paths:
                        if os.path.isdir(p):
                            for root, _, filenames in os.walk(p):
                                for filename in filenames:
                                    filename = os.path.join(root, filename)
                                    filename = os.path.normpath(filename)
                                    zf.write(filename, os.path.relpath(filename, self.project_path))
                        else:
                            zf.write(p, os.path.relpath(p, self.project_path))

                    for tf in touch_files:
                        zf.writestr(tf, "")

                # Validate archive
                with zipfile.ZipFile(outfile, "r") as zf:
                    bad_file = zf.testzip()
                    if bad_file is None:
                        stable = wait_for_stable_file(outfile, outfile)
                        if not stable:
                            log.ODM_WARNING("LRE: Seed archive (%s) did not stabilize on disk, retrying (%s/%s)"
                                            % (self, attempts, max_attempts))
                            os.remove(outfile)
                            continue

                        with zipfile.ZipFile(outfile, "r") as zf_recheck:
                            bad_file = zf_recheck.testzip()
                            if bad_file is not None:
                                log.ODM_WARNING("LRE: Corrupt seed archive (%s) detected after stabilization at %s on file %s, retrying (%s/%s)"
                                                % (self, outfile, bad_file, attempts, max_attempts))
                                os.remove(outfile)
                                continue

                        try:
                            size = os.path.getsize(outfile)
                            sha256 = hash_file_sha256(outfile)
                            log.ODM_INFO("LRE: Seed zip diagnostics for %s: path=%s, size=%s, sha256=%s"
                                         % (self, outfile, size, sha256))
                        except Exception as diag_err:
                            log.ODM_WARNING("LRE: Seed zip diagnostics failed for %s at %s: %s"
                                            % (self, outfile, str(diag_err)))

                        return outfile
                    else:
                        log.ODM_WARNING("LRE: Corrupt seed archive (%s) detected at %s on file %s, retrying (%s/%s)"
                                        % (self, outfile, bad_file, attempts, max_attempts))
                        os.remove(outfile)
            except Exception as e:
                log.ODM_WARNING("LRE: Failed to create seed archive (%s) attempt %s/%s: %s"
                                % (self, attempts, max_attempts, str(e)))
                try:
                    if os.path.exists(outfile):
                        os.remove(outfile)
                except Exception:
                    pass

        log.ODM_WARNING("LRE: Exhausted attempts creating seed archive for %s. Input files (path:size): %s"
                        % (self, ", ".join(["%s:%s" % (p, s) for p, s in input_desc])))
        raise Exception("Could not create a valid seed archive for %s after %s attempts" % (self, max_attempts))

    def _process_local(self, done):
        try:
            self.process_local()
            done()
        except Exception as e:
            done(e)
    
    def _process_remote(self, done):
        try:
            self.process_remote(done)
            done(error=None, partial=True) # Upload is completed, but processing is not (partial)
        except Exception as e:
            done(e)

    def execute_remote_task(self, done, seed_files = [], seed_touch_files = [], outputs = [], name_override = None):
        """
        Run a task by creating a seed file with all files in seed_files, optionally
        creating empty files (for flag checks) specified in seed_touch_files
        and returning the results specified in outputs. Yeah it's pretty cool!
        """
        use_import_path = os.environ.get("ODM_REMOTE_USE_IMPORT_PATH", "0") == "1"

        def build_images(seed_file):
            # Find all images
            images = glob.glob(self.path("images/**"))

            # Add GCP (optional)
            if os.path.exists(self.path("gcp_list.txt")):
                images.append(self.path("gcp_list.txt"))

            # Add GEO (optional)
            if os.path.exists(self.path("geo.txt")):
                images.append(self.path("geo.txt"))

            # Add seed file
            images.append(seed_file)
            return images

        class nonloc:
            last_update = 0

        def print_progress(percentage):
            if (time.time() - nonloc.last_update >= 2) or int(percentage) == 100:
                log.ODM_INFO("LRE: Upload of %s at [%s%%]" % (self, int(percentage)))
                nonloc.last_update = time.time()

        # Prefer import_path when enabled and supported to avoid seed.zip transfers
        task = None
        seed_file = None

        def log_import_path_status(label_path):
            try:
                if not label_path:
                    return
                if not os.path.exists(label_path):
                    log.ODM_WARNING("LRE: import_path %s does not exist yet" % label_path)
                    return
                stats = os.stat(label_path)
                kind = "dir" if os.path.isdir(label_path) else "file"
                top_entries = []
                if os.path.isdir(label_path):
                    try:
                        top_entries = os.listdir(label_path)[:10]
                    except Exception:
                        top_entries = []
                image_count = 0
                try:
                    import glob
                    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.tif", "*.tiff", "*.TIF", "*.TIFF"]
                    for pat in patterns:
                        image_count += len(glob.glob(os.path.join(label_path, "**", pat), recursive=True))
                except Exception:
                    image_count = 0
                log.ODM_INFO("LRE: import_path status %s (%s): size=%s mtime=%s images≈%s entries=%s" %
                             (label_path, kind, stats.st_size, stats.st_mtime, image_count, ",".join(top_entries)))
            except Exception as e:
                log.ODM_WARNING("LRE: failed to log import_path status for %s: %s" % (label_path, str(e)))

        # Optionally re-root the import_path to a shared base (e.g., Tapis working dir)
        # Try explicit override first, then fall back to the job working directory (Tapis)
        import_path_base = os.environ.get("ODM_IMPORT_PATH_BASE") or os.environ.get("_tapisJobWorkingDir")
        import_path_override = None
        if use_import_path:
            rel = None
            try:
                # If project_path is under /var/www/data, preserve the relative portion
                rel = os.path.relpath(self.project_path, "/var/www/data")
            except Exception:
                rel = None

            def discover_import_base(job_dir, project_rel):
                """
                Best-effort discovery of the runtime data base when ODM_IMPORT_PATH_BASE is not set
                or when it points to the raw job directory (missing the nodeodm_workdir* segment).
                """
                if not job_dir or not project_rel:
                    return None
                try:
                    import glob
                    project_uuid = project_rel.split(os.sep)[0]
                    pattern = os.path.join(job_dir, "nodeodm_workdir*", "runtime", "data", project_uuid)
                    matches = glob.glob(pattern)
                    if matches:
                        return os.path.dirname(matches[0])
                except Exception as e:
                    log.ODM_WARNING("LRE: Failed autodetecting import_path base from _tapisJobWorkingDir: %s" % str(e))
                return None

            candidate_base = import_path_base
            candidate_path = None
            if candidate_base and rel:
                try:
                    candidate_path = os.path.normpath(os.path.join(candidate_base, rel))
                except Exception:
                    candidate_path = None

            # If the candidate path does not exist (common when the nodeodm_workdir* segment is missing),
            # try to discover the runtime data base from the job directory.
            if rel:
                if not candidate_path or not os.path.exists(candidate_path):
                    job_dir = os.environ.get("_tapisJobWorkingDir")
                    discovered_base = discover_import_base(job_dir, rel)
                    if discovered_base:
                        import_path_base = discovered_base
                        candidate_path = os.path.normpath(os.path.join(import_path_base, rel))
                        log.ODM_INFO("LRE: Using discovered import_path base %s (candidate %s was missing)"
                                     % (import_path_base, candidate_base))
                    else:
                        if candidate_base:
                            log.ODM_WARNING("LRE: import_path candidate %s/%s does not exist"
                                            % (candidate_base, rel))
                import_path_override = candidate_path

            def has_top_level_images(path):
                try:
                    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.tif", "*.tiff", "*.TIF", "*.TIFF"]
                    for pat in patterns:
                        if glob.glob(os.path.join(path, pat)):
                            return True
                except Exception:
                    pass
                return False

            def flatten_import_path(base_path):
                """
                NodeODM import_path expects the images directory directly at the import root.
                Submodels store images under <project>/images, which leads to images/images.
                Build a shallow view with images at the root (symlinks) and carry over support files.
                """
                images_dir = os.path.join(base_path, "images")
                if not os.path.isdir(images_dir):
                    return base_path

                # If images already exist at the top level, nothing to flatten.
                if has_top_level_images(base_path):
                    return base_path

                flat_dir = os.path.join(base_path, "_import_path_flat")
                try:
                    if os.path.exists(flat_dir):
                        shutil.rmtree(flat_dir)
                    os.makedirs(flat_dir, exist_ok=True)

                    for root, _, files in os.walk(images_dir):
                        rel_root = os.path.relpath(root, images_dir)
                        dst_root = flat_dir if rel_root == "." else os.path.join(flat_dir, rel_root)
                        os.makedirs(dst_root, exist_ok=True)
                        for filename in files:
                            src = os.path.join(root, filename)
                            dst = os.path.join(dst_root, filename)
                            try:
                                os.symlink(src, dst)
                            except FileExistsError:
                                pass
                            except Exception as e:
                                log.ODM_WARNING("LRE: Failed linking %s -> %s: %s" % (src, dst, str(e)))

                    support_files = ["gcp_list.txt", "align.las", "align.laz", "align.tif"]
                    for sf in support_files:
                        src_sf = os.path.join(base_path, sf)
                        if os.path.exists(src_sf):
                            dst_sf = os.path.join(flat_dir, sf)
                            try:
                                os.symlink(src_sf, dst_sf)
                            except FileExistsError:
                                pass
                            except Exception as e:
                                log.ODM_WARNING("LRE: Failed linking support file %s -> %s: %s" % (src_sf, dst_sf, str(e)))

                    log.ODM_INFO("LRE: Using flattened import_path view for submodel images at %s" % flat_dir)
                    return flat_dir
                except Exception as e:
                    log.ODM_WARNING("LRE: Failed to build flattened import_path at %s: %s" % (flat_dir, str(e)))
                    return base_path

            # Flatten submodel layout so NodeODM sees images at the import root (avoids images/images).
            target_import_path = import_path_override or self.project_path
            import_path_override = flatten_import_path(target_import_path)

            log_import_path_status(import_path_override)

        if use_import_path:
            try:
                task_name = name_override or str(self)
                # Prefer native pyodm helper when available
                if hasattr(self.node, "create_task_from_path"):
                    imp_path = import_path_override or self.project_path
                    log.ODM_INFO("LRE: Attempting import_path submission for %s via %s" % (self, imp_path))
                    task = self.node.create_task_from_path(imp_path,
                            get_submodel_args_dict(config.config()),
                            name=task_name)
                else:
                    # Fallback: manually POST import_path to the node
                    host = getattr(self.node, "host", None) or getattr(self.node, "hostname", None)
                    port = getattr(self.node, "port", None)
                    token = getattr(self.node, "token", None)
                    if host is None or port is None:
                        raise Exception("Missing node host/port for import_path submission")

                    # Try HTTPS first if the node indicates SSL or is using port 443; otherwise try HTTP first and fallback.
                    prefers_ssl = getattr(self.node, "ssl", False) or str(port) == "443"
                    protocols = ["https", "http"] if prefers_ssl else ["http", "https"]

                    options_dict = get_submodel_args_dict(config.config())
                    options_array = []
                    for k, v in options_dict.items():
                        options_array.append({"name": k, "value": v})

                    imp_path = import_path_override or self.project_path
                    payload = {
                        "name": task_name,
                        "import_path": imp_path,
                        "options": json.dumps(options_array),
                        "skipPostProcessing": True,
                        "outputs": json.dumps(outputs or [])
                    }

                    last_error = None
                    for scheme in protocols:
                        url_base = f"{scheme}://{host}:{port}/task/new"
                        if token:
                            url_base = f"{url_base}?token={token}"
                        try:
                            log.ODM_INFO("LRE: Attempting import_path submission (manual, %s) for %s via %s" % (scheme, self, self.project_path))
                            resp = requests.post(url_base, data=payload, timeout=300)
                            if resp.status_code != 200:
                                # If we hit HTTPS but sent plain HTTP (or vice-versa), try the alternate scheme
                                if "plain HTTP request was sent to HTTPS port" in resp.text:
                                    last_error = Exception("HTTP->HTTPS mismatch: %s" % resp.text)
                                    continue
                                raise Exception("HTTP %s: %s" % (resp.status_code, resp.text))
                            data = resp.json()
                            if data.get("error"):
                                raise Exception(data.get("error"))
                            if not data.get("uuid"):
                                raise Exception("Node import_path response missing uuid")
                            class SimpleTask:
                                def __init__(self, uuid, node):
                                    self.uuid = uuid
                                    self.node = node
                            task = SimpleTask(data["uuid"], self.node)
                            break
                        except Exception as sub_err:
                            last_error = sub_err
                            continue
                    if task is None:
                        if last_error:
                            raise last_error
                        raise Exception("Unable to submit import_path task")
                    # Provide a minimal Task-like wrapper so downstream cleanup/limit logic works
                    class SimpleTask:
                        def __init__(self, uuid, node):
                            self.uuid = uuid
                            self.node = node
                            self.remote_task = None

                        def remove(self):
                            # Try both delete_task and remove_task if available
                            for attr in ("delete_task", "remove_task"):
                                fn = getattr(self.node, attr, None)
                                if callable(fn):
                                    try:
                                        return fn(self.uuid)
                                    except Exception:
                                        continue
                            raise exceptions.OdmError("remove not supported for import_path task %s" % self.uuid)

                        def info(self, with_output=None):
                            try:
                                log.ODM_INFO("LRE: Attempting to get info for import_path task %s via %s" % (self, getattr(self.node, "url", self.node)))
                                fn = getattr(self.node, "get_task_info", None)
                                if callable(fn):
                                    return fn(self.uuid, with_output=with_output)
                                log.ODM_WARNING("LRE: import_path task %s has no get_task_info method on node %s; falling back to /task/<uuid>/info" %
                                                (self.uuid, type(self.node).__name__))
                                # Fallback to generic Node.get (pyodm.Node)
                                fn_get = getattr(self.node, "get", None)
                                if callable(fn_get):
                                    query = {}
                                    if with_output is not None:
                                        query["with_output"] = with_output
                                    info_json = fn_get(f"/task/{self.uuid}/info", query)
                                    return TaskInfo(info_json)
                            except Exception as info_exc:  # noqa: BLE001 - diagnostic logging
                                log.ODM_WARNING("LRE: import_path task %s get_task_info failed: %s" % (self.uuid, str(info_exc)))
                                return None
                            # Fallback: return None to signal that info is unavailable
                            return None

                        def wait_for_completion(self, *args, **kwargs):
                            # Try native wait if available
                            fn = getattr(self.node, "wait_for_task_completion", None)
                            if callable(fn):
                                return fn(self.uuid, *args, **kwargs)
                            # Otherwise, no-op/fallback: return a minimal object
                            return self.info()

                        def download_assets(self, *args, **kwargs):
                            fn = getattr(self.node, "download_assets", None)
                            if callable(fn):
                                return fn(self.uuid, *args, **kwargs)
                            # Nothing to download in import_path mode; treat as no-op
                            return True

                        def output(self, *args, **kwargs):
                            fn = getattr(self.node, "get_task_output", None)
                            if callable(fn):
                                return fn(self.uuid, *args, **kwargs)
                            return []

                    task = SimpleTask(data["uuid"], self.node)
            except Exception as e:
                # Do not fall back to seed.zip when import_path is requested; fail fast
                log.ODM_WARNING("LRE: import_path submission failed for %s (%s); not falling back to seed.zip because ODM_REMOTE_USE_IMPORT_PATH=1"
                                % (self, str(e)))
                raise

        # Upload task (retry once if the remote node rejects the seed archive)
        if task is None:
            seed_attempt = 0
            max_seed_retries = 1
            while seed_attempt <= max_seed_retries:
                seed_file = self.create_seed_payload(seed_files, touch_files=seed_touch_files)
                images = build_images(seed_file)
                try:
                    size = os.path.getsize(seed_file)
                    sha = hash_file_sha256(seed_file)
                    log.ODM_INFO("LRE: Seed zip ready for upload %s: path=%s, size=%s, sha256=%s" % (self, seed_file, size, sha))
                except Exception as diag_err:
                    log.ODM_WARNING("LRE: Unable to log seed zip diagnostics for %s at %s: %s" % (self, seed_file, str(diag_err)))
                try:
                    task_name = name_override or str(self)
                    task = self.node.create_task(images,
                            get_submodel_args_dict(config.config()),
                            name=task_name,
                            progress_callback=print_progress,
                            skip_post_processing=True,
                            outputs=outputs)
                    break
                except exceptions.NodeResponseError as e:
                    message = str(e)
                    if 'seed.zip failed integrity check' in message and seed_attempt < max_seed_retries:
                        seed_attempt += 1
                        log.ODM_WARNING("LRE: Remote node rejected seed archive (%s) for %s; regenerating and retrying (%s/%s)"
                                        % (message, self, seed_attempt, max_seed_retries))
                        continue
                    raise
        self.remote_task = task

        # Keep seed file for debugging
        if seed_file:
            log.ODM_INFO("LRE: Keeping seed archive for debugging at %s" % seed_file)

        # Keep track of tasks for cleanup
        self.params['tasks'].append(task)

        # Check status
        info = task.info()
        if info is None:
            raise Exception("LRE: task.info() returned None for %s (%s)" %
                            (self, getattr(task, "uuid", "unknown")))
        if info.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]:
            def monitor():
                class nonloc:
                    status_callback_calls = 0
                    last_update = 0

                def status_callback(info):
                    # If a task switches from RUNNING to QUEUED, then we need to 
                    # stop the process and re-add the task to the queue.
                    if info.status == TaskStatus.QUEUED:
                        log.ODM_WARNING("LRE: %s (%s) turned from RUNNING to QUEUED. Re-adding to back of the queue." % (self, task.uuid))
                        raise NodeTaskLimitReachedException("Delayed task limit reached")
                    elif info.status == TaskStatus.RUNNING:
                        # Print a status message once in a while
                        nonloc.status_callback_calls += 1
                        if nonloc.status_callback_calls > 30:
                            log.ODM_INFO("LRE: %s (%s) is still running" % (self, task.uuid))
                            nonloc.status_callback_calls = 0
                try:
                    def print_progress(percentage):
                        if (time.time() - nonloc.last_update >= 2) or int(percentage) == 100:
                            log.ODM_INFO("LRE: Download of %s at [%s%%]" % (self, int(percentage)))
                            nonloc.last_update = time.time()

                    status_retries = 0
                    max_status_retries = 8
                    retry_backoff = 5

                    while True:
                        try:
                            try:
                                node_url = None
                                candidate = getattr(self.node, "url", None)
                                if callable(candidate):
                                    try:
                                        node_url = candidate()
                                    except Exception:
                                        node_url = None
                                else:
                                    node_url = candidate

                                if not node_url:
                                    log.ODM_INF(f"Wills DEBUG node {self.node.__dict__}")
                                    proto = getattr(self.node, "protocol", "http")
                                    host = getattr(self.node, "host", "")
                                    port = getattr(self.node, "port", "")
                                    port_part = f":{port}" if port else ""
                                    node_url = f"{proto}://{host}{port_part}"

                                log.ODM_INFO("LRE: polling %s/task/%s/info" % (node_url, task.uuid))
                            except Exception:
                                pass

                            task.wait_for_completion(status_callback=status_callback)
                            # Defensive: verify the task actually reached COMPLETED before moving on.
                            info_check = None
                            try:
                                info_check = task.info(with_output=-3)
                            except Exception:
                                info_check = None

                            status_val = getattr(info_check, "status", None) if info_check else None
                            progress_val = getattr(info_check, "progress", None) if info_check else None

                            if info_check:
                                try:
                                    raw_payload = getattr(info_check, "_raw", None)
                                    if raw_payload is None:
                                        raw_payload = getattr(info_check, "__dict__", info_check)
                                    log.ODM_INFO("LRE: post-completion status check for %s (%s): raw=%s"
                                                 % (self, task.uuid, raw_payload))
                                except Exception:
                                    pass

                                log.ODM_INFO("LRE: post-completion status check for %s (%s): status=%s code=%s progress=%s (missing_progress=%s)"
                                             % (self, task.uuid, status_val, getattr(status_val, "code", status_val),
                                                progress_val, progress_val is None))

                            status_running = (
                                status_val == TaskStatus.RUNNING
                                or getattr(status_val, "code", None) == TaskStatus.RUNNING.value
                                or getattr(status_val, "code", None) == 20
                            )
                            progress_hundred = progress_val is not None and progress_val >= 100

                            if status_running and not progress_hundred:
                                # Some path-based tasks on ClusterODM/NodeODM can briefly report completion;
                                # keep polling until the backend switches to COMPLETED.
                                log.ODM_WARNING("LRE: %s (%s) reported completion but is still RUNNING; continuing to poll"
                                                % (self, task.uuid))
                                time.sleep(5)
                                continue
                            break
                        except exceptions.TaskFailedError:
                            # Bubble up task failures (handled below)
                            raise
                        except Exception as poll_err:
                            # ClusterODM can briefly return "status <uuid> not found" while
                            # it distributes the task to worker nodes. Treat this as a
                            # transient condition and retry a few times before failing.
                            err_msg = str(poll_err).lower()
                            if "not found" in err_msg and "status" in err_msg and status_retries < max_status_retries:
                                status_retries += 1
                                wait_for = retry_backoff * status_retries
                                log.ODM_WARNING(
                                    "LRE: %s (%s) status not available yet (attempt %s/%s), retrying in %ss" %
                                    (self, task.uuid, status_retries, max_status_retries, wait_for))
                                time.sleep(wait_for)
                                continue

                            raise
                    log.ODM_INFO("LRE: Downloading assets for %s" % self)
                    task.download_assets(self.project_path, progress_callback=print_progress)
                    log.ODM_INFO("LRE: Downloaded and extracted assets for %s" % self)
                    # Some import_path tasks extract into /var/www/data/<task_uuid> instead of the submodel path.
                    # If key outputs are missing in the submodel path, try to copy from the task uuid folder.
                    try:
                        if outputs:
                            missing = []
                            for rel in outputs:
                                dst = os.path.join(self.project_path, rel)
                                if not os.path.exists(dst):
                                    missing.append(rel)
                            if missing:
                                data_root = os.path.dirname(os.path.dirname(os.path.dirname(self.project_path)))
                                alt_root = os.path.join(data_root, task.uuid)
                                if os.path.isdir(alt_root):
                                    log.ODM_WARNING("LRE: Missing outputs under %s; copying from %s (missing=%s)" %
                                                    (self.project_path, alt_root, ", ".join(missing)))
                                    for rel in outputs:
                                        src = os.path.join(alt_root, rel)
                                        dst = os.path.join(self.project_path, rel)
                                        if os.path.isdir(src):
                                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                                            shutil.copytree(src, dst, dirs_exist_ok=True)
                                        elif os.path.isfile(src):
                                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                                            shutil.copy2(src, dst)
                    except Exception as fix_exc:  # noqa: BLE001 - best effort fix
                        log.ODM_WARNING("LRE: Failed to relocate import_path outputs for %s: %s" % (self, str(fix_exc)))
                    done()
                except exceptions.TaskFailedError as e:
                    # Try to get output
                    try:
                        output_lines = []
                        try:
                            output_lines = task.output()
                        except Exception as output_exc:  # noqa: BLE001 - best effort logging
                            log.ODM_WARNING("LRE: Could not retrieve task output for %s (%s): %s" % (self, task.uuid, str(output_exc)))

                        # Save to file (with a helpful placeholder when output is empty)
                        error_log_path = self.path("error.log")
                        log_dir = os.path.dirname(error_log_path)
                        if log_dir and not os.path.exists(log_dir):
                            os.makedirs(log_dir, exist_ok=True)

                        snippet = ""
                        log_write_failed = None
                        try:
                            with open(error_log_path, 'w') as f:
                                if output_lines:
                                    f.write('\n'.join(output_lines) + '\n')
                                    snippet = "\n".join(output_lines[-10:])
                                else:
                                    # Try to gather any status info so the log file is not empty
                                    try:
                                        info = task.info()
                                        status_str = getattr(info, "status", None)
                                        status_str = status_str if status_str is not None else "unknown"
                                        f.write("Task failed but returned no output. Status: %s\n" % status_str)
                                        error_attr = getattr(info, "error", None)
                                        if error_attr:
                                            f.write("Error: %s\n" % error_attr)
                                        exit_code = getattr(info, "exit_code", None)
                                        if exit_code is not None:
                                            f.write("Exit code: %s\n" % exit_code)
                                    except Exception as info_exc:  # noqa: BLE001 - best effort logging
                                        f.write("Task failed, no output returned, and task.info() could not be retrieved: %s\n" % str(info_exc))
                                    snippet = "Task failed but no remote output was available; check the node logs."
                        except Exception as write_exc:
                            log_write_failed = str(write_exc)
                            log.ODM_WARNING("LRE: Failed to write error log at %s: %s" % (error_log_path, log_write_failed))

                        msg = "(%s) failed with task output: %s" % (task.uuid, snippet or "No output retrieved")
                        if log_write_failed:
                            msg += "\nFull log could not be written (%s)" % log_write_failed
                        else:
                            msg += "\nFull log saved at %s" % error_log_path
                        done(exceptions.TaskFailedError(msg))
                    except:
                        log.ODM_WARNING("LRE: Could not retrieve task output for %s (%s)" % (self, task.uuid))
                        done(e)
                except Exception as e:
                    done(e)

            # Launch monitor thread and return
            t = threading.Thread(target=monitor)
            self.params['threads'].append(t)
            t.start()
        elif info.status == TaskStatus.QUEUED:
            raise NodeTaskLimitReachedException("Task limit reached")
        else:
            raise Exception("Could not send task to node, task status is %s" % str(info.status))

    
    def process_local(self):
        raise NotImplementedError()
    
    def process_remote(self, done):
        raise NotImplementedError()

    def __str__(self):
        return os.path.basename(self.project_path)


class ReconstructionTask(Task):
    def process_local(self):
        octx = OSFMContext(self.path("opensfm"))
        log.ODM_INFO("==================================")
        log.ODM_INFO("Local Reconstruction %s" % octx.name())
        log.ODM_INFO("==================================")
        octx.feature_matching(self.params['rerun'])
        octx.create_tracks(self.params['rerun'])
        octx.reconstruct(self.params['rolling_shutter'], True, self.params['rerun'])
    
    def process_remote(self, done):
        octx = OSFMContext(self.path("opensfm"))
        if not octx.is_feature_matching_done() or not octx.is_reconstruction_done() or self.params['rerun']:
            submodel_name = octx.name()
            self.execute_remote_task(done, seed_files=["opensfm/exif", 
                                                "opensfm/camera_models.json",
                                                "opensfm/reference_lla.json"],
                                    seed_touch_files=["opensfm/split_merge_stop_at_reconstruction.txt"],
                                    outputs=["opensfm/matches", "opensfm/features", 
                                            "opensfm/reconstruction.json",
                                            "opensfm/tracks.csv",
                                            "cameras.json"],
                                    name_override=f"{submodel_name}_recon")
        else:
            log.ODM_INFO("Already processed feature matching and reconstruction for %s" % octx.name())
            done()

class ToolchainTask(Task):
    def process_local(self):
        completed_file = self.path("toolchain_completed.txt")
        submodel_name = os.path.basename(self.project_path)
        
        if not os.path.exists(completed_file) or self.params['rerun']:
            log.ODM_INFO("=============================")
            log.ODM_INFO("Local Toolchain %s" % self)
            log.ODM_INFO("=============================")

            submodels_path = os.path.abspath(self.path(".."))
            argv = get_submodel_argv(config.config(), submodels_path, submodel_name)

            # Always invoke run.py through venv python
            cmd = ["python3"] + argv

            # Re-run the ODM toolchain on the submodel
            system.run(" ".join(map(double_quote, cmd)), env_vars=os.environ.copy())

            # This will only get executed if the command above succeeds
            self.touch(completed_file)
        else:
            log.ODM_INFO("Already processed toolchain for %s" % submodel_name)
    
    def process_remote(self, done):
        completed_file = self.path("toolchain_completed.txt")
        submodel_name = os.path.basename(self.project_path)

        def handle_result(error = None):
            # Mark task as completed if no error
            if error is None:
                self.touch(completed_file)
            done(error=error)

        if not os.path.exists(completed_file) or self.params['rerun']:
            # If the reconstruction artifacts are present, include them in the seed; otherwise
            # create empty markers so downstream steps don't fail on missing paths.
            optional_osfm_dirs = ["opensfm/features", "opensfm/matches", "opensfm/exif"]
            present_optional_dirs = [d for d in optional_osfm_dirs if os.path.exists(self.path(d))]
            missing_optional_markers = [f"{d}/empty" for d in optional_osfm_dirs if not os.path.exists(self.path(d))]

            self.execute_remote_task(handle_result, seed_files=["opensfm/camera_models.json",
                                                "opensfm/reference_lla.json",
                                                "opensfm/reconstruction.json",
                                                "opensfm/tracks.csv"] + present_optional_dirs,
                                seed_touch_files=missing_optional_markers,
                                outputs=["odm_orthophoto/cutline.gpkg",
                                        "odm_orthophoto/odm_orthophoto_cut.tif",
                                        "odm_orthophoto/odm_orthophoto_feathered.tif",
                                        "odm_dem",
                                        "odm_report",
                                        "odm_georeferencing"],
                                name_override=f"{submodel_name}_toolchain")
        else:
            log.ODM_INFO("Already processed toolchain for %s" % submodel_name)
            handle_result()
