import os
import errno
import json
import datetime
import sys
import subprocess
import string
import signal
import io
import shutil
import time
import threading
import queue
from collections import deque

from opendm import context
from opendm import log

class SubprocessException(Exception):
    def __init__(self, msg, errorCode):
        super().__init__(msg)
        self.errorCode = errorCode


class SubprocessTimeoutException(SubprocessException):
    def __init__(self, msg, errorCode, timeout_type, elapsed=None, idle_for=None):
        super().__init__(msg, errorCode)
        self.timeout_type = timeout_type
        self.elapsed = elapsed
        self.idle_for = idle_for

class ExitException(Exception):
    pass

def get_ccd_widths():
    """Return the CCD Width of the camera listed in the JSON defs file."""
    with open(context.ccd_widths_path) as f:
        sensor_data = json.loads(f.read())
    return dict(zip(map(string.lower, sensor_data.keys()), sensor_data.values()))

running_subprocesses = []
cleanup_callbacks = []

def add_cleanup_callback(func):
    global cleanup_callbacks
    cleanup_callbacks.append(func)

def remove_cleanup_callback(func):
    global cleanup_callbacks

    try:
        cleanup_callbacks.remove(func)
    except ValueError as e:
        log.ODM_EXCEPTION("Tried to remove %s from cleanup_callbacks but got: %s" % (str(func), str(e)))

def exit_gracefully():
    global running_subprocesses
    global cleanup_callbacks

    log.ODM_WARNING("Caught TERM/INT signal, attempting to exit gracefully...")

    for cb in cleanup_callbacks:
        cb()

    for sp in running_subprocesses:
        log.ODM_WARNING("Sending TERM signal to PID %s..." % sp.pid)
        if sys.platform == 'win32':
            os.kill(sp.pid, signal.CTRL_C_EVENT)
        else:
            os.killpg(os.getpgid(sp.pid), signal.SIGTERM)
    
    os._exit(1)

def sighandler(signum, frame):
    exit_gracefully()

signal.signal(signal.SIGINT, sighandler)
signal.signal(signal.SIGTERM, sighandler)

def _enqueue_subprocess_output(stream, q):
    try:
        for line in stream:
            q.put(line)
    finally:
        q.put(None)


def _log_subprocess_diagnostics(p, cmd, elapsed, idle_for, lines):
    pid = getattr(p, "pid", None)

    last_output = list(lines)[-5:]
    if last_output:
        log.ODM_WARNING("Subprocess diagnostics for `%s`: last output lines:\n%s" % (
            cmd, "\n".join(last_output)
        ))
    else:
        log.ODM_WARNING("Subprocess diagnostics for `%s`: no output captured" % cmd)

    log.ODM_WARNING(
        "Subprocess diagnostics for `%s`: pid=%s elapsed=%.1fs idle_for=%.1fs" % (
            cmd, pid, elapsed, idle_for
        )
    )

    if sys.platform == 'win32' or pid is None:
        return

    try:
        pgid = os.getpgid(pid)
        ps_cmd = [
            "ps",
            "-o", "pid,ppid,pgid,stat,etime,%cpu,%mem,wchan,command",
            "-g", str(pgid)
        ]
        ps_output = subprocess.check_output(ps_cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace").strip()
        if ps_output:
            log.ODM_WARNING("Subprocess diagnostics for `%s`: process group snapshot:\n%s" % (cmd, ps_output))
    except Exception as e:
        log.ODM_WARNING("Could not capture process diagnostics for `%s`: %s" % (cmd, str(e)))


def _terminate_subprocess(p, cmd, reason, grace_period=5):
    if p.poll() is not None:
        return p.wait()

    log.ODM_WARNING("Terminating subprocess for `%s` (%s)" % (cmd, reason))

    try:
        if sys.platform == 'win32':
            p.terminate()
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception as e:
        log.ODM_WARNING("Failed to send TERM to subprocess `%s`: %s" % (cmd, str(e)))

    deadline = time.time() + grace_period
    while time.time() < deadline:
        retcode = p.poll()
        if retcode is not None:
            return retcode
        time.sleep(0.1)

    try:
        if sys.platform == 'win32':
            p.kill()
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception as e:
        log.ODM_WARNING("Failed to send KILL to subprocess `%s`: %s" % (cmd, str(e)))

    return p.wait()


def run(cmd, env_paths=[context.superbuild_bin_path], env_vars={}, packages_paths=context.python_packages_paths,
        quiet=False, timeout=None, idle_timeout=None, poll_interval=1.0):
    """Run a system command"""
    global running_subprocesses

    if not quiet:
        log.ODM_INFO('running %s' % cmd)
    env = os.environ.copy()

    sep = ":"
    if sys.platform == 'win32':
        sep = ";"

    if len(env_paths) > 0:
        env["PATH"] = env["PATH"] + sep + sep.join(env_paths)
    
    if len(packages_paths) > 0:
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + sep + sep.join(packages_paths) 
    if sys.platform == 'darwin':
        # Propagate DYLD_LIBRARY_PATH
        cmd = "export DYLD_LIBRARY_PATH=\"%s\" && %s" % (env.get("DYLD_LIBRARY_PATH", ""), cmd)

    for k in env_vars:
        env[k] = str(env_vars[k])

    p = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    running_subprocesses.append(p)
    lines = deque()
    start_time = time.time()
    last_output_time = start_time
    output_queue = queue.Queue()
    reader = threading.Thread(target=_enqueue_subprocess_output, args=(p.stdout, output_queue), daemon=True)
    reader.start()

    try:
        while True:
            try:
                line = output_queue.get(timeout=poll_interval)
                if line is None:
                    break

                print(line, end="")
                line = line.strip()
                if line:
                    lines.append(line)
                    if len(lines) == 11:
                        lines.popleft()
                last_output_time = time.time()
            except queue.Empty:
                now = time.time()
                elapsed = now - start_time
                idle_for = now - last_output_time

                if timeout is not None and elapsed > timeout:
                    _log_subprocess_diagnostics(p, cmd, elapsed, idle_for, lines)
                    retcode = _terminate_subprocess(p, cmd, "timeout after %.1fs" % elapsed)
                    raise SubprocessTimeoutException(
                        "Child exceeded timeout after %.1f seconds" % elapsed,
                        retcode if retcode is not None else 124,
                        "timeout",
                        elapsed=elapsed,
                        idle_for=idle_for,
                    )

                if idle_timeout is not None and idle_for > idle_timeout:
                    _log_subprocess_diagnostics(p, cmd, elapsed, idle_for, lines)
                    retcode = _terminate_subprocess(p, cmd, "idle timeout after %.1fs without output" % idle_for)
                    raise SubprocessTimeoutException(
                        "Child produced no output for %.1f seconds" % idle_for,
                        retcode if retcode is not None else 125,
                        "idle_timeout",
                        elapsed=elapsed,
                        idle_for=idle_for,
                    )

                if p.poll() is not None and output_queue.empty():
                    break

        retcode = p.wait()
    finally:
        if p in running_subprocesses:
            running_subprocesses.remove(p)

    if not quiet:
        log.logger.log_json_process(cmd, retcode, list(lines))

    if retcode < 0:
        raise SubprocessException("Child was terminated by signal {}".format(-retcode), -retcode)
    elif retcode > 0:
        raise SubprocessException("Child returned {}".format(retcode), retcode)


def now():
    """Return the current time"""
    return datetime.datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')


def now_raw():
    return datetime.datetime.now()


def benchmark(start, benchmarking_file, process):
    """
    runs a benchmark with a start datetime object
    :return: the running time (delta)
    """
    # Write to benchmark file
    delta = (datetime.datetime.now() - start).total_seconds()
    with open(benchmarking_file, 'a') as b:
        b.write('%s runtime: %s seconds\n' % (process, delta))

def mkdir_p(path):
    """Make a directory including parent directories.
    """
    try:
        os.makedirs(path)
    except os.error as exc:
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise

# Python2 shutil.which
def which(program):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,program)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p

def link_file(src, dst):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))

    if not os.path.isfile(dst):
        if sys.platform == 'win32':
            os.link(src, dst)
        else:
            os.symlink(os.path.relpath(os.path.abspath(src), os.path.dirname(os.path.abspath(dst))), dst)

def move_files(src, dst):
    if not os.path.isdir(dst):
        raise IOError("Not a directory: %s" % dst)

    for f in os.listdir(src):
        if os.path.isfile(os.path.join(src, f)):
            shutil.move(os.path.join(src, f), dst)

def delete_files(folder, exclude=()):
    if not os.path.isdir(folder):
        return

    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            if not exclude or not f.endswith(exclude):
                os.unlink(os.path.join(folder, f))
