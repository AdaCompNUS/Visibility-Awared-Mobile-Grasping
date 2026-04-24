import logging
import time

from rich.logging import RichHandler

# Create a logger
log = logging.getLogger("rich")
log.setLevel(logging.DEBUG)

# Create a RichHandler and set its level
handler = RichHandler(rich_tracebacks=True)
handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
handler.setFormatter(formatter)

# Add the handler to the logger
if not log.handlers:
    log.addHandler(handler)

_last_log_time = {}


def throttle(period, msg, *args, **kwargs):
    now = time.time()
    # Using the message itself as a key might be too simple if the message is formatted.
    # A better key would be the call site (filename, lineno), but that's more complex.
    # For "simple and stupid", the message content as a key is a start.
    key = msg % args if args else msg
    last_time = _last_log_time.get(key)
    if last_time is None or now - last_time > period:
        _last_log_time[key] = now
        return True
    return False


def log_warn_throttle(period, msg, *args, **kwargs):
    if throttle(period, msg, *args, **kwargs):
        log.warning(msg, *args, **kwargs)


def log_info_throttle(period, msg, *args, **kwargs):
    if throttle(period, msg, *args, **kwargs):
        log.info(msg, *args, **kwargs)


def log_debug_throttle(period, msg, *args, **kwargs):
    if throttle(period, msg, *args, **kwargs):
        log.debug(msg, *args, **kwargs)


def log_error_throttle(period, msg, *args, **kwargs):
    if throttle(period, msg, *args, **kwargs):
        log.error(msg, *args, **kwargs)


def log_critical_throttle(period, msg, *args, **kwargs):
    if throttle(period, msg, *args, **kwargs):
        log.critical(msg, *args, **kwargs)
