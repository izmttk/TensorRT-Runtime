import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def set_severity(severity):
    TRT_LOGGER.min_severity = severity
    
def get_severity():
    return TRT_LOGGER.min_severity

def log(level, msg):
    TRT_LOGGER.log(level, msg)

def info(msg):
    TRT_LOGGER.log(trt.Logger.INFO, msg)
    
def warning(msg):
    TRT_LOGGER.log(trt.Logger.WARNING, msg)
    
def error(msg):
    TRT_LOGGER.log(trt.Logger.ERROR, msg)
    
def verbose(msg):
    TRT_LOGGER.log(trt.Logger.VERBOSE, msg)
    
def internal_error(msg):
    TRT_LOGGER.log(trt.Logger.INTERNAL_ERROR, msg)