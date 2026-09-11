from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
_spec = spec_from_file_location("_local_inspection", Path(__file__).with_name("_inspection_impl.py"))
_module = module_from_spec(_spec); _spec.loader.exec_module(_module)
draw_labelled_boxes = _module.draw_labelled_boxes
DetectionInspectionCallback = _module.DetectionInspectionCallback
