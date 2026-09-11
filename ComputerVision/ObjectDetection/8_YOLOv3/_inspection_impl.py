"""Local implementation for YOLOv3 labelled validation grids."""
import os
import cv2
import numpy as np
import tensorflow as tf

def _name(names, class_id): return names[class_id] if 0 <= int(class_id) < len(names) else f"class_{class_id}"
def draw_labelled_boxes(image, boxes, labels, names, *, normalized, prediction=False, max_draw=20):
    canvas=image.copy(); h,w=canvas.shape[:2]; color=(0,220,0) if prediction else (0,0,255)
    for box,label in list(zip(boxes,labels))[:max_draw]:
        class_id,score=label if prediction else (label,None); y1,x1,y2,x2=box
        if normalized:x1,x2,y1,y2=x1*w,x2*w,y1*h,y2*h
        x1,y1,x2,y2=map(int,(x1,y1,x2,y2)); text=f"{_name(names,class_id)}: {float(score):.2f}" if prediction else f"GT {_name(names,class_id)}"
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2);cv2.putText(canvas,text,(x1,max(16,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    return canvas
class DetectionInspectionCallback(tf.keras.callbacks.Callback):
    def __init__(self,samples,output_dir,class_names,infer_image,*,gt_normalized,pred_normalized=True):
        super().__init__();self.samples=list(samples[:16]);self.class_names=class_names;self.infer_image=infer_image;self.gt_normalized=gt_normalized;self.pred_normalized=pred_normalized;self.output_dir=output_dir;os.makedirs(output_dir,exist_ok=True)
    def _render(self,sample):
        image=cv2.imread(sample["image_path"])
        if image is None:return None
        canvas=draw_labelled_boxes(image,sample["boxes"],sample["labels"],self.class_names,normalized=self.gt_normalized);detections,_=self.infer_image(self.model,sample["image_path"])
        return cv2.resize(draw_labelled_boxes(canvas,[d["box"] for d in detections],[(d["class_id"],d["score"]) for d in detections],self.class_names,normalized=self.pred_normalized,prediction=True),(320,240),interpolation=cv2.INTER_AREA)
    def on_epoch_end(self,epoch,logs=None):
        panels=[p for sample in self.samples if (p:=self._render(sample)) is not None]
        if not panels:print("[inspection] No validation images could be rendered.");return
        panels.extend(np.zeros_like(panels[0]) for _ in range(16-len(panels)));grid=np.concatenate([np.concatenate(panels[row:row+4],1) for row in range(0,16,4)],0);path=os.path.join(self.output_dir,f"epoch_{epoch+1:03d}.png")
        if not cv2.imwrite(path,grid):raise RuntimeError(f"Could not write inspection image: {path}")
        print(f"[inspection] Saved 4x4 ground-truth/prediction grid: {path}")
