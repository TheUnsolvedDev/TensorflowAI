"""Folder-local labelled validation inspection grid with native mask overlays."""
import os
import cv2
import numpy as np
import tensorflow as tf
from test import render_masks

def _name(names,label):return names[label] if 0<=int(label)<len(names) else f"class_{label}"
def _ground_truth(image,boxes,labels,names):
    canvas=image.copy()
    for box,label in list(zip(boxes,labels))[:20]:
        x1,y1,x2,y2=map(int,box);cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),2);cv2.putText(canvas,f"GT {_name(names,label)}",(x1,max(16,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)
    return canvas
class DetectionInspectionCallback(tf.keras.callbacks.Callback):
    def __init__(self,samples,output_dir,class_names,infer_image):
        super().__init__();self.samples=list(samples[:16]);self.output_dir=output_dir;self.class_names=class_names;self.infer_image=infer_image;os.makedirs(output_dir,exist_ok=True)
    def on_epoch_end(self,epoch,logs=None):
        panels=[]
        for sample in self.samples:
            image=cv2.imread(sample["image_path"])
            if image is None:continue
            canvas=_ground_truth(image,sample["boxes"],sample["labels"],self.class_names);detections,_=self.infer_image(self.model,sample["image_path"],self.class_names)
            panels.append(cv2.resize(render_masks(canvas,detections,self.class_names),(320,240)))
        if not panels:print("[inspection] No validation images could be rendered.");return
        panels.extend(np.zeros_like(panels[0]) for _ in range(16-len(panels)));grid=np.concatenate([np.concatenate(panels[row:row+4],1) for row in range(0,16,4)],0);path=os.path.join(self.output_dir,f"epoch_{epoch+1:03d}.png")
        if not cv2.imwrite(path,grid):raise RuntimeError(f"Could not write inspection image: {path}")
        print(f"[inspection] Saved 4x4 ground-truth/prediction grid: {path}")
