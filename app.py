import matplotlib
matplotlib.use('Agg') 

from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import uuid
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from sklearn.linear_model import LinearRegression

from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
import os, uuid
# ----------------- Flask App Setup -----------------
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

# Ensure full numpy array print
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

default_weights = os.path.join(BASE_DIR, 'yolo', 'runs', 'train', 'laser_seg_model', 'weights', 'best.pt')
app.config['YOLO_WEIGHTS_PATH'] = os.environ.get('YOLO_WEIGHTS_PATH', default_weights)

# ----------------- Point & Mask Processing -----------------

def show(image, instances=None, title=None, visualize_bboxes=True, figsize=(10, 15)):
    image_np = np.asarray(image)
    plt.figure(figsize=figsize)
    plt.imshow(image_np)
    plt.axis('off')
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    colormap = ListedColormap(colors[:len(instances) if instances else 0])
    ax = plt.gca()
    if instances is not None:
        for i, inst in enumerate(instances):
            mask = inst.get('mask')
            bbox = inst.get('bbox')
            points = inst.get('points')
            score = inst.get('score')
            label = inst.get('text_label') or inst.get('label')
            color = colors[i % len(colors)]
            if mask is not None:
                mask_rgba = np.zeros((*mask.shape, 4))
                cmap_color = plt.colormaps.get_cmap(colormap)(i)
                mask_rgba[..., :3] = cmap_color[:3]
                mask_rgba[..., 3] = 0.3 * mask
                plt.imshow(mask_rgba)
                if bbox is None:
                    rows = mask.any(axis=1)
                    cols = mask.any(axis=0)
                    if rows.any() and cols.any():
                        y0, y1 = np.where(rows)[0][[0, -1]]
                        x0, x1 = np.where(cols)[0][[0, -1]]
                        bbox = [x0, y0, x1, y1]
            if points is not None:
                pts = np.array(points)
                poly = patches.Polygon(pts, closed=True, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(poly)
                top = pts[np.argmin(pts[:,1])]
                txt = ''
                if label is not None:
                    txt = str(label)
                if score is not None:
                    txt = f"{label or 'Score'}: {score:.2f}"
                if txt:
                    ax.text(top[0], top[1]-5, txt, color='white', fontsize=10,
                            bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', pad=2))
            elif visualize_bboxes and bbox is not None:
                x0,y0,x1,y1 = bbox
                rect = patches.Rectangle((x0,y0), x1-x0, y1-y0,
                                          linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                txt = ''
                if label is not None:
                    txt = str(label)
                if score is not None:
                    txt = f"{label or 'Score'}: {score:.2f}"
                if txt:
                    ax.text(x0, y0-5, txt, color='white', fontsize=10,
                            bbox=dict(facecolor=color, alpha=0.5, edgecolor='none', pad=2))
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def get_points(mask: np.ndarray,
               eps_start: float=0.001, eps_step: float=0.001, eps_max: float=0.1,
               subpixel_window: int=7, refine_iterations: int=3):
    img = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Контуры не найдены - проверьте маску")
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    circ = (4*np.pi*area)/(peri**2) if peri>0 else 0
    hull = cv2.convexHull(contour)
    hull_peri = cv2.arcLength(hull, True)
    gray = cv2.GaussianBlur(img, (5,5), 0).astype(np.float32)
    pts = hull.squeeze().astype(np.float32)
    for _ in range(refine_iterations):
        crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        cv2.cornerSubPix(gray, pts, (subpixel_window, subpixel_window), (-1,-1), crit)
    if circ>0.85 and len(contour)>=5:
        ellipse = cv2.fitEllipse(contour)
        (x,y),(MA,ma),ang=ellipse
        rad = np.deg2rad(ang)
        major=np.array([np.cos(rad), np.sin(rad)])
        minor=np.array([-np.sin(rad),np.cos(rad)])
        corners=[]
        for dir,len_ in [(major,MA/2),(-major,MA/2),(minor,ma/2),(-minor,ma/2)]:
            pt = np.array([x,y])+ len_*dir
            closest,md=None,float('inf')
            for p in contour[:,0]:
                d=np.linalg.norm(p-pt)
                if d<md: md,d,p = d,d,p
            corners.append(p)
        corners=np.array(corners,dtype=np.float32)
    else:
        eps=eps_start
        best=None
        while eps<=eps_max:
            approx=cv2.approxPolyDP(pts, eps*hull_peri, True)
            if 4<=len(approx)<=6:
                best=approx
                if len(approx)==4: break
            eps+=eps_step
        if best is None or len(best)<4:
            rect=cv2.minAreaRect(pts)
            corners=cv2.boxPoints(rect)
        else:
            arr=best.reshape(-1,2)
            if len(arr)>4:
                curv=[]
                n=len(arr)
                for i in range(n):
                    v1=arr[i-1]-arr[i]
                    v2=arr[(i+1)%n]-arr[i]
                    a=np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-5),-1,1))
                    curv.append(a)
                idx=np.argsort(curv)[-4:]
                corners=arr[idx]
            else:
                corners=arr
    # refine corners subpixel
    final=[]
    for x,y in corners:
        roi=gray[int(y)-7:int(y)+8, int(x)-7:int(x)+8]
        c=cv2.goodFeaturesToTrack(roi,1,0.01,5)
        if c is not None:
            rx,ry=c[0][0]
            final.append([x-7+rx,y-7+ry])
        else:
            final.append([x,y])
    final=np.array(final,dtype=np.float32)
    # order
    s=final.sum(axis=1)
    rect=np.zeros((4,2),dtype=np.float32)
    rect[0]=final[np.argmin(s)]
    rect[2]=final[np.argmax(s)]
    diff=np.diff(final,axis=1)
    rect[1]=final[np.argmin(diff)]
    rect[3]=final[np.argmax(diff)]
    return rect


def get_angle(corners: np.ndarray):
    if len(corners)!=4: raise ValueError("Функция ожидает ровно 4 вершины")
    dx,dy=corners[1][0]-corners[0][0],corners[1][1]-corners[0][1]
    return float((np.degrees(np.arctan2(dy,dx))%360))


def order_points(pts: np.ndarray):
    s=pts.sum(axis=1)
    rect=np.zeros((4,2),dtype="float32")
    rect[0],rect[2]=pts[np.argmin(s)],pts[np.argmax(s)]
    diff=np.diff(pts,axis=1)
    rect[1],rect[3]=pts[np.argmin(diff)],pts[np.argmax(diff)]
    return rect


def warp_object_mask(object_mask: np.ndarray, platform_pts: np.ndarray, output_size=(640,352)):
    src=order_points(platform_pts)
    dst=np.array([[0,0],[output_size[0],0],[output_size[0],output_size[1]],[0,output_size[1]]],dtype="float32")
    M=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(object_mask.astype(np.uint8)*255, M, output_size)


def get_mask_centroid(mask: np.ndarray):
    coords=np.column_stack(np.where(mask>0))
    if coords.size==0: return None
    y,x=coords.mean(axis=0)
    return float(x),float(y)

# pre-trained regressions
pred=np.array([[-19.76,-19.172],[-10.127,-2.8887],[-76.228,5.4855],[-32.025,73.973],[6.0302,-0.92795],
               [34.45,-15.92],[-33.74,-23.405],[-57.606,-52.9],[-8.9866,-53.086],[59.648,-55.487],[-38.73,9.5021]])
true=np.array([[4,6],[0,0],[30,-2],[7,-26],[-5,-1],[-15,5],[8,0],[7,18],[4,18],[-20,19],[10,-4]])
reg_x=LinearRegression().fit(pred[:,[0]],true[:,0])
reg_y=LinearRegression().fit(pred[:,[1]],true[:,1])


def compute_deviation_percentages(object_mask: np.ndarray, platform_pts: np.ndarray, size=(640,352)):
    warped=warp_object_mask(object_mask,platform_pts,size)
    cent=get_mask_centroid(warped)
    if cent is None: raise ValueError("Не найден центр объекта")
    cx,cy=cent
    cx0,cy0=size[0]/2,size[1]/2
    dev_x=(cy-cy0)/(size[1]/2)*100
    dev_y=(cx-cx0)/(size[0]/2)*100
    dx=round(reg_x.predict([[dev_x]]).item())
    dy=round(reg_y.predict([[dev_y]]).item())
    return dx,dy


def get_masks(image: Image.Image):
    from ultralytics import YOLO
    model = YOLO(app.config['YOLO_WEIGHTS_PATH'])
    pred=model.predict(image,verbose=False)[0]
    masks=pred.masks.data.cpu().numpy().astype(int)
    labels=pred.boxes.cls.cpu().numpy().astype(int)
    get_m=lambda lbl: masks[np.where(labels==lbl)[0][0]] if lbl in labels else np.zeros(masks.shape[1:],dtype=int)
    return get_m(0),get_m(1),get_m(2)


def get_data(image_path: str, visualization=True):
    img=Image.open(image_path).resize((640,352))
    edge,platform,obj=get_masks(img)
    pts=get_points(platform)
    combined=(obj+edge).astype(bool).astype(int)
    angle=get_angle(pts)
    coords=compute_deviation_percentages(combined,pts)
    if visualization:
        inst=[{'label':'edge','mask':edge},
              {'label':'platform','mask':platform,'points':pts},
              {'label':'object','mask':obj}]
        show(img,inst)
    return {'angle':angle,'coords':coords,'edge_mask':edge,'platform_mask':platform,'object_mask':obj}

def predict_image(image_path: str):
    return get_data(image_path)

# ----------------- Flask Routes -----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return "No file", 400

    # Generate unique ID
    uid = uuid.uuid4().hex
    
    # Save original as BMP
    bmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}.bmp")
    img = Image.open(file.stream)
    img.save(bmp_path, format='BMP')

    # Process image
    result = get_data(bmp_path, visualization=False)
    angle = round(result['angle'], 2)
    coords = result['coords']
    coords_str = f"({coords[0]}, {coords[1]})"

    # Prepare visualization using your show() function
    vis_img = Image.open(bmp_path).resize((640, 352))
    vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}.png")
    
    # Create instances for visualization
    instances = [
        {'label': 'edge', 'mask': result['edge_mask']},
        {'label': 'platform', 'mask': result['platform_mask'], 'points': get_points(result['platform_mask'])},
        {'label': 'object', 'mask': result['object_mask']}
    ]
    
    # Create figure and visualize
    plt.figure(figsize=(10, 6))
    show(vis_img, 
         instances=instances,
         title=f"Angle: {angle}°, Coords: {coords_str}",
         visualize_bboxes=True,
         figsize=(10, 6))
    
    # Save visualization
    plt.savefig(vis_path)
    plt.close()

    # Create TXT file with results
    txt_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"angle: {result['angle']}\n")
        f.write(f"coords: {result['coords']}\n")
        f.write("edge_mask:\n")
        np.savetxt(f, result['edge_mask'], fmt='%d')
        f.write("\nplatform_mask:\n")
        np.savetxt(f, result['platform_mask'], fmt='%d')
        f.write("\nobject_mask:\n")
        np.savetxt(f, result['object_mask'], fmt='%d')

    # Render result page
    image_url = url_for('static', filename=f"uploads/{uid}.png")
    return render_template('result.html', 
                         uid=uid, 
                         image_url=image_url, 
                         angle=angle, 
                         coords=coords_str)

@app.route('/download/<uid>')
def download(uid):
    txt_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}.txt")
    return send_file(txt_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)